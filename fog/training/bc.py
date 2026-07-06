"""Training stage 1 — behavior cloning (spec §5).

Objective:
    CE(policy, human action)
  + belief losses (censored)                     [opponent-team belief head]
  + value MSE against final outcome (γ=1)
  + self-belief losses (censored, own team)      [off by default in BC — see note]

Self-belief note: its correct input is the OPPONENT's view of the acting player (a mirror encoding),
which the replay shards don't carry. So in BC `self_belief_coef` defaults to 0; the self-belief /
leakage predictor is trained in self-play (§7 / training/leakage) where both perspectives are natural.
The self-belief LABELS are still parsed and stored for that path.

Optimizer AdamW, lr 3e-4 cosine to 3e-5, batch 512, ~10 epochs, mixed precision on CUDA.
Go/no-go (§5): action top-1 vs strong players ≥45%; belief item top-1 before reveal ≥55%; win-rate vs
SimpleHeuristics ≥65% (the last two are measured by the eval suite once a checkpoint exists).

    python -m fog.training.bc --config configs/bc_ou.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from fog.config import config_hash
from fog.training import losses


def bc_loss(agent, batch, loss_cfg: dict) -> tuple[torch.Tensor, dict]:
    out = agent(batch)
    action_ce = losses.masked_cross_entropy(
        out.log_probs, batch["action"], torch.ones_like(batch["action"], dtype=torch.bool)
    )
    value = losses.value_loss(out.value, batch["outcome"])
    belief = out.log_probs.new_zeros(())
    if out.belief is not None and loss_cfg.get("belief_coef", 1.0) > 0:
        belief = losses.belief_loss(out.belief.logits, batch["belief_targets"], batch["belief_valid"])
    self_belief = out.log_probs.new_zeros(())
    if out.self_belief is not None and loss_cfg.get("self_belief_coef", 0.0) > 0:
        self_belief = losses.belief_loss(
            out.self_belief.logits, batch["self_belief_targets"], batch["self_belief_valid"]
        )
    total = (
        action_ce
        + loss_cfg.get("belief_coef", 1.0) * belief
        + loss_cfg.get("value_coef", 0.5) * value
        + loss_cfg.get("self_belief_coef", 0.0) * self_belief
    )
    with torch.no_grad():
        top1 = (out.log_probs.argmax(-1) == batch["action"]).float().mean().item()
    metrics = {
        "loss": float(total), "action_ce": float(action_ce), "belief": float(belief),
        "value": float(value), "action_top1": top1,
    }
    return total, metrics


def train(cfg: dict) -> dict:
    from fog.models.agent import FogAgent
    from fog.training.dataset import ReplayDataset, StreamingReplayDataset

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    device = torch.device("cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")

    bc_cfg = cfg.get("bc", {})
    loss_cfg = cfg.get("loss", {})
    manifest = bc_cfg.get("manifest", "data/parsed/manifest.parquet")
    batch_size = int(bc_cfg.get("batch_size", 512))
    epochs = int(bc_cfg.get("epochs", 10))
    lr0 = float(bc_cfg.get("lr", 3e-4))
    lr1 = float(bc_cfg.get("lr_final", 3e-5))
    limit = bc_cfg.get("limit_rows")
    ckpt_dir = Path(cfg.get("log", {}).get("dir", "runs/bc")) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if bc_cfg.get("streaming", False):  # 150k-game corpora: shard-streaming, no full concat
        ds = StreamingReplayDataset(manifest, shard_buffer=int(bc_cfg.get("shard_buffer", 64)))
    else:
        ds = ReplayDataset(manifest, limit_rows=limit)
    print(f"BC dataset: {len(ds)} decision rows  (config_hash={config_hash(cfg)})")
    agent = FogAgent.from_tables(cfg).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=lr0)
    steps_per_epoch = max(1, math.ceil(len(ds) / batch_size))
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    metrics: dict = {}
    step = 0
    for epoch in range(epochs):
        agent.train()
        for batch in ds.batches(batch_size, shuffle=True, seed=seed + epoch, device=device):
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    total, metrics = bc_loss(agent, batch, loss_cfg)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                total, metrics = bc_loss(agent, batch, loss_cfg)
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
            sched.step()
            step += 1
        print(
            f"[bc] epoch {epoch+1}/{epochs}  loss={metrics['loss']:.3f}  "
            f"action_ce={metrics['action_ce']:.3f}  top1={metrics['action_top1']:.3f}  "
            f"belief={metrics['belief']:.3f}  value={metrics['value']:.3f}",
            flush=True,
        )
        torch.save(agent.state_dict(), ckpt_dir / f"bc_epoch{epoch+1}.pt")

    final = ckpt_dir / "bc_final.pt"
    torch.save(agent.state_dict(), final)
    print(f"[bc] done -> {final}", flush=True)
    return metrics


def main() -> None:
    from fog.config import load_config_with_args

    ap = argparse.ArgumentParser(description="FOG behavior cloning (§5).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", help="dotted overrides, e.g. bc.epochs=3")
    args = ap.parse_args()
    train(load_config_with_args(args.config, args.overrides))


if __name__ == "__main__":
    main()
