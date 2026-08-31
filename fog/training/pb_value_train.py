"""PBValueNet outcome training on two-sided decision logs (ARCHITECTURE.md S2).

The training target: at each public leaf actually reached, the net's per-pair table should price
the TRUE cell — V[k*, j*] ≈ outcome — one labeled cell per example; off-diagonal cells are shaped by
cross-example generalization (recorded G3 honesty note; the exact treatment — solver continuation
values at queried beliefs — is the gated bootstrapping step, Phase I.6).

The public leaf and its belief are recorded together after the played actions and their truthful
mechanical evidence. Hypothetical solver beliefs remain evaluation inputs only until a matching
hypothetical leaf and a valid continuation target are available. Reusing the root board with those
beliefs is forbidden: it creates internally inconsistent examples and encourages the net to ignore
belief.

Row filtering (G3 spec): rows with j* = -1 (off-pool or truncated-out opponent) and rows of
cap-truncated games (outcome 0 is an artifact) are dropped at dataset build.

The agent's trunk supplies private-state and public-context tensors per leaf — computed once and
trunk is FROZEN here; only PBValueNet trains). Rows carry per-decision candidate sets, so the v1
trainer runs one forward per row (L = 1); bucketing rows that share candidate sets is the known
optimization if profiling demands it.

    python -m fog.training.pb_value_train --data runs/pb_datagen_a \
        --ckpt runs/g0_v2_champion/checkpoints/ppo_final.pt --out runs/pb_value_g1
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_pb_value_rows(path: str | Path) -> list[dict]:
    """Build aligned played-leaf outcome examples from v2 log shards.

    Legacy rows that contain only a root observation and hypothetical belief variants are rejected
    rather than silently reused as value examples.
    """
    import torch

    p = Path(path)
    files = ([p] if p.is_file()
             else sorted(p.glob("shard_*.pt")) or sorted(p.glob("*.pt")))
    if not files:
        raise FileNotFoundError(f"no v2 log .pt files under {path}")
    out: list[dict] = []
    game_key: dict[tuple[int, int], int] = {}
    dropped_j = dropped_trunc = dropped_unaligned = 0
    for si, f in enumerate(files):
        for r in torch.load(f, weights_only=False)["rows"]:
            if r.get("truncated"):
                dropped_trunc += 1
                continue
            if r.get("value_obs") is None or r.get("value_rho") is None:
                dropped_unaligned += 1
                continue
            if int(r.get("value_opp_true_index", -1)) < 0:
                dropped_j += 1
                continue
            belief = np.asarray(r["value_rho"], dtype=np.float64)
            own_tokens = np.asarray(r["value_own_tokens"], dtype=np.int64)
            opp_tokens = np.asarray(r["value_opp_tokens"], dtype=np.int64)
            if belief.shape != (len(own_tokens), len(opp_tokens)) or belief.sum() <= 0.0:
                raise ValueError(f"misaligned value belief in {f}: {belief.shape}")
            key = (si, int(r["game_id"]))
            game_key.setdefault(key, len(game_key))
            out.append({
                "obs": r["value_obs"],
                "root_obs": r.get("obs"),
                "own_tokens": own_tokens,
                "opp_tokens": opp_tokens,
                "k_star": int(r["value_true_index"]),
                "j_star": int(r["value_opp_true_index"]),
                "outcome": float(r["outcome"]),
                "belief": belief / belief.sum(),
                "target_kind": r.get("value_target_kind"),
                "one_sided": r.get("labels2") == ["<observe>"],
                "game": game_key[key],
            })
    if not out:
        raise RuntimeError(f"no usable rows under {path} "
                           f"(dropped {dropped_j} j*=-1, {dropped_trunc} truncated, "
                           f"{dropped_unaligned} without aligned public leaves)")
    if dropped_j or dropped_trunc or dropped_unaligned:
        print(f"[pb_value_train] dropped {dropped_j} rows with j*=-1 and {dropped_trunc} "
              f"cap-truncated rows and {dropped_unaligned} unaligned legacy/terminal rows",
              flush=True)
    return out


def public_only_ablation_rows(rows: list[dict]) -> list[dict]:
    """Destroy candidate identity and belief information while preserving each public board.

    This creates the G4 public-history-only control. It is trained from scratch with the same
    optimizer, split, seed, and architecture as the full model; merely zeroing inputs at evaluation
    would test robustness to an out-of-distribution corruption rather than whether the extra inputs
    improve generalization.
    """
    out = []
    for row in rows:
        copy = dict(row)
        copy["own_tokens"] = np.full_like(row["own_tokens"], -1)
        copy["opp_tokens"] = np.full_like(row["opp_tokens"], -1)
        belief = np.ones_like(row["belief"], dtype=np.float64)
        copy["belief"] = belief / belief.sum()
        out.append(copy)
    return out


def train_pb_value(agent: Any, net: Any, rows: list[dict], *, epochs: int = 10,
                   lr: float = 1e-3, batch_size: int = 64, holdout_frac: float = 0.2,
                   seed: int = 0, trunk_batch: int = 256, device=None) -> dict:
    """Train `net` (PBValueNet) on `rows`; the agent's trunk is frozen context. Mutates `net`;
    returns the report (held-out MSE at the ROOT belief before/after, the predict-mean-outcome
    baseline, and train curves). Holdout is split by game (rows share outcomes within a game)."""
    import torch

    from fog.env.engine_runner import _device, _stack_obs
    from fog.training.value_finetune import _split_by_game

    device = device or _device(agent)
    net = net.to(device)
    agent.eval()

    # ---- frozen trunk context, once per row ---------------------------------------------
    sb_parts, cd_parts = [], []
    for lo in range(0, len(rows), int(trunk_batch)):
        chunk = rows[lo: lo + int(trunk_batch)]
        with torch.no_grad():
            sb, cd = agent.head_inputs(_stack_obs([r["obs"] for r in chunk], device))
        sb_parts.append(sb)
        cd_parts.append(cd)
    private_state = torch.cat(sb_parts)
    public_context = torch.cat(cd_parts)

    games = np.asarray([r["game"] for r in rows], dtype=np.int64)
    train_idx, hold_idx = _split_by_game(games, holdout_frac, seed)
    outcome = torch.tensor([r["outcome"] for r in rows], dtype=torch.float32, device=device)

    def _forward(i: int) -> torch.Tensor:
        r = rows[i]
        rho = torch.as_tensor(r["belief"][None], dtype=torch.float32, device=device)
        rho = rho / rho.sum()
        v = net(private_state[i][None], public_context[i][None],
                torch.as_tensor(r["own_tokens"], device=device),
                torch.as_tensor(r["opp_tokens"], device=device), rho)
        return v[0, r["k_star"], r["j_star"]]

    @torch.no_grad()
    def evaluate(idx: np.ndarray) -> float:
        ses = [(float(_forward(int(i))) - float(outcome[int(i)])) ** 2
               for i in idx]
        return float(np.mean(ses))

    base = float(outcome[torch.as_tensor(hold_idx, device=device)].mean())
    baseline_mse = float(((outcome[torch.as_tensor(hold_idx, device=device)] - base) ** 2)
                         .mean())
    before = evaluate(hold_idx)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(int(epochs)):
        order = train_idx[rng.permutation(len(train_idx))]
        total = 0.0
        for lo in range(0, len(order), int(batch_size)):
            part = order[lo: lo + int(batch_size)]
            preds, targs = [], []
            for i in part:
                preds.append(_forward(int(i)))
                targs.append(outcome[int(i)])
            loss = ((torch.stack(preds) - torch.stack(targs)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(part)
        curves.append(total / len(order))

    after = evaluate(hold_idx)
    return {
        "n_rows": len(rows), "n_train": len(train_idx), "n_holdout": len(hold_idx),
        "n_games": len(np.unique(games)),
        "belief_variants_mean": 1.0,
        "value_target_kind": "played_leaf_outcome",
        "epochs": int(epochs), "lr": float(lr), "batch_size": int(batch_size),
        "seed": int(seed),
        "holdout_mse_before": before, "holdout_mse_after": after,
        "holdout_mse_predict_mean_baseline": baseline_mse,
        "train_loss_first": curves[0], "train_loss_last": curves[-1],
        "improved_over_baseline": after < baseline_mse,
    }


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent
    from fog.models.pb_value import PBValueNet

    ap = argparse.ArgumentParser(description="PBValueNet outcome training (Phase G3).")
    ap.add_argument("--data", type=Path, required=True, help="v2 shard file or directory")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="agent checkpoint for the frozen trunk (the G0 champion); "
                         "default: random weights (smoke only)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs/pb_value_train"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--public-only-ablation",
        action="store_true",
        help="train the matched G4 control with candidate identities and belief weights removed",
    )
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    if args.ckpt:
        agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        print("[pb_value_train] WARNING: no --ckpt — random trunk (smoke use only).", flush=True)
    # Dims come from the constructed agent, not config keys: the trunk's actual output widths
    # are what PBValueNet consumes (a config-side mismatch would crash at the first forward).
    net = PBValueNet.from_tables(state_dim=int(agent.backbone.state_dim),
                                 dyn_dim=int(agent.backbone.dyn_dim))

    rows = build_pb_value_rows(args.data)
    if args.public_only_ablation:
        rows = public_only_ablation_rows(rows)
    report = train_pb_value(agent, net, rows, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, holdout_frac=args.holdout,
                            seed=args.seed, device=device)
    report["input_mode"] = "public_only_ablation" if args.public_only_ablation else "full"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out / "pb_value.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()}, indent=2))
    print(f"[pb_value_train] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
