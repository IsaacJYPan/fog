"""Posterior-aware value fine-tune (SEARCH_PLAN §4.2).

The value head is the coupling CONSUMER of joint search: it must price "what does the opponent
now believe about me" or the joint optimization degenerates to independent per-world searches
(fog/search/joint.py, design decision 3). Its posterior-features input columns are untrained by
PPO (search never runs in the training loop), so before any Phase-2 conclusion the head is
fine-tuned here on real (root observation, executed-action posterior, outcome) triples logged by
`fog.search.posterior_log`.

Protocol (the §4.2 regression, binding): train TWO copies of the value MLP from the same starting
weights on the same batches — one seeing the logged posterior features, one seeing zeros (the
posterior-blind control, exactly what the head computes when features are absent). The gate is
held-out MSE: aware < blind. Split is BY GAME (decisions within a game share an outcome — a row
split would leak labels into the holdout). Only the value MLP trains; the trunk/belief/policy are
frozen, so (s_board, c_dyn) are precomputed once and the fine-tune never risks the acting policy.

Checkpoint compatibility: a posterior-blind checkpoint (posterior_dim=0, e.g. the M1.5 run) is
loaded by zero-padding the first Linear of both head MLPs with the 4 posterior columns
(`expand_state_dict_for_posterior`) — zero columns are behavior-preserving at zero features, so
the expanded agent plays byte-identically until the fine-tune gives the new columns weight.

    python -m fog.training.value_finetune --data runs/posterior_log \
        [--ckpt runs/.../ppo_final.pt] --out runs/value_finetune
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from fog.models.heads import N_POSTERIOR_FEATURES

_HEAD_LAYER0 = ("policy_value.policy.0.weight", "policy_value.value.0.weight")


# -------------------------------------------------------------------------- checkpoint expansion
def expand_state_dict_for_posterior(sd: dict, posterior_dim: int = N_POSTERIOR_FEATURES) -> dict:
    """Zero-pad a posterior-blind state dict's head input layers with `posterior_dim` columns.
    Posterior features are concatenated LAST in PolicyValueHead.forward, so the pad goes on the
    right; zeros preserve the checkpoint's behavior exactly while features are zero/absent."""
    import torch

    sd = dict(sd)
    for key in _HEAD_LAYER0:
        w = sd[key]
        sd[key] = torch.cat(
            [w, torch.zeros(w.shape[0], posterior_dim, dtype=w.dtype, device=w.device)], dim=1
        )
    return sd


def load_posterior_agent(cfg: dict, ckpt_path: str | Path, device) -> Any:
    """Build a posterior-aware FogAgent and load `ckpt_path`, expanding it if it was saved
    posterior-blind. Mixed widths other than blind->aware are a real config mismatch: raised."""
    import torch

    from fog.models.agent import FogAgent

    cfg = dict(cfg)
    cfg["model"] = dict(cfg.get("model", {}), use_posterior_features=True)
    agent = FogAgent.from_tables(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    expected = agent.policy_value.policy[0].weight.shape[1]
    got = sd[_HEAD_LAYER0[0]].shape[1]
    if got == expected - N_POSTERIOR_FEATURES:
        sd = expand_state_dict_for_posterior(sd)
    elif got != expected:
        raise ValueError(f"checkpoint head width {got} matches neither posterior-aware "
                         f"({expected}) nor blind ({expected - N_POSTERIOR_FEATURES}) layouts")
    agent.load_state_dict(sd)
    return agent


# ------------------------------------------------------------------------------- fine-tune core
def fit_value_mlp(mlp, x, y, *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 512,
                  seed: int = 0) -> list[float]:
    """Fit one value MLP (output through tanh, matching PolicyValueHead) with MSE on precomputed
    inputs. Returns per-epoch train losses. Deterministic given the seed."""
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    n = x.shape[0]
    losses = []
    mlp.train()
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for lo in range(0, n, int(batch_size)):
            idx = order[lo : lo + int(batch_size)]
            v = torch.tanh(mlp(x[idx])).squeeze(-1)
            loss = torch.nn.functional.mse_loss(v, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / n)
    mlp.eval()
    return losses


def _value_mse(mlp, x, y) -> float:
    import torch

    with torch.no_grad():
        return float(torch.nn.functional.mse_loss(torch.tanh(mlp(x)).squeeze(-1), y))


def _split_by_game(games: np.ndarray, holdout_frac: float, seed: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(games)
    if len(uniq) < 2:
        raise ValueError("need >=2 games for a game-level holdout split")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_hold = max(1, int(round(len(uniq) * holdout_frac)))
    n_hold = min(n_hold, len(uniq) - 1)  # never hold out everything
    hold = set(uniq[:n_hold].tolist())
    mask = np.array([g in hold for g in games])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def finetune_value(agent: Any, data: dict[str, np.ndarray], *, holdout_frac: float = 0.2,
                   epochs: int = 30, lr: float = 1e-3, batch_size: int = 512, seed: int = 0,
                   precompute_batch: int = 256) -> dict:
    """The §4.2 fine-tune + regression on a loaded decision log. Mutates `agent` in place
    (value MLP <- the posterior-aware fit) and returns the report dict, including
    `gate_passed` = aware held-out MSE < blind held-out MSE."""
    import torch

    from fog.search.posterior_log import batch_from_rows

    if not agent.use_posterior_features or agent.policy_value.posterior_dim == 0:
        raise ValueError("finetune_value needs a posterior-aware agent "
                         "(model.use_posterior_features: true)")
    device = next(agent.parameters()).device
    agent.eval()

    n = data["outcome"].shape[0]
    train_idx, hold_idx = _split_by_game(data["game"], holdout_frac, seed)

    # frozen trunk: precompute head inputs once (only the value MLP trains)
    xs = []
    for lo in range(0, n, int(precompute_batch)):
        idx = np.arange(lo, min(lo + int(precompute_batch), n))
        with torch.no_grad():
            s_board, c_dyn = agent.head_inputs(batch_from_rows(data, idx, device))
        xs.append(torch.cat([s_board, c_dyn], dim=-1))
    x_base = torch.cat(xs)
    feats = torch.as_tensor(data["posterior_features"], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["outcome"], dtype=torch.float32, device=device)
    x_aware = torch.cat([x_base, feats], dim=-1)
    x_blind = torch.cat([x_base, torch.zeros_like(feats)], dim=-1)

    fit = dict(epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    aware = copy.deepcopy(agent.policy_value.value)
    blind = copy.deepcopy(agent.policy_value.value)
    aware_losses = fit_value_mlp(aware, x_aware[train_idx], y[train_idx], **fit)
    blind_losses = fit_value_mlp(blind, x_blind[train_idx], y[train_idx], **fit)

    report = {
        "n_decisions": int(n), "n_train": len(train_idx), "n_holdout": len(hold_idx),
        "n_games": int(len(np.unique(data["game"]))),
        "aware_holdout_mse": _value_mse(aware, x_aware[hold_idx], y[hold_idx]),
        "blind_holdout_mse": _value_mse(blind, x_blind[hold_idx], y[hold_idx]),
        "aware_train_loss": aware_losses[-1], "blind_train_loss": blind_losses[-1],
        **fit,
    }
    report["gate_passed"] = report["aware_holdout_mse"] < report["blind_holdout_mse"]
    agent.policy_value.value.load_state_dict(aware.state_dict())
    return report


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent
    from fog.search.posterior_log import load_decisions

    ap = argparse.ArgumentParser(description="Posterior-aware value fine-tune (§4.2).")
    ap.add_argument("--data", type=Path, required=True,
                    help="posterior_log output dir (or decisions.npz)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="agent checkpoint (blind ones expanded); default: fresh (smoke only)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs/value_finetune"))
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.ckpt:
        agent = load_posterior_agent(cfg, args.ckpt, device)
    else:
        cfg.setdefault("model", {})["use_posterior_features"] = True
        agent = FogAgent.from_tables(cfg).to(device)
        print("[value_finetune] WARNING: no --ckpt — fine-tuning a FRESH agent (smoke use only).",
              flush=True)

    report = finetune_value(
        agent, load_decisions(args.data), holdout_frac=args.holdout, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, seed=args.seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), args.out / "value_finetune.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    gate = "PASSED" if report["gate_passed"] else "FAILED"
    print(f"[value_finetune] gate {gate}: aware {report['aware_holdout_mse']:.4f} vs "
          f"blind {report['blind_holdout_mse']:.4f} (held-out MSE, {report['n_holdout']} rows) "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
