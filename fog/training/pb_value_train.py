"""PBValueNet outcome training on v2 decision logs (PROPOSAL_IMPL_PLAN Phase G3; proposal §9.3).

The training target: at each played decision the net's per-pair table should price the TRUE cell
— V[k*, j*] ≈ outcome — one labeled cell per example; off-diagonal cells are shaped only by
cross-example generalization (recorded G3 honesty note; the exact treatment — solver continuation
values at queried beliefs — is the gated bootstrapping step, Phase I.6).

Belief-input coverage (§9.3): each training step draws the example's conditioning belief from
{root rho, the G1 random-iteration mid-solve posteriors (`sample_beliefs`), the final-sigma
`queried_beliefs`} so the net trains across the query distribution `pb_value_fn` will actually
serve — outcome labels only exist along played lines, but the solver queries early near-uniform
and action-conditioned beliefs the final averages never revisit. Attaching one outcome to several
belief variants of a state is the recorded coverage approximation.

Row filtering (G3 spec): rows with j* = -1 (off-pool or truncated-out opponent) and rows of
cap-truncated games (outcome 0 is an artifact) are dropped at dataset build.

The agent's trunk supplies (s_board, c_dyn) per root observation — computed ONCE and cached (the
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
    """v2 shard file/dir -> trainer rows: {obs, own_tokens [K,6,9], opp_tokens [J,6,9],
    k_star, j_star, outcome, beliefs: list of [K, J] arrays (root rho first), game}. Games are
    re-keyed globally across shards; j* = -1 and cap-truncated rows are dropped (module
    docstring)."""
    import torch

    p = Path(path)
    files = ([p] if p.is_file()
             else sorted(p.glob("shard_*.pt")) or sorted(p.glob("*.pt")))
    if not files:
        raise FileNotFoundError(f"no v2 log .pt files under {path}")
    out: list[dict] = []
    game_key: dict[tuple[int, int], int] = {}
    dropped_j = dropped_trunc = 0
    for si, f in enumerate(files):
        for r in torch.load(f, weights_only=False)["rows"]:
            if r.get("truncated"):
                dropped_trunc += 1
                continue
            if int(r["opp_true_index"]) < 0:
                dropped_j += 1
                continue
            beliefs = [np.asarray(r["rho"], dtype=np.float64)]
            for coll in ("sample_beliefs", "queried_beliefs"):
                for entry in r.get(coll) or []:
                    post = np.asarray(entry[2], dtype=np.float64)
                    if post.shape == beliefs[0].shape and post.sum() > 0:
                        beliefs.append(post)
            key = (si, int(r["game_id"]))
            game_key.setdefault(key, len(game_key))
            out.append({
                "obs": r["obs"],
                "own_tokens": np.asarray(r["own_tokens"], dtype=np.int64),
                "opp_tokens": np.asarray(r["opp_tokens"], dtype=np.int64),
                "k_star": int(r["true_index"]), "j_star": int(r["opp_true_index"]),
                "outcome": float(r["outcome"]),
                "beliefs": beliefs,
                "game": game_key[key],
            })
    if not out:
        raise RuntimeError(f"no usable rows under {path} "
                           f"(dropped {dropped_j} j*=-1, {dropped_trunc} truncated)")
    if dropped_j or dropped_trunc:
        print(f"[pb_value_train] dropped {dropped_j} rows with j*=-1 and {dropped_trunc} "
              f"cap-truncated rows", flush=True)
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
    s_board = torch.cat(sb_parts)
    c_dyn = torch.cat(cd_parts)

    games = np.asarray([r["game"] for r in rows], dtype=np.int64)
    train_idx, hold_idx = _split_by_game(games, holdout_frac, seed)
    outcome = torch.tensor([r["outcome"] for r in rows], dtype=torch.float32, device=device)

    def _forward(i: int, belief: np.ndarray) -> torch.Tensor:
        r = rows[i]
        rho = torch.as_tensor(belief[None], dtype=torch.float32, device=device)
        rho = rho / rho.sum()
        v = net(s_board[i][None], c_dyn[i][None],
                torch.as_tensor(r["own_tokens"], device=device),
                torch.as_tensor(r["opp_tokens"], device=device), rho)
        return v[0, r["k_star"], r["j_star"]]

    @torch.no_grad()
    def evaluate(idx: np.ndarray) -> float:
        ses = [(float(_forward(int(i), rows[int(i)]["beliefs"][0])) - float(outcome[int(i)])) ** 2
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
                bs = rows[int(i)]["beliefs"]
                preds.append(_forward(int(i), bs[rng.integers(len(bs))]))
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
        "n_games": int(len(np.unique(games))),
        "belief_variants_mean": float(np.mean([len(r["beliefs"]) for r in rows])),
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
    report = train_pb_value(agent, net, rows, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, holdout_frac=args.holdout,
                            seed=args.seed, device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out / "pb_value.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()}, indent=2))
    print(f"[pb_value_train] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
