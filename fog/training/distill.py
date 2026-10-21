"""Distill logged search distributions into policy and value heads.

`load_decisions` reads flattened v1 logs from posterior_log; `load_v2_decisions`
converts pb_log rows to the same column layout and drops cap-truncated games.
For v2, posterior features summarize P(k | a) proportional to
(sum_j rho[k, j]) * sigma1[k, a].

The policy minimizes KL(search || policy) with posterior features zeroed to match
its deployment inputs. The value head fits outcomes using logged posterior
features. `scope="heads"` freezes the trunk and belief models and caches head
inputs. `scope="full"` also trains the backbone; belief models then need separate
retraining because their inputs have changed.

Holdout splits are by game. Reports compare held-out KL by turn bucket before
and after training. Dropout is disabled for these fixed-distribution targets.
Posterior-blind checkpoints are zero-pad expanded by `load_posterior_agent`.

    python -m fog.training.distill --data runs/posterior_log_m15 \
        --ckpt runs/value_finetune_m15/value_finetune.pt --out runs/distill_g1"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fog.encoding.dims import DIMS

TURN_BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 10**9))
SCOPES = ("heads", "full")


# ------------------------------------------------------------------------------ v2 log reading
def _own_posterior_features(post_own: np.ndarray, k_index: int) -> np.ndarray:
    """Four concentration features of an own-side posterior: [normalized entropy, mass on the
    true candidate, top-1 mass, effective-candidate fraction]. Kept local so the log
    reader does not depend on the search value implementation."""
    p = np.asarray(post_own, dtype=np.float64)
    k = len(p)
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    h_norm = h / np.log(k) if k > 1 else 0.0
    return np.array([h_norm, float(p[k_index]), float(p.max()) if k else 0.0,
                     float(np.exp(h)) / k if k else 0.0], dtype=np.float32)


def _v2_row_columns(row: dict) -> dict[str, np.ndarray]:
    """One pb_log v2 row -> the v1 column contract (posterior_log._row's schema)."""
    from fog.encoding.history import pad_or_truncate_front
    from fog.search.posterior_log import MASK_PREFIX

    obs = row["obs"]
    rho = np.asarray(row["rho"], dtype=np.float64)
    margin = rho.sum(axis=1)
    col = int(row.get("action_label_col", -1))
    if col >= 0:
        post = margin * np.asarray(row["sigma1"], dtype=np.float64)[:, col]
        z = post.sum()
        post = post / z if z > 0 else margin
    else:  # pre-action root margin (older logs without the executed-label column)
        post = margin / margin.sum() if margin.sum() > 0 else margin
    history_ids = getattr(obs, "history_ids", None)
    history_state = getattr(obs, "history_state", None)
    out = {
        "entities": obs.entities.astype(np.float32),
        "entity_ids": obs.entity_ids.astype(np.int64),
        "field": obs.field.astype(np.float32),
        "history": pad_or_truncate_front(obs.history).astype(np.float32),
        "history_ids": pad_or_truncate_front(
            history_ids if history_ids is not None
            else np.zeros((len(obs.history), DIMS.hist_id_fields), dtype=np.int64)
        ).astype(np.int64),
        "history_state": pad_or_truncate_front(
            history_state if history_state is not None
            else np.zeros((len(obs.history), DIMS.hist_state), dtype=np.float32)
        ).astype(np.float32),
        "history_len": np.int64(min(
            getattr(
                obs, "history_len",
                int(np.any(np.asarray(obs.history) != 0, axis=-1).sum()),
            ),
            DIMS.history_cap,
        )),
        "active_idx": (obs.active_idx if obs.active_idx is not None
                       else np.array([0, 6], dtype=np.int64)),
        "legality_mask": obs.legality_mask.astype(bool),
        "opp_tokens": (obs.opp_revealed or {}).get(
            "tokens", np.full((6, 15), -1, dtype=np.int64)).astype(np.int64),
        "posterior_features": _own_posterior_features(post, int(row["true_index"])),
        "search_dist": np.asarray(row["action_dist"], dtype=np.float32),
        "action": np.int64(row["action"]),
        "turn": np.int64(row["turn"]),
        "outcome": np.float32(row["outcome"]),
    }
    for fam, m in ((obs.opp_revealed or {}).get("masks") or {}).items():
        out[MASK_PREFIX + fam] = m.astype(bool)
    return out


def load_v2_decisions(path: str | Path) -> dict[str, np.ndarray]:
    """pb_log v2 log(s) -> the same column dict `load_decisions` returns. Accepts one .pt file
    or a directory of shards (pb_datagen layout: shard_*.pt, else every .pt in the directory).
    Game ids are re-keyed globally across shards (shards restart game_id at 0), and rows from
    cap-truncated games are dropped (their outcome 0 is an artifact — the module docstring)."""
    import torch

    p = Path(path)
    files = ([p] if p.is_file()
             else sorted(p.glob("shard_*.pt")) or sorted(p.glob("*.pt")))
    if not files:
        raise FileNotFoundError(f"no v2 log .pt files under {path}")
    rows: list[dict[str, np.ndarray]] = []
    games: list[tuple[int, int]] = []
    dropped = 0
    for si, f in enumerate(files):
        for r in torch.load(f, weights_only=False)["rows"]:
            if r.get("truncated"):
                dropped += 1
                continue
            rows.append(_v2_row_columns(r))
            games.append((si, int(r["game_id"])))
    if not rows:
        raise RuntimeError(f"no usable v2 rows under {path} (dropped {dropped} truncated)")
    key_ids = {g: i for i, g in enumerate(dict.fromkeys(games))}
    data = {k: np.stack([r[k] for r in rows]) for k in sorted(rows[0])}
    data["game"] = np.asarray([key_ids[g] for g in games], dtype=np.int64)
    if dropped:
        print(f"[distill] dropped {dropped} rows from cap-truncated games "
              f"(outcome 0 is an artifact there)", flush=True)
    return data


def kl_search_policy(search_dist, log_probs):
    """Per-row KL(search ‖ π) in nats: [N,14] search distributions x [N,14] policy log-probs.
    log-probs are clamped at -30 so a support mismatch (search mass on an action the policy has
    masked away — impossible when both used the same legality mask) degrades loudly, not to inf.
    p is clamped before its log (never 0·(-inf) — the ppo_update entropy NaN lesson)."""
    import torch

    p = search_dist
    logq = log_probs.clamp(min=-30.0)
    return (p * (torch.log(p.clamp(min=1e-12)) - logq)).sum(-1)


def _bucket_label(lo: int, hi: int) -> str:
    return f"t{lo}+" if hi >= 10**9 else f"t{lo}-{hi}"


def kl_by_turn(kl_rows: np.ndarray, turns: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in TURN_BUCKETS:
        m = (turns >= lo) & (turns <= hi)
        if m.any():
            out[_bucket_label(lo, hi)] = float(kl_rows[m].mean())
    return out


def distill(agent: Any, data: dict[str, np.ndarray], *, scope: str = "heads",
            holdout_frac: float = 0.2, epochs: int = 20, lr: float = 3e-4,
            batch_size: int = 256, value_coef: float = 0.5, seed: int = 0,
            precompute_batch: int = 256) -> dict:
    """One distillation generation on a loaded decision log. Mutates `agent` in place and returns
    the report dict (held-out KL/value/top-1, before vs after, by turn)."""
    import torch

    from fog.search.posterior_log import batch_from_rows
    from fog.training.value_finetune import _split_by_game

    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if not agent.use_posterior_features or agent.policy_value.posterior_dim == 0:
        raise ValueError("distill needs a posterior-aware agent (model.use_posterior_features: "
                         "true) — the value loss trains at logged posterior features")
    if getattr(agent, "oracle", "none") not in (None, "none"):
        raise ValueError("distill targets the deployed (non-oracle) policy")
    device = next(agent.parameters()).device
    agent.eval()  # dropout off everywhere — targets are fixed distributions (module docstring)

    n = data["outcome"].shape[0]
    train_idx, hold_idx = _split_by_game(data["game"], holdout_frac, seed)
    search_dist = torch.as_tensor(data["search_dist"], dtype=torch.float32, device=device)
    feats = torch.as_tensor(data["posterior_features"], dtype=torch.float32, device=device)
    outcome = torch.as_tensor(data["outcome"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(data["legality_mask"], dtype=torch.bool, device=device)
    turns = data["turn"]

    # ---- head inputs: cached once (heads scope) or recomputed with grad (full scope) ----
    cache_s, cache_b = None, None
    if scope == "heads":
        ss, bs = [], []
        for lo in range(0, n, int(precompute_batch)):
            idx = np.arange(lo, min(lo + int(precompute_batch), n))
            with torch.no_grad():
                s, b = agent.head_inputs(batch_from_rows(data, idx, device))
            ss.append(s)
            bs.append(b)
        cache_s = torch.cat(ss)
        cache_b = torch.cat(bs) if bs[0] is not None else None

    def head_io(idx: np.ndarray, grad: bool):
        if cache_s is not None:
            return cache_s[idx], (cache_b[idx] if cache_b is not None else None)
        batch = batch_from_rows(data, idx, device)
        if grad:
            return agent.head_inputs(batch)
        with torch.no_grad():
            return agent.head_inputs(batch)

    def losses(idx: np.ndarray, grad: bool):
        """(kl_rows [B], value_se_rows [B]) — policy at zero features, value at logged features."""
        s, b = head_io(idx, grad)
        t = torch.as_tensor(idx, dtype=torch.long, device=device)
        log_probs, _ = agent.policy_value(s, mask[t], b, posterior_features=None)
        _, v = agent.policy_value(s, mask[t], b, posterior_features=feats[t])
        return kl_search_policy(search_dist[t], log_probs), (v - outcome[t]) ** 2

    @torch.no_grad()
    def evaluate(idx: np.ndarray) -> dict:
        kls, ses, top1 = [], [], []
        for lo in range(0, len(idx), int(precompute_batch)):
            part = idx[lo : lo + int(precompute_batch)]
            kl, se = losses(part, grad=False)
            s, b = head_io(part, grad=False)
            t = torch.as_tensor(part, dtype=torch.long, device=device)
            lp, _ = agent.policy_value(s, mask[t], b, posterior_features=None)
            top1.append((lp.argmax(-1) == search_dist[t].argmax(-1)).float().cpu().numpy())
            kls.append(kl.cpu().numpy())
            ses.append(se.cpu().numpy())
        kls = np.concatenate(kls)
        return {"kl": float(kls.mean()), "kl_by_turn": kl_by_turn(kls, turns[idx]),
                "value_mse": float(np.concatenate(ses).mean()),
                "top1_agreement": float(np.concatenate(top1).mean())}

    before = evaluate(hold_idx)

    if scope == "heads":
        params = list(agent.policy_value.parameters())
    else:
        params = list(agent.backbone.parameters()) + list(agent.policy_value.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    train_losses = []
    for _ in range(int(epochs)):
        order = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        total = 0.0
        for lo in range(0, len(order), int(batch_size)):
            part = order[lo : lo + int(batch_size)]
            kl, se = losses(part, grad=True)
            loss = kl.mean() + float(value_coef) * se.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(part)
        train_losses.append(total / len(order))

    after = evaluate(hold_idx)
    report = {
        "n_decisions": int(n), "n_train": len(train_idx), "n_holdout": len(hold_idx),
        "n_games": len(np.unique(data["game"])), "scope": scope,
        "epochs": int(epochs), "lr": float(lr), "batch_size": int(batch_size),
        "value_coef": float(value_coef), "seed": int(seed),
        "train_loss_first": train_losses[0], "train_loss_last": train_losses[-1],
        "holdout_before": before, "holdout_after": after,
        "improved": after["kl"] < before["kl"],
    }
    return report


def print_report(report: dict) -> None:
    b, a = report["holdout_before"], report["holdout_after"]
    print(f"[distill] scope={report['scope']} | {report['n_train']} train / "
          f"{report['n_holdout']} holdout rows ({report['n_games']} games)")
    print(f"[distill] held-out KL(search‖π) {b['kl']:.4f} -> {a['kl']:.4f} "
          f"({'improved' if report['improved'] else 'NOT improved'}) | "
          f"top-1 agreement {b['top1_agreement']:.3f} -> {a['top1_agreement']:.3f} | "
          f"value MSE {b['value_mse']:.4f} -> {a['value_mse']:.4f}")
    buckets = sorted(set(b["kl_by_turn"]) | set(a["kl_by_turn"]))
    line = "  ".join(
        f"{t}:{b['kl_by_turn'].get(t, float('nan')):.3f}->{a['kl_by_turn'].get(t, float('nan')):.3f}"
        for t in buckets
    )
    print(f"[distill] KL by turn: {line}")


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent
    from fog.search.posterior_log import load_decisions

    ap = argparse.ArgumentParser(description="Distill logged search action distributions.")
    ap.add_argument("--data", type=Path, required=True,
                    help="v1: posterior_log dir (or decisions.npz); v2: a pb_log/pb_datagen "
                         ".pt file or shard directory (auto-detected)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="agent checkpoint (posterior-aware weights are preferred; "
                         "blind ones are expanded); default: fresh (smoke only)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs/distill"))
    ap.add_argument("--scope", choices=SCOPES, default="heads")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.ckpt:
        from fog.training.value_finetune import load_posterior_agent

        agent = load_posterior_agent(cfg, args.ckpt, device)
    else:
        cfg.setdefault("model", {})["use_posterior_features"] = True
        agent = FogAgent.from_tables(cfg).to(device)
        print("[distill] WARNING: no --ckpt — distilling into a FRESH agent (smoke use only).",
              flush=True)

    p = Path(args.data)
    is_v2 = (p.suffix == ".pt"
             or (p.is_dir() and not (p / "decisions.npz").exists()
                 and (any(p.glob("shard_*.pt")) or any(p.glob("*.pt")))))
    data = load_v2_decisions(p) if is_v2 else load_decisions(p)
    print(f"[distill] loaded {'v2 (pb_log)' if is_v2 else 'v1 (posterior_log)'} format: "
          f"{data['outcome'].shape[0]} rows", flush=True)
    report = distill(
        agent, data, scope=args.scope, holdout_frac=args.holdout,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        value_coef=args.value_coef, seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), args.out / "distill.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"[distill] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
