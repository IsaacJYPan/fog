"""Generation loop driver: solver self-play -> distill -> belief refresh -> value retrain.

One generation:
  1. **generate** — solver-vs-solver self-play v2 logs, sharded over workers via
     `pb_datagen.run_sharded_datagen` (the L2 machinery; single-process generation was ~4x
     slower and broke the §6.2 day-per-generation arithmetic). The current agent is saved to
     `agent_in.pt` and workers load it posterior-aware; `policy_init` warm-starts the solves
     from the CURRENT policy (§10.2). Opening-solve caches are per-worker and per-generation —
     cached solves are stale the moment the weights move.
  2. **distill** — policy <- KL(solver executed distribution ‖ pi) at deployment inputs, value
     <- outcome (fog/training/distill, v2 reader).
  3. **refresh** — belief + self-belief retrained on FRESH policy self-play of the agent as it
     now plays (fog/training/belief_refresh — the §11 step-8 consistency step).
  4. **value retrain** (only when the loop runs with a trained PBValueNet, `--value-net`) —
     the net warm-starts from the incoming generation's weights and retrains on this
     generation's rows (fog/training/pb_value_train), because the belief/outcome distribution
     it conditions on moves with the weights; the next generation's solves load the retrained
     net. Without `--value-net` the loop runs on the §18 interim head (the pre-G5 mode).

Stopping (§15.4 gate 1, amended 2026-07-20 — A8): the WITHIN-generation gate is binding (held-out
KL must improve over that generation's own data; if distillation stops helping, iterating is
noise). The ACROSS-generation trend is reported two ways and is diagnostic, not a hard stop:
per-generation holdout KL (each on its own data — the §15.4 number, but its data distribution
moves with the loop), and KL on a FIXED reference set (generation 0's holdout), which is
apples-to-apples across generations but measures against a frozen solver whose play later
generations legitimately leave behind. Read the pair together; neither alone is a stop.

Honesty of the gate battery: this driver RUNS the KL gates and the refresh-CE and value gates,
and RECORDS the rest of §15.4 (fixed-gauntlet strength, solver-assisted >= policy-only, bounded
best-response) as not_run — they are separate eval CLIs (fog/eval/search_gauntlet, harness) whose
cadence and budget are an owner scheduling choice, not something to silently fake here. The
report names them so no generation claims a gate it did not run.

    python -m fog.scripts.pb_loop --ckpt runs/g0_v2_champion_resume/checkpoints/ppo_final.pt \
        --pool data/teams/pool_a --generations 3 --games 400 --out runs/pb_loop \
        --value-net runs/pb_value_a/pb_value.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def eval_reference_kl(agent: Any, data: dict, idx: np.ndarray, *, batch: int = 256) -> float:
    """Held-out KL(solver ‖ pi) of the CURRENT policy on a FIXED decision set (deployment
    inputs: posterior features zero). The across-generation comparison distill's own holdout
    cannot give — same rows every generation, only the policy moves."""
    import torch

    from fog.search.posterior_log import batch_from_rows
    from fog.training.distill import kl_search_policy

    device = next(agent.parameters()).device
    agent.eval()
    kls = []
    for lo in range(0, len(idx), batch):
        part = idx[lo: lo + batch]
        rows = batch_from_rows(data, part, device)
        with torch.no_grad():
            s, b = agent.head_inputs(rows)
            mask = torch.as_tensor(data["legality_mask"][part], dtype=torch.bool, device=device)
            log_probs, _ = agent.policy_value(s, mask, b, posterior_features=None)
            sd = torch.as_tensor(data["search_dist"][part], dtype=torch.float32, device=device)
            kls.append(kl_search_policy(sd, log_probs).cpu().numpy())
    return float(np.concatenate(kls).mean())


def run_generation(agent: Any, pool_path: str, gen_dir: Path, *,
                   games: int, workers: int, seed: int, flush_every: int,
                   search_kwargs: dict, value_net_path: str | None,
                   value_recompute_every: int = 8,
                   distill_kwargs: dict | None = None,
                   refresh_kwargs: dict | None = None,
                   value_kwargs: dict | None = None) -> tuple[dict, str | None]:
    """One full generation, mutating `agent` in place. Returns (report, path of the retrained
    value net for the NEXT generation — None when running on the interim head)."""
    import torch

    from fog.env.teams import load_pool
    from fog.scripts.pb_datagen import run_sharded_datagen
    from fog.training.belief_refresh import refresh_beliefs
    from fog.training.distill import distill, load_v2_decisions

    gen_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # ---- 1. generate: sharded solver-vs-solver self-play with the CURRENT weights ----------
    ckpt_in = gen_dir / "agent_in.pt"
    torch.save(agent.state_dict(), ckpt_in)
    shard_dir = gen_dir / "selfplay"
    gen_summary = run_sharded_datagen(
        pool=pool_path, games=games, workers=workers, out_dir=shard_dir,
        search_kwargs=search_kwargs, seed=rng.randrange(2**31),
        checkpoint=str(ckpt_in), selfplay=True, posterior_aware=True,
        value_net=value_net_path, value_recompute_every=value_recompute_every,
        flush_every=flush_every,
    )

    # ---- 2. distill + 3. refresh ------------------------------------------------------------
    data = load_v2_decisions(shard_dir)
    d_report = distill(agent, data, **(distill_kwargs or {}))
    r_report = refresh_beliefs(agent, load_pool(pool_path), seed=rng.randrange(2**31),
                               **(refresh_kwargs or {}))

    # ---- 4. value retrain (only in value-net mode) -------------------------------------------
    v_report = None
    next_net = None
    if value_net_path is not None:
        from fog.models.pb_value import PBValueNet
        from fog.training.pb_value_train import build_pb_value_rows, train_pb_value

        net = PBValueNet.from_tables(state_dim=int(agent.backbone.state_dim),
                                     dyn_dim=int(agent.backbone.dyn_dim))
        net.load_state_dict(torch.load(value_net_path, map_location="cpu"))
        v_report = train_pb_value(agent, net, build_pb_value_rows(shard_dir),
                                  **(value_kwargs or {}))
        next_net = str(gen_dir / "pb_value.pt")
        torch.save(net.state_dict(), next_net)

    report = {
        "games": int(games), "rows": int(data["outcome"].shape[0]),
        "generate": {k: v for k, v in gen_summary.items() if k != "per_shard"},
        "distill": d_report, "refresh": r_report, "value": v_report,
        "gates": {
            "kl_improved_within_generation": bool(d_report["improved"]),   # the binding stop
            "holdout_kl_after": d_report["holdout_after"]["kl"],
            "refresh_belief_ce_not_worse": bool(r_report["improved"]),
            "value_beats_predict_mean": (None if v_report is None
                                         else bool(v_report["improved_over_baseline"])),
            "strength_vs_fixed_gauntlet": "not_run (fog/eval/search_gauntlet — schedule "
                                          "separately at a declared budget)",
            "solver_assisted_vs_policy_only": "not_run (same harness)",
            "bounded_best_response_probe": "not_run (M3 machinery)",
        },
    }
    torch.save(agent.state_dict(), gen_dir / "agent.pt")
    (gen_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report, next_net


def main() -> None:
    import torch

    from fog.config import load_config_with_args
    from fog.training.distill import load_v2_decisions
    from fog.training.value_finetune import _split_by_game, load_posterior_agent

    ap = argparse.ArgumentParser(description="Solver->distill->refresh generation loop (I.5).")
    ap.add_argument("--ckpt", required=True, help="starting champion checkpoint")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pool", default="data/teams/pool_a")
    ap.add_argument("--out", type=Path, default=Path("runs/pb_loop"))
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="explicit chance-pruning threshold (default 0: exact enumeration)")
    ap.add_argument("--max-branches", type=int, default=None,
                    help="explicit per-cell branch cap (default none: exact enumeration)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="reserved; live own-side truncation is refused because it can leak truth")
    ap.add_argument("--top-j", type=int, default=8)
    ap.add_argument("--policy-init", action=argparse.BooleanOptionalAction, default=True,
                    help="warm-start solves from the current policy (--no-policy-init to "
                         "disable, e.g. for an iterations-to-stability A/B)")
    ap.add_argument("--value-net", default=None,
                    help="trained PBValueNet state dict (G5): solves use it and each "
                         "generation retrains it on its own rows; default = interim head")
    ap.add_argument("--value-recompute-every", type=int, default=8)
    ap.add_argument("--value-epochs", type=int, default=10)
    ap.add_argument("--flush-every", type=int, default=25)
    ap.add_argument("--distill-epochs", type=int, default=20)
    ap.add_argument("--refresh-games", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = load_posterior_agent(cfg, args.ckpt, device)
    search_kwargs = {"iters": args.iters, "min_prob": args.min_prob,
                     "max_branches": args.max_branches, "top_k": args.top_k,
                     "top_j": args.top_j, "policy_init": bool(args.policy_init)}

    value_net = args.value_net
    kl_track: list[float] = []
    ref_kl_track: list[float] = []
    ref_data = ref_hold = None
    for g in range(args.generations):
        report, next_net = run_generation(
            agent, args.pool, args.out / f"gen_{g:02d}",
            games=args.games, workers=args.workers, seed=args.seed * 1000 + g,
            flush_every=args.flush_every, search_kwargs=search_kwargs,
            value_net_path=value_net, value_recompute_every=args.value_recompute_every,
            distill_kwargs={"epochs": args.distill_epochs},
            refresh_kwargs={"games": args.refresh_games},
            value_kwargs={"epochs": args.value_epochs})
        value_net = next_net or value_net
        kl_track.append(float(report["gates"]["holdout_kl_after"]))
        if ref_data is None:   # generation 0's holdout = the fixed reference set (A8)
            ref_data = load_v2_decisions(args.out / "gen_00" / "selfplay")
            _, ref_hold = _split_by_game(ref_data["game"], 0.2, args.seed)
        ref_kl_track.append(eval_reference_kl(agent, ref_data, ref_hold))
        print(f"[pb_loop] generation {g}: rows={report['rows']} "
              f"KL={kl_track[-1]:.4f} refKL={ref_kl_track[-1]:.4f} "
              f"gates={report['gates']}", flush=True)
        if not report["gates"]["kl_improved_within_generation"]:
            print("[pb_loop] within-generation distill gate FAILED (held-out KL did not "
                  "improve on this generation's data) — stopping; iterating past a "
                  "non-improving distill is noise. Inspect the generation report.", flush=True)
            break
    (args.out / "loop_summary.json").write_text(json.dumps(
        {"kl_by_generation": kl_track,
         "reference_kl_by_generation": ref_kl_track,
         "generations_run": len(kl_track),
         "final_value_net": value_net}, indent=2))


if __name__ == "__main__":
    main()
