"""Generation loop driver — solver self-play -> distill -> belief refresh (PROPOSAL_IMPL_PLAN
Phase I.5; proposal §11 steps 1-8, §15.4 gates).

One generation:
  1. **generate** — solver-vs-solver self-play v2 logs (`play_and_log_pb_selfplay_games`, both
     sides through the two-sided search; `policy_init` warm-starts the solves from the CURRENT
     policy, §10.2). The opening-solve cache is FRESH each generation — cached solves are stale
     the moment the weights move.
  2. **distill** — policy <- KL(solver executed distribution ‖ pi) at deployment inputs, value
     <- outcome (fog/training/distill, v2 reader). Gate recorded: held-out KL improved, and
     falling ACROSS generations (§15.4 gate 1).
  3. **refresh** — belief + self-belief retrained on FRESH policy self-play of the agent as it
     now plays (fog/training/belief_refresh — the §11 step-8 consistency step). Gate recorded:
     held-out belief/self-belief CE not worse.

Honesty of the gate battery: this driver RUNS gates 1 (KL) and the refresh CE gate, and RECORDS
the rest of §15.4 (fixed-gauntlet strength, solver-assisted >= policy-only, bounded
best-response) as not_run — they are separate eval CLIs (fog/eval/search_gauntlet, harness) whose
cadence and budget are an owner scheduling choice, not something to silently fake here. The
report names them so no generation claims a gate it did not run.

    python -m fog.scripts.pb_loop --ckpt runs/g0_v2_champion/checkpoints/ppo_final.pt \
        --pool data/teams/pool_a --generations 3 --games 400 --out runs/pb_loop
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any


def run_generation(agent: Any, pool_pastes: list[str], gen_dir: Path, *,
                   games: int, seed: int, search_kwargs: dict | None = None,
                   distill_kwargs: dict | None = None,
                   refresh_kwargs: dict | None = None) -> dict:
    """One full generation, mutating `agent` in place. Returns the generation report."""
    import torch

    from fog.env.engine_battle import EngineBattle
    from fog.search.own_states import StablePoolOwnStates
    from fog.search.pb_log import OpeningSolveCache, play_and_log_pb_selfplay_games
    from fog.training.belief_refresh import refresh_beliefs
    from fog.training.distill import distill, load_v2_decisions

    gen_dir.mkdir(parents=True, exist_ok=True)
    src = StablePoolOwnStates(pool_pastes)
    rng = random.Random(seed)
    n = len(pool_pastes)

    def factory(game_id: int, s: int) -> EngineBattle:
        m = game_id % (n * n)
        return EngineBattle(pool_pastes[m // n], pool_pastes[m % n], seed=s,
                            battle_tag=f"pbloop-{gen_dir.name}-{game_id}")

    t0 = time.perf_counter()
    shard = gen_dir / "shard_00.pt"
    play_and_log_pb_selfplay_games(
        agent, factory, src, src.tokens, n_games=games, out_path=str(shard),
        seed=rng.randrange(2**31), solve_cache=OpeningSolveCache(),
        search_kwargs=search_kwargs)
    gen_s = time.perf_counter() - t0

    data = load_v2_decisions(shard)
    d_report = distill(agent, data, **(distill_kwargs or {}))
    r_report = refresh_beliefs(agent, pool_pastes, seed=rng.randrange(2**31),
                               **(refresh_kwargs or {}))

    report = {
        "games": int(games), "rows": int(data["outcome"].shape[0]),
        "generate_wall_s": round(gen_s, 1),
        "distill": d_report, "refresh": r_report,
        "gates": {
            "kl_improved_within_generation": bool(d_report["improved"]),
            "holdout_kl_after": d_report["holdout_after"]["kl"],
            "refresh_belief_ce_not_worse": bool(r_report["improved"]),
            "strength_vs_fixed_gauntlet": "not_run (fog/eval/search_gauntlet — schedule "
                                          "separately at a declared budget)",
            "solver_assisted_vs_policy_only": "not_run (same harness)",
            "bounded_best_response_probe": "not_run (M3 machinery)",
        },
    }
    torch.save(agent.state_dict(), gen_dir / "agent.pt")
    (gen_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    import torch

    from fog.config import load_config_with_args
    from fog.env.teams import load_pool

    ap = argparse.ArgumentParser(description="Solver->distill->refresh generation loop (I.5).")
    ap.add_argument("--ckpt", required=True, help="starting champion checkpoint")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pool", default="data/teams/pool_a")
    ap.add_argument("--out", type=Path, default=Path("runs/pb_loop"))
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--max-branches", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--top-j", type=int, default=8)
    ap.add_argument("--policy-init", action="store_true", default=True)
    ap.add_argument("--distill-epochs", type=int, default=20)
    ap.add_argument("--refresh-games", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    cfg.setdefault("model", {})["use_posterior_features"] = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from fog.training.value_finetune import load_posterior_agent

    agent = load_posterior_agent(cfg, args.ckpt, device)
    pool = load_pool(args.pool)
    search_kwargs = {"iters": args.iters, "min_prob": args.min_prob,
                     "max_branches": args.max_branches, "top_k": args.top_k,
                     "top_j": args.top_j, "policy_init": bool(args.policy_init)}

    kl_track: list[float] = []
    for g in range(args.generations):
        report = run_generation(
            agent, pool, args.out / f"gen_{g:02d}",
            games=args.games, seed=args.seed * 1000 + g, search_kwargs=search_kwargs,
            distill_kwargs={"epochs": args.distill_epochs},
            refresh_kwargs={"games": args.refresh_games})
        kl = float(report["gates"]["holdout_kl_after"])
        kl_track.append(kl)
        print(f"[pb_loop] generation {g}: rows={report['rows']} "
              f"KL={kl:.4f} gates={report['gates']}", flush=True)
        if len(kl_track) >= 2 and kl_track[-1] > kl_track[-2]:
            print("[pb_loop] §15.4 gate 1 REGRESSED across generations (held-out KL rose) — "
                  "stopping; inspect the generation report before continuing.", flush=True)
            break
    (args.out / "loop_summary.json").write_text(json.dumps(
        {"kl_by_generation": kl_track, "generations_run": len(kl_track)}, indent=2))


if __name__ == "__main__":
    main()
