"""Multiprocess two-sided-search data generation on a team pool (PROPOSAL_IMPL_PLAN Phase G2,
cost lever L2; §6.2 acceptance bar: sustained >= 1,000 POOL-A games/day on this box).

Games are independent, the blind/interim value path needs no GPU, and the box has 20 cores; RAM
is the binding constraint (16 GB), so the worker count is a measured, pinned choice — run with
`--workers` and report, never assume. Each worker plays its shard of games through
`play_and_log_pb_games` (persistent mode over the full pool, both sides' teams drawn round-robin
over the K x K matchup grid so the L1 opening-solve cache and data coverage are both served) and
writes an independent v2 log shard; the parent aggregates throughput.

The learner is `p1` with a `StablePoolOwnStates` source (battle-lifetime-stable table axes); the
opponent defaults to the scripted max-damage bot (the data-plumbing default — solver-vs-solver
self-play is Phase I.2). Budget knobs (`--iters`, `--min-prob`, `--max-branches`, `--top-k`,
`--top-j`) are logged into each shard's meta so every row's generation profile is auditable
(lever L4 is "run this with a reduced profile", not separate code).

Usage (POOL-A smoke, 4 workers):
    python -m fog.scripts.pb_datagen --pool data/teams/pool_a --games 48 --workers 4 \
        --out runs/pb_datagen_a
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _run_shard(cfg: dict) -> dict:
    """One worker's shard: play `cfg['games']` games, save a v2 log shard, return stats.
    Top-level function (Windows spawn requires picklable workers); heavy imports stay inside."""
    import numpy as np
    import torch

    torch.set_num_threads(max(1, int(cfg["torch_threads"])))

    from fog.env.engine_battle import EngineBattle
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent
    from fog.search.own_states import StablePoolOwnStates
    from fog.search.pb_log import (
        OpeningSolveCache,
        play_and_log_pb_games,
        play_and_log_pb_selfplay_games,
    )

    pastes = load_pool(cfg["pool"])
    n = len(pastes)
    own_source = StablePoolOwnStates(pastes)
    opp_pool_tokens = own_source.tokens          # same pool both sides (POOL-A convention)

    shard, games, offset = cfg["shard"], cfg["games"], cfg["game_offset"]

    def factory(game_id: int, seed: int) -> EngineBattle:
        m = (offset + game_id) % (n * n)          # round-robin over the K x K matchup grid
        return EngineBattle(pastes[m // n], pastes[m % n], seed=seed,
                            battle_tag=f"pbgen-{shard}-{game_id}")

    agent = FogAgent.from_tables({})
    if cfg["checkpoint"] is not None:
        agent.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    cache = OpeningSolveCache()
    t0 = time.perf_counter()
    if cfg["selfplay"]:
        # Phase I.2: both sides solve (2x rows/game, 19.6 exact for both sides) — the
        # generation-loop distribution the PBValueNet actually serves.
        rows = play_and_log_pb_selfplay_games(
            agent, factory, own_source, opp_pool_tokens,
            n_games=games, seed=cfg["seed"], max_decisions=cfg["max_decisions"],
            search_kwargs=cfg["search_kwargs"], solve_cache=cache,
            out_path=cfg["out_path"],
        )
    else:
        rows = play_and_log_pb_games(
            agent, factory, own_source, opp_pool_tokens,
            n_games=games, seed=cfg["seed"], max_decisions=cfg["max_decisions"],
            search_kwargs=cfg["search_kwargs"], solve_cache=cache,
            out_path=cfg["out_path"],
        )
    wall = time.perf_counter() - t0
    n_truncated_games = len({r["game_id"] for r in rows if r["truncated"]})
    return {
        "shard": shard, "games": games, "rows": len(rows), "wall_s": wall,
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "truncated_games": n_truncated_games,
        "solver_decisions_per_game": len(rows) / max(1, games),
        "mean_dropped_opp_mass": float(np.mean([r["dropped_opp_mass"] for r in rows]) if rows
                                       else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, help="team pool directory (data/teams/pool_a)")
    ap.add_argument("--games", type=int, default=48, help="total games across all workers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, help="output directory for shard .pt files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-decisions", type=int, default=200)
    ap.add_argument("--iters", type=int, default=256, help="solver iterations per decision")
    ap.add_argument("--min-prob", type=float, default=0.005)
    ap.add_argument("--max-branches", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=None, help="G2 own-side truncation cap")
    ap.add_argument("--top-j", type=int, default=None, help="G2 opponent-side truncation cap")
    ap.add_argument("--n-queried", type=int, default=8)
    ap.add_argument("--checkpoint", default=None,
                    help="agent weights (default: random weights — the blind G0-recipe path)")
    ap.add_argument("--selfplay", action="store_true",
                    help="solver-vs-solver self-play (Phase I.2): both sides solve, 2x rows")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="torch threads per worker (default: 20 // workers)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_kwargs = {"iters": args.iters, "min_prob": args.min_prob,
                     "max_branches": args.max_branches, "n_queried": args.n_queried}
    if args.top_k is not None:
        search_kwargs["top_k"] = args.top_k
    if args.top_j is not None:
        search_kwargs["top_j"] = args.top_j

    per = [args.games // args.workers + (1 if w < args.games % args.workers else 0)
           for w in range(args.workers)]
    offsets = [sum(per[:w]) for w in range(args.workers)]
    shard_cfgs = [{
        "shard": w, "games": per[w], "game_offset": offsets[w],
        "pool": args.pool, "seed": args.seed * 100_003 + w,
        "max_decisions": args.max_decisions, "search_kwargs": search_kwargs,
        "checkpoint": args.checkpoint, "selfplay": args.selfplay,
        "torch_threads": args.torch_threads or max(1, (os.cpu_count() or 4) // args.workers),
        "out_path": str(out_dir / f"shard_{w:02d}.pt"),
    } for w in range(args.workers) if per[w] > 0]

    t0 = time.perf_counter()
    if len(shard_cfgs) == 1:
        stats = [_run_shard(shard_cfgs[0])]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(len(shard_cfgs)) as pool:
            stats = pool.map(_run_shard, shard_cfgs)
    wall = time.perf_counter() - t0

    games = sum(s["games"] for s in stats)
    rows = sum(s["rows"] for s in stats)
    hits = sum(s["cache_hits"] for s in stats)
    misses = sum(s["cache_misses"] for s in stats)
    truncated = sum(s["truncated_games"] for s in stats)
    summary = {
        "games": games, "rows": rows, "wall_s": round(wall, 1),
        "games_per_hour": round(games / (wall / 3600.0), 1),
        "games_per_day_extrapolated": round(games / (wall / 86400.0)),
        "solver_decisions_per_game": round(rows / max(1, games), 2),
        "opening_cache_hits": hits, "opening_cache_misses": misses,
        "truncated_games": truncated,
        "workers": len(shard_cfgs), "search_kwargs": search_kwargs,
        "selfplay": args.selfplay, "checkpoint": args.checkpoint,
        "per_shard": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shard"}, indent=2))


if __name__ == "__main__":
    main()
