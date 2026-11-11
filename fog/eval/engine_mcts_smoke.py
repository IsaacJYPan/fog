"""Smoke evaluation of EngineMCTS against scripted engine policies on the engine backend.

This is a smoke test on the approximate poke-engine simulator, not a gate result (gates are
measured on pinned Showdown). Games come in matched pairs: the same two teams and battle seed,
with EngineMCTS on p1 with team A in the first game and on p2 with team B in the second, so each
side and each team is played by both controllers.

Example:
    python -m fog.eval.engine_mcts_smoke --games 40 --k 4 --iterations 2000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def play_game(battle, controllers: dict, *, max_steps: int = 2000) -> str | None:
    """Drive one EngineBattle to completion. Returns the winning role (or None for a draw)."""
    for _ in range(max_steps):
        if battle.finished:
            break
        moves = {}
        for role in battle.pending_sides():
            idx = int(controllers[role].choose(battle, role))
            if not battle.action_mask(role)[idx]:
                raise RuntimeError(f"{role} chose illegal action {idx}")
            moves[role] = battle.engine_action(role, idx)
        battle.step(moves)
    return battle.winner if battle.finished else None


def run(pool_path: str, opponent: str, games: int, *, k: int, budget_ms: int,
        iterations: int | None, seed: int) -> dict:
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES
    from fog.env.teams import load_pool
    from fog.search.engine_mcts import EngineMCTSPolicy

    pool = load_pool(pool_path)
    rng = random.Random(seed)
    mcts = EngineMCTSPolicy(k=k, budget_ms=budget_ms, iterations=iterations, threads=1, seed=seed)
    opp_cls = SCRIPTED_ENGINE_POLICIES[opponent]
    opp = opp_cls(seed=seed + 1)
    wins = draws = 0
    per_side = {"p1": [0, 0], "p2": [0, 0]}
    start = time.perf_counter()
    for game in range(games):
        if game % 2 == 0:
            team_a, team_b = rng.sample(range(len(pool)), 2)
            battle_seed = rng.randrange(2**31)
        mcts_role = "p1" if game % 2 == 0 else "p2"
        # p1 always holds team A and p2 team B; swapping controllers between the two games of a
        # pair gives MCTS team A on p1, then team B on p2 (the opponent gets the other team).
        battle = EngineBattle(pool[team_a], pool[team_b], seed=battle_seed,
                              battle_tag=f"smoke-{game}")
        controllers = {mcts_role: mcts, ("p2" if mcts_role == "p1" else "p1"): opp}
        winner = play_game(battle, controllers)
        per_side[mcts_role][1] += 1
        if winner == mcts_role:
            wins += 1
            per_side[mcts_role][0] += 1
        elif winner is None:
            draws += 1
        print(f"{opponent} game {game + 1}/{games}: mcts={mcts_role} winner={winner} "
              f"turns={battle.turn}", flush=True)
    elapsed = time.perf_counter() - start
    c = dict(mcts.counters)
    searched = max(1, c["searched"])
    return {
        "opponent": opponent, "games": games, "wins": wins, "draws": draws,
        "win_rate": wins / max(1, games),
        "win_rate_by_side": {r: (w / n if n else None) for r, (w, n) in per_side.items()},
        "counters": c,
        "fallback_rate": c["fallbacks"] / max(1, c["decisions"]),
        "seconds_per_searched_decision": c["search_seconds"] / searched,
        "wall_seconds": elapsed,
        "failure_reasons": mcts.failure_reasons,
        "config": {"k": k, "budget_ms": budget_ms, "iterations": iterations, "seed": seed,
                   "pool": pool_path},
        "scope": "smoke test on the approximate engine backend; not a gate result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="data/teams/pool_b")
    parser.add_argument("--opponents", nargs="+", default=None,
                        help="scripted engine policies (default: every non-Random one)")
    parser.add_argument("--games", type=int, default=40, help="games per opponent (even)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--budget-ms", type=int, default=1600)
    parser.add_argument("--iterations", type=int, default=2000,
                        help="fixed iterations per world; 0 uses --budget-ms instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES

    opponents = args.opponents or [n for n in SCRIPTED_ENGINE_POLICIES if n != "Random"]
    reports = [run(args.pool, name, args.games, k=args.k, budget_ms=args.budget_ms,
                   iterations=args.iterations or None, seed=args.seed) for name in opponents]
    text = json.dumps(reports, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
