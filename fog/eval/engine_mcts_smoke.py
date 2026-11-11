"""Smoke evaluation of EngineMCTS against scripted engine policies on the engine backend.

This is a smoke test on the approximate poke-engine simulator, not a gate result (gates are
measured on pinned Showdown). Games come in matched pairs: the same two teams and battle seed,
with EngineMCTS on p1 with team A in the first game and on p2 with team B in the second, so each
side and each team is played by both controllers.

``--tera-ablation`` instead plays EngineMCTS with Tera moves allowed against an identical
EngineMCTS whose root ``-tera`` options are filtered out (same matched-pair schedule).

Example:
    python -m fog.eval.engine_mcts_smoke --games 40 --k 4 --iterations 2000
    python -m fog.eval.engine_mcts_smoke --tera-ablation --games 100 --k 4 --iterations 2000
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


def _no_tera_policy(**kwargs):
    """EngineMCTSPolicy that never plays a Tera action: root ``-tera`` entries are dropped (as
    after Tera has been used) and the Tera indices are masked out, fallback included."""
    from fog.search.engine_mcts import EngineMCTSPolicy, engine_choice_map
    from fog.search.usage_worlds import public_reveal_grid

    class NoTeraEngineMCTS(EngineMCTSPolicy):
        def choose(self, battle, role: str) -> int:
            mask = battle.action_mask(role).copy()
            mask[4:8] = False
            choice_map = engine_choice_map(battle, role, mask)
            revealed = public_reveal_grid(battle, role, self.tables)

            def build_state(world, seed: int):
                return battle.fork_for_search(role, world, seed=seed).state

            def fallback() -> int:
                idx = int(self.fallback_policy.choose(battle, role))
                return idx - 4 if 4 <= idx < 8 else idx

            return self._decide(mask, choice_map, True, revealed, build_state, role, fallback)

    return NoTeraEngineMCTS(**kwargs)


def run_tera_ablation(pool_path: str, games: int, *, k: int, budget_ms: int,
                      iterations: int | None, seed: int) -> dict:
    """EngineMCTS with Tera allowed ("tera") vs the same controller with Tera filtered out."""
    from fog.env.engine_battle import EngineBattle
    from fog.env.teams import load_pool
    from fog.search.engine_mcts import EngineMCTSPolicy

    pool = load_pool(pool_path)
    rng = random.Random(seed)
    cfg = dict(k=k, budget_ms=budget_ms, iterations=iterations, threads=1)
    tera = EngineMCTSPolicy(seed=seed, **cfg)
    no_tera = _no_tera_policy(seed=seed + 1, **cfg)
    wins = draws = losses = tera_games = 0
    per_side = {"p1": [0, 0], "p2": [0, 0]}
    start = time.perf_counter()
    for game in range(games):
        if game % 2 == 0:
            team_a, team_b = rng.sample(range(len(pool)), 2)
            battle_seed = rng.randrange(2**31)
        tera_role = "p1" if game % 2 == 0 else "p2"
        other = "p2" if tera_role == "p1" else "p1"
        battle = EngineBattle(pool[team_a], pool[team_b], seed=battle_seed,
                              battle_tag=f"tera-ablation-{game}")
        winner = play_game(battle, {tera_role: tera, other: no_tera})
        if battle.tera_used.get(other):
            raise RuntimeError("the Tera-filtered controller terastallized")
        tera_games += bool(battle.tera_used.get(tera_role))
        per_side[tera_role][1] += 1
        if winner == tera_role:
            wins += 1
            per_side[tera_role][0] += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        print(f"tera-ablation game {game + 1}/{games}: tera={tera_role} winner={winner} "
              f"turns={battle.turn} tera_used={battle.tera_used[tera_role]}", flush=True)
    return {
        "match": "EngineMCTS tera-on vs tera-off", "games": games,
        "tera_on_wins": wins, "draws": draws, "tera_on_losses": losses,
        "tera_on_win_rate": wins / max(1, games),
        "tera_on_win_rate_by_side": {r: (w / n if n else None) for r, (w, n) in per_side.items()},
        "games_where_tera_on_terastallized": tera_games,
        "counters": {"tera_on": dict(tera.counters), "tera_off": dict(no_tera.counters)},
        "wall_seconds": time.perf_counter() - start,
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
    parser.add_argument("--tera-ablation", action="store_true",
                        help="play EngineMCTS with Tera allowed vs Tera filtered out instead")
    args = parser.parse_args()

    from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES

    opponents = args.opponents or [n for n in SCRIPTED_ENGINE_POLICIES if n != "Random"]
    if args.tera_ablation:
        reports = [run_tera_ablation(args.pool, args.games, k=args.k, budget_ms=args.budget_ms,
                                     iterations=args.iterations or None, seed=args.seed)]
    else:
        reports = [run(args.pool, name, args.games, k=args.k, budget_ms=args.budget_ms,
                       iterations=args.iterations or None, seed=args.seed)
                   for name in opponents]
    text = json.dumps(reports, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
