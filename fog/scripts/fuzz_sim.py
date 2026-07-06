"""Cross-simulator fuzz harness (plan §7 divergence control): poke-engine vs local Showdown.

PPO will farm any mechanic the fast simulator gets wrong, so before poke-engine enters the training
loop, run the same teams + seeded action policies through BOTH simulators and diff the per-turn
HP/status/faint trajectories. Divergences are expected (the engine approximates some mechanics);
what matters is the RATE and WHICH mechanics — patch upstream, avoid in team pools, or accept and
list in KNOWN_ISSUES. Halt rule for training (separate): cross-sim gauntlet win-rate gap > 3%.

    # engine-only sanity (no server): plays seeded games on the engine, prints turn/faint stats
    python -m fog.scripts.fuzz_sim --games 50 --pool data/teams/pool_b

    # full cross-sim diff (local Showdown server must be running):
    python -m fog.scripts.fuzz_sim --games 200 --pool data/teams/pool_b --showdown

Both sides use the same seeded uniform-over-legal-actions policy; when the sims agree on state,
they draw identical action sequences, so the first trajectory divergence localizes the mechanic.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def engine_trajectory(paste_a: str, paste_b: str, seed: int, max_turns: int = 80) -> list[dict]:
    """Play one seeded game on poke-engine; return per-turn state fingerprints."""
    import poke_engine as pe

    from fog.env.rust_env import paste_to_engine_side

    rng = random.Random(seed)
    state = pe.State(side_one=paste_to_engine_side(paste_a), side_two=paste_to_engine_side(paste_b))
    traj: list[dict] = []

    def legal_moves(side) -> list[str]:
        active = side.pokemon[int(side.active_index)]
        moves = [m.id for m in active.moves if getattr(m, "disabled", False) is False] or ["splash"]
        bench = [p.id for i, p in enumerate(side.pokemon)
                 if i != int(side.active_index) and p.hp > 0]
        if active.hp <= 0:
            return bench or moves
        return moves + bench

    for turn in range(1, max_turns + 1):
        m1 = rng.choice(legal_moves(state.side_one))
        m2 = rng.choice(legal_moves(state.side_two))
        try:
            instructions = pe.generate_instructions(state, m1, m2)
        except ValueError:
            break  # move became illegal (choice lock etc.) — end of comparable segment
        if not instructions:
            break
        # sample an outcome branch by probability, same RNG stream
        r = rng.random() * 100.0
        acc = 0.0
        chosen = instructions[-1]
        for ins in instructions:
            acc += float(ins.percentage)
            if r <= acc:
                chosen = ins
                break
        state.apply_instructions(chosen)
        traj.append(_fingerprint(state, turn, (m1, m2)))
        if _side_lost(state.side_one) or _side_lost(state.side_two):
            break
    return traj


def _fingerprint(state, turn: int, moves) -> dict:
    def side_fp(side):
        return [(p.id, round(p.hp / p.maxhp, 3) if p.maxhp else 0.0, str(p.status))
                for p in side.pokemon]

    return {"turn": turn, "moves": moves, "s1": side_fp(state.side_one),
            "s2": side_fp(state.side_two)}


def _side_lost(side) -> bool:
    return all(p.hp <= 0 for p in side.pokemon)


def diff_trajectories(a: list[dict], b: list[dict], hp_tol: float = 0.02) -> dict:
    """First divergence between two trajectories (None if they agree over the shared prefix)."""
    for ta, tb in zip(a, b):
        for side in ("s1", "s2"):
            for (ida, hpa, sta), (idb, hpb, stb) in zip(ta[side], tb[side]):
                if ida != idb or abs(hpa - hpb) > hp_tol or sta != stb:
                    return {"turn": ta["turn"], "side": side,
                            "a": (ida, hpa, sta), "b": (idb, hpb, stb)}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-sim fuzz (plan §7).")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--showdown", action="store_true",
                    help="also replay each script on the local Showdown server and diff")
    args = ap.parse_args()

    from fog.env.teams import load_pool

    pool = load_pool(args.pool)
    rng = random.Random(args.seed)
    total_turns = 0
    finished = 0
    for g in range(args.games):
        a, b = rng.choice(pool), rng.choice(pool)
        traj = engine_trajectory(a, b, seed=args.seed * 100_000 + g)
        total_turns += len(traj)
        finished += 1
        if args.showdown:
            print("NOTE: Showdown-side scripted replay requires the queue-driven ScriptedPlayer; "
                  "run the cross-sim gauntlet (eval vs SimpleHeuristics on both sims) as the "
                  "operational divergence check until it lands.")
            break
    print(f"engine fuzz: {finished} games, avg {total_turns / max(1, finished):.1f} turns/game "
          f"(engine-only sanity — pair with the cross-sim gauntlet before training on the engine)")


if __name__ == "__main__":
    main()
