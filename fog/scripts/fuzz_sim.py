"""Cross-simulator fuzz harness (SIMULATION_PLAN §5): poke-engine vs local Showdown.

PPO will farm any mechanic the fast simulator gets wrong, so before poke-engine enters the training
loop, run the same seeded games through BOTH simulators and diff the per-decision trajectories.

The trick that makes cross-sim replay well-defined: the policy is a DETERMINISTIC FUNCTION OF THE
OBSERVABLE DECISION (seed, side, turn, 14-way legality mask) — while the two sims agree on the
state, they compute identical masks and therefore pick identical actions; the first fingerprint
divergence localizes the mechanic that differs. Stochastic outcomes (damage rolls, crits,
multi-hits) CANNOT be synchronized across sims, so hp is compared with a loose tolerance
(--hp-tol, default 0.15 ≈ beyond any single roll spread) and the sharp signals are the exact
fields: the legality MASK (choice lock / disable / trapping / forced-switch modeling), species
(switch/faint timing), and status.

    # engine-only sanity (no server): plays seeded games, prints turn/faint stats
    python -m fog.scripts.fuzz_sim --games 50 --pool data/teams/pool_b

    # full cross-sim diff (local Showdown server must be running):
    python -m fog.scripts.fuzz_sim --games 200 --pool data/teams/pool_b --showdown

Divergences are EXPECTED (the engine approximates; known gaps in fog/env/engine_battle.py's
docstring and KNOWN_ISSUES); what matters is the rate and which mechanics. Per plan §5: patch
upstream, exclude from team pools, or accept + document. The online control during engine training
is the cross-sim gauntlet (fog/eval/cross_sim.py, halt at >3% win-rate gap).
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np

_NON_ID = re.compile(r"[^a-z0-9]")


def _to_id(x: object) -> str:
    return _NON_ID.sub("", str(x).lower())


# ------------------------------------------------------------------------- the shared policy
def scripted_action(seed: int, role: str, turn: int, mask: np.ndarray) -> int:
    """Deterministic uniform-over-legal choice, identical across sims while masks agree.

    Seeded via sha256 of the full decision key — stable across processes and Python versions
    (random.Random stopped accepting tuple seeds in 3.11; str seeding depends on hash salting
    only via hash(), which sha256 sidesteps)."""
    import hashlib

    legal = tuple(int(i) for i in np.flatnonzero(mask))
    if not legal:
        return 13
    key = f"{seed}:{role}:{turn}:{legal}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    return legal[rng.randrange(len(legal))]


# -------------------------------------------------------------------------- engine trajectory
def engine_trajectory(paste_a: str, paste_b: str, seed: int, max_turns: int = 100) -> list[dict]:
    """Play one seeded game on poke-engine; one fingerprint per decision point."""
    from fog.env.engine_battle import EngineBattle

    battle = EngineBattle(paste_a, paste_b, seed=seed, battle_tag=f"fuzz-{seed}")
    traj: list[dict] = []
    while not battle.finished and battle.turn <= max_turns:
        moves = {}
        for role in battle.pending_sides():
            mask = battle.action_mask(role)
            traj.append(_engine_fingerprint(battle, role, mask))
            moves[role] = battle.engine_action(role, scripted_action(seed, role, battle.turn, mask))
        battle.step(moves)
    return _sorted_traj(traj)


def _engine_fingerprint(battle, role: str, mask: np.ndarray) -> dict:
    view = battle.view(role)
    side = [
        (m.species, round(float(m.current_hp_fraction), 3), str(m.status or "none"))
        for m in view.team.values()
    ]
    return {"turn": int(battle.turn), "role": role,
            "mask": tuple(int(i) for i in np.flatnonzero(mask)), "side": side}


# ------------------------------------------------------------------------ showdown trajectory
def showdown_trajectory(paste_a: str, paste_b: str, seed: int, fmt: str = "gen9ou",
                        max_turns: int = 100) -> list[dict]:
    """Replay the same seeded policy on the local Showdown server (one game, two scripted
    players). Requires poke-env + a running server."""
    import asyncio

    async def _run() -> list[dict]:
        traj: list[dict] = []
        p1 = _scripted_player(paste_a, seed, fmt, traj, max_turns)
        p2 = _scripted_player(paste_b, seed, fmt, traj, max_turns)
        await p1.battle_against(p2, n_battles=1)
        return _sorted_traj(traj)

    return asyncio.run(_run())


def _scripted_player(paste: str, seed: int, fmt: str, traj: list[dict], max_turns: int):
    from poke_env.player import Player
    from poke_env.teambuilder import ConstantTeambuilder

    from fog.env.actions import action_legality_mask, map_action_to_order
    from fog.env.player import unique_account

    class _Scripted(Player):  # type: ignore[misc]
        def choose_move(self, battle):
            role = getattr(battle, "player_role", None) or "p1"
            turn = int(getattr(battle, "turn", 0) or 0)
            if turn > max_turns:
                return self.choose_default_move()
            mask = np.asarray(action_legality_mask(battle), dtype=bool)
            traj.append(_showdown_fingerprint(battle, role, mask))
            idx = scripted_action(seed, role, turn, mask)
            try:
                return map_action_to_order(battle, idx, self)
            except AssertionError:
                # mask/request disagreement is itself a finding — surface it in the trajectory
                traj.append({"turn": turn, "role": role, "mask": ("MASK_REQUEST_MISMATCH",),
                             "side": []})
                return self.choose_default_move()

        def teampreview(self, battle):
            return "/team 123456"   # matches EngineBattle's default leads (0, 0)

    return _Scripted(
        battle_format=fmt,
        team=ConstantTeambuilder(paste),
        account_configuration=unique_account("fuzz"),
    )


def _showdown_fingerprint(battle, role: str, mask: np.ndarray) -> dict:
    side = []
    for mon in (getattr(battle, "team", None) or {}).values():
        status = getattr(mon, "status", None)
        side.append((
            _to_id(getattr(mon, "species", "")),
            round(float(getattr(mon, "current_hp_fraction", 1.0) or 0.0), 3),
            _to_id(getattr(status, "name", status)) if status else "none",
        ))
    return {"turn": int(getattr(battle, "turn", 0) or 0), "role": role,
            "mask": tuple(int(i) for i in np.flatnonzero(mask)), "side": side}


def _sorted_traj(traj: list[dict]) -> list[dict]:
    """Stable (turn, role) order — the two sims emit decisions in different within-turn order."""
    return sorted(traj, key=lambda f: (f["turn"], f["role"]))


# ----------------------------------------------------------------------------------- diffing
def diff_trajectories(a: list[dict], b: list[dict], hp_tol: float = 0.15) -> dict:
    """First divergence between two decision-fingerprint trajectories, classified by cause.

    Returns {} when they agree over the shared prefix AND have equal length. Causes:
    "mask" (legality modeling), "species" (switch/faint timing), "status", "hp" (beyond
    tolerance — type chart / ability / item effects), "length" (one game ended early).
    """
    for fa, fb in zip(a, b):
        key = {"turn": fa["turn"], "role": fa["role"]}
        if (fa["turn"], fa["role"]) != (fb["turn"], fb["role"]):
            return {"cause": "schedule", **key, "a": (fa["turn"], fa["role"]),
                    "b": (fb["turn"], fb["role"])}
        if fa["mask"] != fb["mask"]:
            return {"cause": "mask", **key, "a": fa["mask"], "b": fb["mask"]}
        for (ida, hpa, sta), (idb, hpb, stb) in zip(fa["side"], fb["side"]):
            if ida != idb:
                return {"cause": "species", **key, "a": ida, "b": idb}
            if sta != stb:
                return {"cause": "status", **key, "mon": ida, "a": sta, "b": stb}
            if abs(hpa - hpb) > hp_tol:
                return {"cause": "hp", **key, "mon": ida, "a": hpa, "b": hpb}
    if len(a) != len(b):
        return {"cause": "length", "a": len(a), "b": len(b),
                "turn": min(a[-1]["turn"] if a else 0, b[-1]["turn"] if b else 0), "role": "-"}
    return {}


# -------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-sim fuzz (SIMULATION_PLAN §5).")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp-tol", type=float, default=0.15)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--showdown", action="store_true",
                    help="replay each game on the local Showdown server and diff (server must run)")
    args = ap.parse_args()

    from collections import Counter

    from fog.env.teams import load_pool

    pool = load_pool(args.pool)
    rng = random.Random(args.seed)
    causes: Counter[str] = Counter()
    diverge_turns: list[int] = []
    diverged = 0
    total_decisions = 0
    finished = 0
    for g in range(args.games):
        a, b = rng.choice(pool), rng.choice(pool)
        game_seed = args.seed * 100_000 + g
        traj_e = engine_trajectory(a, b, seed=game_seed, max_turns=args.max_turns)
        total_decisions += len(traj_e)
        finished += 1
        if not args.showdown:
            continue
        traj_s = showdown_trajectory(a, b, seed=game_seed, max_turns=args.max_turns)
        d = diff_trajectories(traj_e, traj_s, hp_tol=args.hp_tol)
        if d:
            diverged += 1
            causes[d["cause"]] += 1
            diverge_turns.append(int(d.get("turn", 0)))
            print(f"game {g}: DIVERGED {d}")

    if args.showdown:
        turn_note = ""
        if diverge_turns:
            turn_note = (
                f" | divergence turn: min {min(diverge_turns)}, "
                f"mean {sum(diverge_turns) / len(diverge_turns):.1f}"
            )
        print(
            f"\ncross-sim fuzz: {finished} games, {diverged} diverged "
            f"({diverged / max(1, finished):.1%}) | causes: {dict(causes)}{turn_note}\n"
            "READING THIS: near-100% eventual divergence is EXPECTED — damage rolls/crits are "
            "sampled independently per sim and compound, surfacing as late 'hp'/'status'(fnt)/"
            "'schedule' causes. The mechanic signals are 'mask' at ANY turn (legality modeling: "
            "choice lock/trapping/disable) and hp/species/status divergence at turn 1-2 "
            "(type chart, abilities, items). Per plan §5: patch upstream, exclude from pools, or "
            "document in KNOWN_ISSUES — and keep the cross-sim gauntlet halt rule on for training."
        )
    else:
        print(
            f"engine fuzz: {finished} games, avg {total_decisions / max(1, finished):.1f} "
            f"decisions/game (engine-only sanity — pair with --showdown or the cross-sim "
            f"gauntlet before training on the engine)"
        )


if __name__ == "__main__":
    main()
