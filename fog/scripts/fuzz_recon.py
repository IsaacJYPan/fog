"""Reconstruction-fidelity fuzz (search §8 / SIMULATION_PLAN).

Round-trips many random mid-battle engine positions through the Showdown-shaped view and back
(engine state -> EngineView -> state_recon.reconstruct_battle -> engine state) and reports how
often, and on which fields, the rebuilt position diverges from ground truth. NO ply is simulated,
so an exact match is expected; any divergence is reconstruction loss, not damage-roll stochasticity.

Run this ONCE per pool (and after any poke-engine upgrade) before trusting Showdown-search gauntlet
numbers — it is a developer gate, never part of training or eval. Divergence should be ~0 on the
core fields (stats/HP/status/hazards/active/weather); a nonzero screen/volatile tail is the
documented approximation set (state_recon docstring).

    python -m fog.scripts.fuzz_recon --pool data/teams/pool_b --games 200
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def _id(x) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _diverges(battle, role: str, tables) -> list[str]:
    from fog.env.state_recon import reconstruct_battle

    opp = "p2" if role == "p1" else "p1"
    view = battle.view(role)
    recon = reconstruct_battle(view, battle.truth_tokens(opp), tables, battle.trackers[role],
                               role=role)
    o, n = battle.state, recon.state
    out: list[str] = []
    if _id(o.weather) != _id(n.weather):
        out.append("weather")
    if _id(o.terrain) != _id(n.terrain):
        out.append("terrain")
    if bool(o.trick_room) != bool(n.trick_room):
        out.append("trickroom")
    for lbl, so, sn in (("me" if role == "p1" else "opp", o.side_one, n.side_one),
                        ("opp" if role == "p1" else "me", o.side_two, n.side_two)):
        if str(so.active_index) != str(sn.active_index):
            out.append(f"{lbl}.active")
        for hz in ("stealth_rock", "spikes", "toxic_spikes", "sticky_web"):
            if int(getattr(so.side_conditions, hz)) != int(getattr(sn.side_conditions, hz)):
                out.append(f"{lbl}.{hz}")
        o_by = {_id(p.id): p for p in so.pokemon if _id(p.id) not in ("", "none")}
        n_by = {_id(p.id): p for p in sn.pokemon if _id(p.id) not in ("", "none")}
        for sid, po in o_by.items():
            pn = n_by.get(sid)
            if pn is None:
                out.append(f"{lbl}.missing_mon")
                continue
            fo = po.hp / po.maxhp if po.maxhp else 0
            fn = pn.hp / pn.maxhp if pn.maxhp else 0
            if abs(fo - fn) > 0.02:
                out.append(f"{lbl}.hp")
            if _id(po.status) != _id(pn.status):
                out.append(f"{lbl}.status")
            if lbl == "me" and (po.attack, po.defense, po.special_attack, po.special_defense,
                                po.speed, po.maxhp) != (pn.attack, pn.defense, pn.special_attack,
                                                        pn.special_defense, pn.speed, pn.maxhp):
                out.append("me.stats")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruction-fidelity fuzz (search §8).")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from fog.env.engine_battle import HAVE_ENGINE

    if not HAVE_ENGINE:
        raise SystemExit("poke-engine not installed: pip install poke-engine")

    import numpy as np

    from fog.constants.tables import TABLES
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy
    from fog.env.teams import load_pool

    pool = load_pool(args.pool)
    rng = np.random.default_rng(args.seed)
    positions = 0
    bad_positions = 0
    tally: Counter = Counter()

    for g in range(args.games):
        a, b = (pool[i] for i in rng.choice(len(pool), 2, replace=len(pool) < 2))
        battle = EngineBattle(a, b, seed=int(rng.integers(2**31)), battle_tag=f"fuzz-{g}")
        rand = EngineRandomPolicy(seed=int(rng.integers(2**31)))
        maxd = EngineMaxDamagePolicy(seed=int(rng.integers(2**31)))
        for _ in range(int(rng.integers(1, args.max_turns))):
            if battle.finished:
                break
            moves = {}
            for role in battle.pending_sides():
                pol = rand if role == "p1" else maxd
                moves[role] = battle.engine_action(role, pol.choose(battle, role))
            battle.step(moves)
        if battle.finished:
            continue
        for role in ("p1", "p2"):
            issues = _diverges(battle, role, TABLES)
            positions += 1
            if issues:
                bad_positions += 1
                tally.update(set(issues))

    print(f"positions checked : {positions}")
    print(f"with any mismatch : {bad_positions} "
          f"({100 * bad_positions / max(1, positions):.1f}%)")
    if tally:
        print("mismatch fields (positions affected):")
        for field, count in tally.most_common():
            print(f"  {field:<20} {count}")
    else:
        print("no mismatches on any field.")


if __name__ == "__main__":
    main()
