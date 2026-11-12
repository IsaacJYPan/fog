"""Check reconstruction fidelity against the live battle state.

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


def _extended_diverges(battle, role: str, tables) -> list[str]:
    """Fields beyond the core set: timers, boosts, volatiles, last move, Tera, sleep counters and
    the searcher's own PP/items. Opponent-side hidden fields (items, PP, exact stats) are not
    compared: with truth tokens as the world they only differ through documented approximations."""
    from fog.env.state_recon import reconstruct_battle

    opp = "p2" if role == "p1" else "p1"
    recon = reconstruct_battle(battle.view(role), battle.truth_tokens(opp), tables,
                               battle.trackers[role], role=role)
    o, n = battle.state, recon.state
    out: list[str] = []
    for name in ("weather_turns_remaining", "terrain_turns_remaining",
                 "trick_room_turns_remaining"):
        active = {"weather_turns_remaining": _id(o.weather) not in ("", "none"),
                  "terrain_turns_remaining": _id(o.terrain) not in ("", "none"),
                  "trick_room_turns_remaining": bool(o.trick_room)}[name]
        if active and int(getattr(o, name)) != int(getattr(n, name)):
            out.append(name)
    for lbl, so, sn in (("me" if role == "p1" else "opp", o.side_one, n.side_one),
                        ("opp" if role == "p1" else "me", o.side_two, n.side_two)):
        for boost in ("attack", "defense", "special_attack", "special_defense", "speed",
                      "accuracy", "evasion"):
            if int(getattr(so, f"{boost}_boost")) != int(getattr(sn, f"{boost}_boost")):
                out.append(f"{lbl}.boosts")
                break
        vo = {_id(v) for v in (so.volatile_statuses or ())}
        vn = {_id(v) for v in (sn.volatile_statuses or ())}
        for missing in sorted(vo - vn):
            out.append(f"{lbl}.volatile_missing.{missing}")
        for extra in sorted(vn - vo):
            out.append(f"{lbl}.volatile_extra.{extra}")
        if (int(so.substitute_health or 0) > 0) != (int(sn.substitute_health or 0) > 0):
            out.append(f"{lbl}.substitute")
        if _id(so.last_used_move) != _id(sn.last_used_move):
            out.append(f"{lbl}.last_used_move")
        for cond in ("reflect", "light_screen", "aurora_veil", "tailwind", "toxic_count",
                     "protect", "safeguard", "mist"):
            if int(getattr(so.side_conditions, cond)) != int(getattr(sn.side_conditions, cond)):
                out.append(f"{lbl}.{cond}")
        if tuple(so.wish) != tuple(sn.wish):
            out.append(f"{lbl}.wish")
        if tuple(str(x) for x in so.future_sight) != tuple(str(x) for x in sn.future_sight):
            out.append(f"{lbl}.future_sight")
        for dur in ("confusion", "encore", "lockedmove", "slowstart", "taunt", "yawn"):
            if (int(getattr(so.volatile_status_durations, dur))
                    != int(getattr(sn.volatile_status_durations, dur))):
                out.append(f"{lbl}.duration.{dur}")
        n_by = {_id(p.id): p for p in sn.pokemon if _id(p.id) not in ("", "none")}
        for po in so.pokemon:
            sid = _id(po.id)
            pn = n_by.get(sid)
            if sid in ("", "none") or pn is None:
                continue
            if bool(po.terastallized) != bool(pn.terastallized):
                out.append(f"{lbl}.terastallized")
            elif bool(po.terastallized) and _id(po.tera_type) != _id(pn.tera_type):
                out.append(f"{lbl}.tera_type")
            if tuple(_id(t) for t in po.types) != tuple(_id(t) for t in pn.types):
                out.append(f"{lbl}.types")
            if _id(po.status) == "sleep" and (
                    int(po.sleep_turns), int(po.rest_turns)) != (
                    int(pn.sleep_turns), int(pn.rest_turns)):
                out.append(f"{lbl}.sleep_counters")
            if lbl != "me":
                continue
            # the engine writes "unknownitem" for an empty slot in some paths; both mean no item
            if ({_id(po.item), _id(pn.item)} - {"", "none", "unknownitem"}
                    and _id(po.item) != _id(pn.item)):
                out.append("me.item")
            if _id(po.ability) != _id(pn.ability):
                out.append("me.ability")
            pp_o = {_id(m.id): (int(m.pp), bool(m.disabled)) for m in po.moves}
            pp_n = {_id(m.id): (int(m.pp), bool(m.disabled)) for m in pn.moves}
            if set(pp_o) != set(pp_n):
                out.append("me.moves")
            elif any(pp_o[k][0] != pp_n[k][0] for k in pp_o):
                out.append("me.pp")
            elif any(pp_o[k][1] != pp_n[k][1] for k in pp_o):
                out.append("me.move_disabled")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Check reconstruction fidelity.")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extended", action="store_true",
                    help="also compare timers, boosts, volatiles, last move, Tera, sleep counters "
                         "and own PP/items (reported separately from the core fields)")
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
    extended_bad = 0
    extended_tally: Counter = Counter()

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
            if args.extended:
                more = _extended_diverges(battle, role, TABLES)
                if more:
                    extended_bad += 1
                    extended_tally.update(set(more))

    print(f"positions checked : {positions}")
    print(f"with any mismatch : {bad_positions} "
          f"({100 * bad_positions / max(1, positions):.1f}%)")
    if tally:
        print("mismatch fields (positions affected):")
        for field, count in tally.most_common():
            print(f"  {field:<20} {count}")
    else:
        print("no mismatches on any field.")
    if args.extended:
        print(f"extended mismatch : {extended_bad} "
              f"({100 * extended_bad / max(1, positions):.1f}%)")
        for field, count in extended_tally.most_common():
            print(f"  {field:<34} {count}")


if __name__ == "__main__":
    main()
