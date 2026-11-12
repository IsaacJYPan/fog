"""Check Showdown-to-engine reconstruction against Showdown's own battle state.

``fog.scripts.fuzz_recon`` round-trips engine positions through an engine-shaped view, so it cannot
see faults in how a real poke-env battle is read (a weather or effect that poke-env names
differently from the engine, PP read from the request, formes, ...). This tool plays scripted
games on the pinned in-memory Showdown, and at every decision rebuilds the engine state exactly as
``ShowdownEngineMCTS`` does (``state_recon.reconstruct_battle`` plus ``apply_request_overlay``),
giving it the opponent's TRUE team sheet as the sampled world. With the true world, every public
or own-side field should match Showdown's state; what differs is reconstruction loss.

It reports, per field, the number of decision positions where the rebuilt state disagrees with
Showdown. Nothing is searched and no game result is used. Developer diagnostic (ROADMAP P1.3).

    python -m fog.scripts.fuzz_recon_showdown --pool data/teams/pool_b --games 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

_STATUS = {"": "none", "slp": "sleep", "psn": "poison", "tox": "toxic", "brn": "burn",
           "par": "paralyze", "frz": "freeze"}
_WEATHER = {"": "none", "raindance": "rain", "primordialsea": "rain", "sunnyday": "sun",
            "desolateland": "sun", "sandstorm": "sand", "snowscape": "snow", "snow": "snow",
            "hail": "snow", "deltastream": "none"}
_HAZARDS = {"stealthrock": "stealth_rock", "spikes": "spikes", "toxicspikes": "toxic_spikes",
            "stickyweb": "sticky_web"}
_SCREENS = {"reflect": "reflect", "lightscreen": "light_screen", "auroraveil": "aurora_veil",
            "tailwind": "tailwind"}
_BOOSTS = {"atk": "attack", "def": "defense", "spa": "special_attack", "spd": "special_defense",
           "spe": "speed", "accuracy": "accuracy", "evasion": "evasion"}
_STATS = {"atk": "attack", "def": "defense", "spa": "special_attack", "spd": "special_defense",
          "spe": "speed"}
# Showdown volatiles that are bookkeeping for one move or one turn, not state a search needs.
_IGNORED_VOLATILES = {
    "stall", "choicelock", "twoturnmove", "mustrecharge", "flinch", "protect", "endure",
    "banefulbunker", "spikyshield", "kingsshield", "silktrap", "burningbulwark", "obstruct",
    "focuspunch", "beakblast", "shelltrap", "roost", "helpinghand", "magiccoat", "snatch",
    "destinybond", "grudge", "rage", "counter", "mirrorcoat", "metalburst", "bide", "followme",
    "ragepowder", "electrify", "powder", "instruct", "quash", "pursuit", "feint", "healblock",
    "dynamax", "commanding", "commanded", "sparklingaria", "glaiverush", "lockon",
}


def _id(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _species_ids(mon: dict[str, Any]) -> list[str]:
    """Candidate ids for a Showdown-state Pokemon: its team-sheet species, then current forme."""
    out = [_id(mon["set"].get("species") or mon["set"].get("name") or "")]
    for key in ("species", "baseSpecies"):
        match = re.search(r"\[Species:([a-z0-9]+)\]", str(mon.get(key, "")))
        if match:
            out.append(match.group(1))
    return [sid for sid in out if sid]


def _engine_mons(side: Any) -> dict[str, Any]:
    return {_id(p.id): p for p in side.pokemon if _id(p.id) not in ("", "none")}


def compare(truth: dict[str, Any], state: Any, role: str) -> list[str]:
    """Mismatch labels between Showdown's state and a rebuilt engine state for searcher `role`."""
    out: list[str] = []
    field = truth["field"]
    weather = _WEATHER.get(_id(field.get("weather")), "none")
    if weather != _id(state.weather):
        out.append(f"weather.{_id(field.get('weather')) or 'none'}_as_{_id(state.weather)}")
    elif weather != "none":
        left = int((field.get("weatherState") or {}).get("duration") or 0)
        if left and left != int(state.weather_turns_remaining):
            out.append("weather_turns_remaining")
    terrain = _id(field.get("terrain")) or "none"
    if terrain != _id(state.terrain):
        out.append(f"terrain.{terrain}_as_{_id(state.terrain)}")
    elif terrain != "none":
        left = int((field.get("terrainState") or {}).get("duration") or 0)
        if left and left != int(state.terrain_turns_remaining):
            out.append("terrain_turns_remaining")
    trick_room = (field.get("pseudoWeather") or {}).get("trickroom")
    if bool(trick_room) != bool(state.trick_room):
        out.append("trick_room")
    elif trick_room and int(trick_room.get("duration") or 0) != int(
            state.trick_room_turns_remaining):
        out.append("trick_room_turns_remaining")

    for index, side_role in enumerate(("p1", "p2")):
        label = "me" if side_role == role else "opp"
        ts = truth["sides"][index]
        es = state.side_one if side_role == "p1" else state.side_two
        conditions = ts.get("sideConditions") or {}
        for name, attr in _HAZARDS.items():
            layers = int((conditions.get(name) or {}).get("layers") or (1 if name in conditions
                                                                         else 0))
            if layers != int(getattr(es.side_conditions, attr)):
                out.append(f"{label}.{attr}")
        for name, attr in _SCREENS.items():
            left = int((conditions.get(name) or {}).get("duration") or 0)
            if left != int(getattr(es.side_conditions, attr)):
                out.append(f"{label}.{attr}" if bool(left) != bool(
                    getattr(es.side_conditions, attr)) else f"{label}.{attr}_turns")

        engine = _engine_mons(es)
        engine_active = _id(es.pokemon[int(es.active_index)].id)
        for mon in ts["pokemon"]:
            ids = _species_ids(mon)
            em = next((engine[sid] for sid in ids if sid in engine), None)
            if em is None:
                out.append(f"{label}.missing_mon")
                continue
            if mon.get("isActive") and _id(em.id) != engine_active:
                out.append(f"{label}.active")
            true_fraction = mon["hp"] / mon["maxhp"] if mon["maxhp"] else 0.0
            fraction = em.hp / em.maxhp if em.maxhp else 0.0
            if label == "me" and (int(mon["hp"]), int(mon["maxhp"])) != (int(em.hp),
                                                                          int(em.maxhp)):
                out.append("me.hp")
            elif label == "opp" and abs(true_fraction - fraction) > 0.02:
                # a benched Regenerator heals out of sight: the searcher cannot know its HP
                hidden = _id(mon.get("ability")) == "regenerator" and not mon.get("isActive")
                out.append("opp.hp_hidden_regenerator" if hidden else "opp.hp")
            if mon["hp"] > 0 and _STATUS.get(_id(mon.get("status")), "none") != _id(em.status):
                out.append(f"{label}.status")
            # Showdown clears `terastallized` when the Pokemon faints; the engine keeps the flag
            # because that is how it knows the side has used its Tera. Compare living ones only.
            if mon["hp"] > 0 and bool(mon.get("terastallized")) != bool(em.terastallized):
                out.append(f"{label}.terastallized")
            elif (mon["hp"] > 0 and mon.get("terastallized")
                  and _id(mon["terastallized"]) != _id(em.tera_type)):
                out.append(f"{label}.tera_type")
            if _id(mon.get("status")) == "slp" and mon["hp"] > 0:
                sleep = mon.get("statusState") or {}
                # Rest is the only sleep a Pokemon inflicts on itself
                from_rest = bool(sleep.get("source")) and sleep.get("source") == sleep.get(
                    "target")
                if from_rest != (int(em.rest_turns) > 0):
                    out.append(f"{label}.rest_sleep")
            if label == "me":
                stored = mon.get("storedStats") or {}
                if any(int(stored[k]) != int(getattr(em, attr)) for k, attr in _STATS.items()
                       if k in stored) and not mon.get("transformed"):
                    out.append("me.stats")
                if _id(mon.get("item")) != (_id(em.item) if _id(em.item) not in (
                        "none", "unknownitem") else ""):
                    out.append("me.item")
                true_pp = {_id(slot["id"]): int(slot["pp"]) for slot in mon["moveSlots"]}
                pp = {_id(m.id): int(m.pp) for m in em.moves if _id(m.id) not in ("", "none")}
                if set(true_pp) != set(pp):
                    out.append("me.moves")
                elif any((true_pp[k] == 0) != (pp[k] == 0) for k in true_pp):
                    out.append("me.pp_zero")
            if not mon.get("isActive") or mon["hp"] <= 0:
                continue
            boosts = mon.get("boosts") or {}
            if any(int(boosts.get(k, 0)) != int(getattr(es, f"{attr}_boost"))
                   for k, attr in _BOOSTS.items()):
                out.append(f"{label}.boosts")
            true_vols = set()
            for name, data in (mon.get("volatiles") or {}).items():
                # Showdown keeps the boosted stat inside the volatile; the engine names it
                suffix = _id(data.get("bestStat")) if name in ("protosynthesis",
                                                                 "quarkdrive") else ""
                true_vols.add(_id(name) + suffix)
            true_vols -= _IGNORED_VOLATILES
            vols = {_id(v) for v in (es.volatile_statuses or ())} - _IGNORED_VOLATILES
            if "substitute" in true_vols or int(es.substitute_health or 0) > 0:
                if ("substitute" in true_vols) != (int(es.substitute_health or 0) > 0):
                    out.append(f"{label}.substitute")
            true_vols.discard("substitute")
            vols.discard("substitute")
            for name in sorted(true_vols - vols):
                out.append(f"{label}.volatile_missing.{name}")
            for name in sorted(vols - true_vols):
                out.append(f"{label}.volatile_extra.{name}")
            if label == "me":
                disabled = {_id(slot["id"]) for slot in mon["moveSlots"]
                            if slot.get("disabled") or int(slot["pp"]) == 0}
                engine_disabled = {_id(m.id) for m in em.moves
                                   if _id(m.id) not in ("", "none")
                                   and (bool(m.disabled) or int(m.pp) == 0)}
                if disabled != engine_disabled:
                    out.append("me.move_unavailable")
    return out


def run(pool_path: Path, games: int, seed: int = 0, max_turns: int = 60,
        fmt: str = "gen9ou") -> dict[str, Any]:
    from fog.constants.tables import TABLES
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.showdown_local_runner import _choice
    from fog.env.state_recon import reconstruct_battle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import load_pool, to_packed
    from fog.search.engine_mcts import ShowdownEngineMCTSController, apply_request_overlay

    pool = load_pool(pool_path)
    packed = [to_packed(paste) for paste in pool]
    tokens = [truth_tokens_from_paste(paste, TABLES) for paste in pool]
    rng = random.Random(seed)
    scripts = ("random", "simpleheuristics", "maxdamage")
    positions = bad = errors = 0
    tally: Counter = Counter()
    error_reasons: Counter = Counter()
    examples: dict[str, str] = {}

    with ShowdownDirectClient() as client:
        for game in range(games):
            one, two = rng.randrange(len(pool)), rng.randrange(len(pool))
            controllers = (scripts[game % 3], scripts[(game // 3) % 3])
            trackers = ShowdownEngineMCTSController(k=1, iterations=1)
            tag = f"battle-reconfuzz-{game}"
            local = LocalShowdownBattle.create(client, {
                "formatid": fmt, "seed": f"sodium,{rng.getrandbits(256):064x}",
                "strictChoices": True,
                "p1": {"name": "One", "team": packed[one]},
                "p2": {"name": "Two", "team": packed[two]},
            }, battle_tag=tag)
            try:
                for _ in range(1000):
                    if local.ended or int(local.p1.turn) >= max_turns:
                        break
                    requests = local.update.requests
                    truth = None
                    for role, view, world in (("p1", local.p1, tokens[two]),
                                              ("p2", local.p2, tokens[one])):
                        request = requests.get(role)
                        if not request or request.get("wait") or request.get("teamPreview"):
                            continue
                        if truth is None:
                            truth = local.state
                            truth = json.loads(truth) if isinstance(truth, str) else truth
                        positions += 1
                        try:
                            tracker = trackers._tracker(view)
                            state = reconstruct_battle(view, world, TABLES, tracker,
                                                       role=role).state
                            state = apply_request_overlay(state, view, role)
                            issues = sorted(set(compare(truth, state, role)))
                        except Exception as exc:  # noqa: BLE001 - a failed rebuild is a finding
                            errors += 1
                            error_reasons[f"{type(exc).__name__}: {exc}"[:160]] += 1
                            continue
                        if issues:
                            bad += 1
                            tally.update(issues)
                            for issue in issues:
                                examples.setdefault(
                                    issue, f"game {game} ({one} vs {two}) turn "
                                           f"{int(view.turn)} as {role}")
                    choices = [
                        _choice(controllers[0], local.p1, requests.get("p1"), rng),
                        _choice(controllers[1], local.p2, requests.get("p2"), rng),
                    ]
                    update = local.step(choices[0], choices[1], include_state=False)
                    if update.choice_errors:
                        raise RuntimeError(f"scripted choice rejected: {update.choice_errors}")
            finally:
                local.close()
    return {
        "pool": str(pool_path), "games": games, "seed": seed, "positions": positions,
        "positions_with_mismatch": bad, "reconstruction_errors": errors,
        "error_reasons": dict(error_reasons.most_common(10)),
        "mismatch_positions_by_field": dict(tally.most_common()),
        "first_example": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.pool, args.games, args.seed, args.max_turns, args.format)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    positions = max(1, report["positions"])
    print(f"positions checked    : {report['positions']}")
    print(f"with any mismatch    : {report['positions_with_mismatch']} "
          f"({100 * report['positions_with_mismatch'] / positions:.1f}%)")
    print(f"reconstruction errors: {report['reconstruction_errors']}")
    for reason, count in report["error_reasons"].items():
        print(f"  {count:5d}  {reason}")
    print("mismatch fields (positions affected; first example):")
    for name, count in report["mismatch_positions_by_field"].items():
        print(f"  {name:<44} {count:5d}  {report['first_example'][name]}")


if __name__ == "__main__":
    main()
