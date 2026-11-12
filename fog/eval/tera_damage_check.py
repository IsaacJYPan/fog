"""Compare Terastallization damage on the Rust engine with pinned Showdown (ROADMAP P1.2c).

For each case one attacker uses one move into one defender, with and without Terastallization on
either side. Three numbers are compared per case:

* **Showdown**: the exact set of non-critical damage rolls, from ``ShowdownDirectClient.enumerate``
  on a one-versus-one custom game (every chance outcome is an explicit branch, so nothing is
  sampled);
* **engine, Tera this turn**: the non-critical ``Damage`` instruction of
  ``poke_engine.generate_instructions(state, "<move>-tera", ...)``, which is the engine's average
  roll, ``floor(max_roll * 0.925)``;
* **engine, already terastallized**: ``poke_engine.calculate_damage`` (the maximum roll) on a state
  whose Pokemon carries ``terastallized=True`` with its original ``types`` kept. This is the state
  ``fog.env.state_recon.reconstruct_battle`` builds after a Tera has happened.

Cases must not knock the defender out (a faint hides the damage amount). A case agrees when the
engine's maximum roll is within ``--tolerance`` (default 1 HP, for rounding order) of Showdown's
maximum roll and the in-turn instruction equals ``floor(max * 0.925)`` of the same maximum. The
tool reports; it does not change anything. Known disagreement: the engine does not model Tera
Stellar (it applies no Stellar bonus), which the last two default cases show.

    python -m fog.eval.tera_damage_check --out runs/tera_damage_check.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DAMAGE_LINE = re.compile(r"^\|-damage\|(p[12])a: [^|]*\|(\d+)/(\d+)")
_ENGINE_DAMAGE = re.compile(r"^Damage Side(One|Two): (\d+)$")


@dataclass(frozen=True)
class Case:
    label: str
    attacker: str
    move: str
    attacker_tera: str | None = None      # None: the attacker does not terastallize
    defender: str = "Snorlax"
    defender_tera: str | None = None      # None: the defender does not terastallize
    attacker_ability: str = "No Ability"
    defender_ability: str = "No Ability"
    attacker_item: str = ""


def default_cases() -> list[Case]:
    """Same-type bonus, the 60 base-power floor, Adaptability and defensive Tera typing."""
    tusk = "Great Tusk"            # Ground / Fighting
    craw = "Crawdaunt"             # Water / Dark
    out = [
        Case("no Tera, original type", tusk, "Earthquake"),
        Case("no Tera, off type", tusk, "Liquidation"),
        Case("new Tera type, Tera-type move", tusk, "Liquidation", "Water"),
        Case("new Tera type, original-type move", tusk, "Earthquake", "Water"),
        Case("new Tera type, unrelated move", tusk, "Body Slam", "Water"),
        Case("Tera matches original type", tusk, "Earthquake", "Ground"),
        Case("Tera matches original, other original type", tusk, "Brick Break", "Ground"),
        Case("floor: 40 BP move of a new Tera type", tusk, "Water Gun", "Water"),
        Case("floor: 20 BP move, Tera matches original", tusk, "Mud-Slap", "Ground"),
        Case("no floor without Tera", tusk, "Water Gun"),
        Case("floor does not apply to priority", tusk, "Aqua Jet", "Water"),
        Case("Adaptability, no Tera", craw, "Liquidation", attacker_ability="Adaptability"),
        Case("Adaptability, Tera matches original", craw, "Liquidation", "Water",
             attacker_ability="Adaptability"),
        Case("Adaptability, new Tera type", craw, "Fire Punch", "Fire",
             attacker_ability="Adaptability"),
        Case("Adaptability, new Tera, original-type move", craw, "Liquidation", "Fire",
             attacker_ability="Adaptability"),
        Case("defender Tera resists", tusk, "Liquidation", defender_tera="Water"),
        Case("defender Tera is weak", tusk, "Earthquake", defender_tera="Steel"),
        Case("defender Tera is immune", tusk, "Body Slam", defender_tera="Ghost"),
        Case("defender Tera removes a weakness", tusk, "Brick Break", defender_tera="Ghost"),
        Case("both terastallize", tusk, "Liquidation", "Water", defender_tera="Fire"),
        Case("Tera Stellar, original-type move", tusk, "Earthquake", "Stellar"),
        Case("Tera Stellar, off-type move", tusk, "Liquidation", "Stellar"),
    ]
    return out


def _paste(species: str, moves: list[str], tera: str, ability: str, item: str,
           speed_evs: int) -> str:
    head = f"{species} @ {item}" if item else species
    lines = [head, f"Ability: {ability}", f"Tera Type: {tera}", f"EVs: {speed_evs} Spe",
             "Serious Nature", *[f"- {m}" for m in moves]]
    return "\n".join(lines) + "\n"


def _team(species: str, moves: list[str], tera: str, ability: str, item: str,
          speed_evs: int) -> list[dict[str, Any]]:
    return [{"species": species, "ability": ability, "item": item, "moves": moves,
             "nature": "Serious", "evs": {"spe": speed_evs}, "level": 100, "teraType": tera}]


def showdown_rolls(client: Any, case: Case) -> dict[str, Any]:
    """Exact non-critical damage rolls of ``case`` on pinned Showdown."""
    attacker_tera = case.attacker_tera or "Normal"
    defender_tera = case.defender_tera or "Normal"
    root = client.create_state({
        "formatid": "gen9customgame@@@!teampreview",
        "seed": "sodium," + "1" * 64,
        "strictChoices": True,
        "p1": {"name": "Attacker", "team": _team(case.attacker, [case.move], attacker_tera,
                                                 case.attacker_ability, case.attacker_item, 252)},
        "p2": {"name": "Defender", "team": _team(case.defender, ["Splash"], defender_tera,
                                                 case.defender_ability, "", 0)},
    })
    choices = ("move 1" + (" terastallize" if case.attacker_tera else ""),
               "move 1" + (" terastallize" if case.defender_tera else ""))
    result = client.enumerate(root["state"], choices, include_state=False)
    rolls: set[int] = set()
    immune = False
    for branch in result.branches:
        log = branch.player_logs["p2"]
        text = "\n".join(log)
        if "|-crit|" in text or "|-miss|" in text:
            continue
        if "|-immune|" in text:
            immune = True
            continue
        for line in log:
            match = _DAMAGE_LINE.match(line)
            if match and match.group(1) == "p2":
                rolls.add(int(match.group(3)) - int(match.group(2)))
                break
    if immune and not rolls:
        rolls = {0}
    return {"rolls": sorted(rolls), "exact": bool(result.exact)}


def _engine_state(case: Case, *, attacker_terastallized: bool, defender_terastallized: bool):
    import poke_engine as pe

    from fog.env.engine_battle import EngineBattle, _rebuild_pokemon

    battle = EngineBattle(
        _paste(case.attacker, [case.move], case.attacker_tera or "Normal",
               case.attacker_ability, case.attacker_item, 252),
        _paste(case.defender, ["Splash"], case.defender_tera or "Normal",
               case.defender_ability, "", 0),
        seed=0, battle_tag="tera-damage-check",
    )
    state = battle.state
    one = _rebuild_pokemon(state.side_one.pokemon[0], terastallized=attacker_terastallized)
    two = _rebuild_pokemon(state.side_two.pokemon[0], terastallized=defender_terastallized)
    return pe.State(side_one=pe.Side(pokemon=[one]), side_two=pe.Side(pokemon=[two]))


def engine_numbers(case: Case) -> dict[str, Any]:
    """Engine maximum roll on an already-terastallized state, and the in-turn Tera instruction."""
    import poke_engine as pe

    from fog.env.engine_battle import to_id

    move = to_id(case.move)
    done = _engine_state(case, attacker_terastallized=bool(case.attacker_tera),
                         defender_terastallized=bool(case.defender_tera))
    max_roll = int(pe.calculate_damage(done, move, "splash", True)[0][0])

    fresh = _engine_state(case, attacker_terastallized=False, defender_terastallized=False)
    branches = pe.generate_instructions(
        fresh, move + ("-tera" if case.attacker_tera else ""),
        "splash" + ("-tera" if case.defender_tera else ""),
    )
    amounts = []
    for branch in branches:
        hit = [int(m.group(2)) for m in
               (_ENGINE_DAMAGE.match(str(i)) for i in branch.instruction_list)
               if m and m.group(1) == "Two"]
        amounts.append(hit[0] if hit else 0)
    # the non-critical hit is the smallest positive damage; 0 everywhere means immune or a miss
    positive = [a for a in amounts if a > 0]
    return {"max_roll": max_roll, "in_turn_damage": min(positive) if positive else 0}


def check_case(client: Any, case: Case, tolerance: int = 1) -> dict[str, Any]:
    showdown = showdown_rolls(client, case)
    engine = engine_numbers(case)
    sd_max = max(showdown["rolls"]) if showdown["rolls"] else None
    expected_average = int(sd_max * 0.925) if sd_max is not None else None
    agrees = (
        sd_max is not None
        and abs(engine["max_roll"] - sd_max) <= tolerance
        and abs(engine["in_turn_damage"] - expected_average) <= tolerance
    )
    return {
        **asdict(case),
        "showdown_min": min(showdown["rolls"]) if showdown["rolls"] else None,
        "showdown_max": sd_max,
        "showdown_exact": showdown["exact"],
        "engine_max_roll": engine["max_roll"],
        "engine_in_turn_damage": engine["in_turn_damage"],
        "expected_in_turn_damage": expected_average,
        "ratio": (engine["max_roll"] / sd_max) if sd_max else None,
        "agrees": bool(agrees),
    }


def run(cases: list[Case] | None = None, tolerance: int = 1) -> dict[str, Any]:
    from fog.env.showdown_direct import ShowdownDirectClient

    cases = default_cases() if cases is None else cases
    with ShowdownDirectClient() as client:
        rows = [check_case(client, case, tolerance) for case in cases]
    return {"cases": rows, "agree": sum(r["agrees"] for r in rows), "total": len(rows),
            "tolerance": tolerance}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Case | Attacker (Tera) | Move | Defender (Tera) | Showdown rolls | Engine max |"
        " Engine in-turn | Agrees |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in report["cases"]:
        ability = f", {r['attacker_ability']}" if r["attacker_ability"] != "No Ability" else ""
        lines.append(
            f"| {r['label']} | {r['attacker']} ({r['attacker_tera'] or 'none'}{ability}) |"
            f" {r['move']} | {r['defender']} ({r['defender_tera'] or 'none'}) |"
            f" {r['showdown_min']}–{r['showdown_max']} | {r['engine_max_roll']} |"
            f" {r['engine_in_turn_damage']} | {'yes' if r['agrees'] else '**no**'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tolerance", type=int, default=1)
    args = parser.parse_args()
    report = run(tolerance=args.tolerance)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown(report))
    print(f"\n{report['agree']}/{report['total']} cases agree within {args.tolerance} HP")


if __name__ == "__main__":
    main()
