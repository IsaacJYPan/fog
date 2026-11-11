"""Focused tests for the Showdown reconstruction fixes: PP at 0, Rest sleep, used-Tera state and
Tera typing (ROADMAP P1.2c / P1.2d). Skipped without poke-engine."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

P_ONE = """Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Water
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Knock Off

Kingambit @ Black Glasses
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance
"""

P_TWO = """Tyranitar @ Leftovers
Ability: Sand Stream
Tera Type: Ghost
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave

Hydreigon @ Choice Specs
Ability: Levitate
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- U-turn
"""


def _tables():
    from fog.constants.tables import TABLES

    return TABLES


def _battle(seed=0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(P_ONE, P_TWO, seed=seed, battle_tag=f"recon-fix-{seed}")


def _reconstruct(battle, role="p1"):
    from fog.env.state_recon import reconstruct_battle

    opp = "p2" if role == "p1" else "p1"
    return reconstruct_battle(battle.view(role), battle.truth_tokens(opp), _tables(),
                              battle.trackers[role], role=role).state


def _root_choices(state, side="p1"):
    import poke_engine as pe

    result = pe.monte_carlo_tree_search(state, iterations=64)
    entries = result.side_one if side == "p1" else result.side_two
    return [str(e.move_choice) for e in entries]


def _own_mon(**kw):
    """A poke-env-shaped own mon (Great Tusk) for direct `_engine_mon_from_self` calls."""
    base = dict(
        species="greattusk", level=100, current_hp_fraction=1.0, fainted=False,
        stats={"hp": 404, "atk": 268, "def": 299, "spa": 127, "spd": 143, "spe": 339},
        ability="protosynthesis", item="heavydutyboots", status=None, status_counter=0,
        is_terastallized=False, tera_type=None, types=["ground", "fighting"],
        moves={"headlongrush": SimpleNamespace(current_pp=0, max_pp=8, disabled=False),
               "rapidspin": SimpleNamespace(current_pp=None, max_pp=64, disabled=False),
               "knockoff": SimpleNamespace(current_pp=5, max_pp=32, disabled=False)},
        _last_request={"teraType": "Water"},
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------------------------------- PP
def test_zero_pp_is_rebuilt_as_zero_for_own_and_opponent_moves():
    from fog.env.state_recon import _engine_mon_from_self, _engine_mon_from_world

    mon = _engine_mon_from_self(_own_mon(), _tables())
    pp = {str(m.id): int(m.pp) for m in mon.moves}
    assert pp == {"headlongrush": 0, "rapidspin": 64, "knockoff": 5}

    battle = _battle()
    world = battle.truth_tokens("p2")
    view_mon = SimpleNamespace(
        species="tyranitar", level=100, current_hp_fraction=1.0, fainted=False, status=None,
        status_counter=0, is_terastallized=False, types=["rock", "dark"],
        moves={"earthquake": SimpleNamespace(current_pp=0, max_pp=16, disabled=False)},
    )
    opp = _engine_mon_from_world(world[0], view_mon, 100, _tables())
    opp_pp = {str(m.id): int(m.pp) for m in opp.moves}
    assert opp_pp["earthquake"] == 0
    assert all(v > 0 for k, v in opp_pp.items() if k != "earthquake")


# ---------------------------------------------------------------------------------------- sleep
def _events_view(events):
    return SimpleNamespace(observations={1: SimpleNamespace(events=events)})


def test_rest_sleep_is_read_from_the_public_log():
    from fog.env.state_recon import _rest_attempts

    events = [
        ["", "move", "p1a: Great Tusk", "Rest", "p1a: Great Tusk"],
        ["", "-status", "p1a: Great Tusk", "slp", "[from] move: Rest"],
        ["", "-status", "p2a: Tyranitar", "slp"],
        ["", "cant", "p1a: Great Tusk", "slp"],
        ["", "move", "p1a: Great Tusk", "Sleep Talk", "p1a: Great Tusk"],
        ["", "move", "p1a: Great Tusk", "Rapid Spin", "p2a: Tyranitar", "[from]Sleep Talk"],
    ]
    assert _rest_attempts(_events_view(events)) == {"p1: Great Tusk": 2}
    woke = events + [["", "-curestatus", "p1a: Great Tusk", "slp", "[msg]"]]
    assert _rest_attempts(_events_view(woke)) == {}


def test_sleep_fields_split_rest_and_ordinary_sleep():
    from fog.env.state_recon import _sleep_fields

    mon = SimpleNamespace(status_counter=4)
    # Rest: 3 right after Rest, one less per attempt, never below 1 (wakes on that attempt)
    assert _sleep_fields(mon, "sleep", 0) == {"sleep_turns": 0, "rest_turns": 3}
    assert _sleep_fields(mon, "sleep", 2) == {"sleep_turns": 0, "rest_turns": 1}
    assert _sleep_fields(mon, "sleep", 5) == {"sleep_turns": 0, "rest_turns": 1}
    # ordinary sleep: counter clamped to the engine's Gen 9 maximum
    assert _sleep_fields(mon, "sleep", None) == {"sleep_turns": 3, "rest_turns": 0}
    # a toxic counter never leaks into the sleep counter
    assert _sleep_fields(mon, "toxic", None) == {"sleep_turns": 0, "rest_turns": 0}
    # an EngineView mon carries the engine's own counter
    assert _sleep_fields(SimpleNamespace(rest_turns=2, status_counter=0), "sleep", None) == {
        "sleep_turns": 0, "rest_turns": 2}


def test_rest_sleeping_own_mon_searches_without_panic():
    import poke_engine as pe

    from fog.env.state_recon import _engine_mon_from_self

    tables = _tables()
    asleep = _engine_mon_from_self(_own_mon(status="slp", status_counter=4), tables, 1)
    assert (str(asleep.status).lower(), int(asleep.rest_turns), int(asleep.sleep_turns)) == (
        "sleep", 2, 0)
    battle = _battle()
    state = battle.state
    mons = list(state.side_one.pokemon)
    mons[0] = asleep
    from fog.env.engine_battle import _rebuild_side

    side = _rebuild_side(state.side_one, mons, active_index="0")
    new = pe.State(side_one=side, side_two=state.side_two)
    assert _root_choices(new)


def test_engine_view_round_trip_keeps_rest_turns():
    from fog.env.engine_battle import _rebuild_pokemon, _rebuild_side

    battle = _battle()
    side = battle.state.side_one
    mons = list(side.pokemon)
    mons[0] = _rebuild_pokemon(mons[0], status="sleep", rest_turns=2, sleep_turns=0)
    battle.state = battle._rebuild_state(_rebuild_side(side, mons), battle.state.side_two)
    rebuilt = _reconstruct(battle).side_one.pokemon[0]
    assert (int(rebuilt.rest_turns), int(rebuilt.sleep_turns)) == (2, 0)


# ----------------------------------------------------------------------------------------- Tera
def _tera_then_switch(battle):
    """p1 terastallizes Great Tusk, then switches it out for Kingambit."""
    battle.step({"p1": battle.engine_action("p1", 4), "p2": battle.engine_action("p2", 0)})
    assert battle.tera_used["p1"]
    if battle.finished or "p1" not in battle.pending_sides():
        pytest.skip("battle state does not allow the follow-up switch")
    switch = next(i for i in range(8, 13) if battle.action_mask("p1")[i])
    moves = {"p1": battle.engine_action("p1", switch)}
    if "p2" in battle.pending_sides():
        moves["p2"] = battle.engine_action("p2", 0)
    battle.step(moves)
    return battle


def test_used_tera_survives_benching_in_rebuilt_state():
    battle = _tera_then_switch(_battle(seed=3))
    state = _reconstruct(battle)
    tusk = next(p for p in state.side_one.pokemon if str(p.id).lower() == "greattusk")
    assert bool(tusk.terastallized)
    assert str(state.side_one.pokemon[int(state.side_one.active_index)].id).lower() != "greattusk"
    assert not any(c.endswith("-tera") for c in _root_choices(state, "p1"))


def test_used_tera_survives_fainting_in_rebuilt_state():
    from fog.env.engine_battle import _rebuild_pokemon, _rebuild_side

    battle = _tera_then_switch(_battle(seed=5))
    side = battle.state.side_one
    mons = [
        _rebuild_pokemon(p, hp=0) if str(p.id).lower() == "greattusk" else p for p in side.pokemon
    ]
    battle.state = battle._rebuild_state(_rebuild_side(side, mons), battle.state.side_two)
    state = _reconstruct(battle)
    tusk = next(p for p in state.side_one.pokemon if str(p.id).lower() == "greattusk")
    assert int(tusk.hp) == 0 and bool(tusk.terastallized)
    assert not any(c.endswith("-tera") for c in _root_choices(state, "p1"))


def test_opponent_tera_keeps_original_types_and_blocks_its_tera():
    battle = _battle(seed=7)
    battle.step({"p1": battle.engine_action("p1", 0), "p2": battle.engine_action("p2", 4)})
    if not battle.tera_used["p2"] or battle.finished:
        pytest.skip("opponent did not terastallize")
    state = _reconstruct(battle, role="p1")
    ttar = next(p for p in state.side_two.pokemon if str(p.id).lower() == "tyranitar")
    assert bool(ttar.terastallized) and str(ttar.tera_type).lower() == "ghost"
    assert tuple(str(t).lower() for t in ttar.types) == ("rock", "dark")
    assert not any(c.endswith("-tera") for c in _root_choices(state, "p2"))
    # and the engine's own fork path agrees
    fork = battle.fork_for_search("p1", battle.truth_tokens("p2"), seed=1)
    fttar = next(p for p in fork.state.side_two.pokemon if str(p.id).lower() == "tyranitar")
    assert tuple(str(t).lower() for t in fttar.types) == ("rock", "dark")


def test_own_terastallized_mon_keeps_original_types_and_request_tera_type():
    from fog.env.state_recon import _engine_mon_from_self

    tables = _tables()
    # poke-env reports the Tera type as `types` once terastallized
    tera = _engine_mon_from_self(
        _own_mon(is_terastallized=True, tera_type="water", types=["water"]), tables)
    assert tuple(str(t).lower() for t in tera.types) == ("ground", "fighting")
    assert bool(tera.terastallized) and str(tera.tera_type).lower() == "water"
    # before use poke-env has no Tera type; the request's teraType supplies it
    plain = _engine_mon_from_self(_own_mon(), tables)
    assert str(plain.tera_type).lower() == "water" and not bool(plain.terastallized)


def _damage_state(terastallized: bool, tera_type: str):
    import poke_engine as pe

    attacker = pe.Pokemon(
        id="greattusk", level=100, types=("ground", "fighting"),
        base_types=("ground", "fighting"), hp=300, maxhp=300, attack=300, defense=200,
        special_attack=100, special_defense=200, speed=300, ability="none", item="none",
        moves=[pe.Move(id=m) for m in ("earthquake", "closecombat", "liquidation", "bodyslam")],
        terastallized=terastallized, tera_type=tera_type,
    )
    defender = pe.Pokemon(
        id="snorlax", level=100, types=("normal", "typeless"), base_types=("normal", "typeless"),
        hp=400, maxhp=400, attack=200, defense=200, special_attack=100, special_defense=200,
        speed=100, ability="none", item="none", moves=[pe.Move(id="splash")],
    )
    return pe.State(side_one=pe.Side(pokemon=[attacker]), side_two=pe.Side(pokemon=[defender]))


def test_tera_same_type_bonus_with_original_types_kept():
    """Gen 9: 1.5x for original types, 1.5x for a new Tera type, 2.0x when Tera matches."""
    import poke_engine as pe

    moves = ("earthquake", "closecombat", "liquidation", "bodyslam")

    def dmg(state, move):
        return pe.calculate_damage(state, move, "splash", True)[0][0]

    plain = _damage_state(False, "normal")
    # unboosted reference per move (Ground/Fighting moves carry 1.5x without Tera)
    ref = {m: dmg(plain, m) / (1.5 if m in ("earthquake", "closecombat") else 1.0) for m in moves}

    def ratios(state):
        return {m: dmg(state, m) / ref[m] for m in moves}

    same = ratios(_damage_state(True, "ground"))
    new = ratios(_damage_state(True, "water"))
    expect_same = {"earthquake": 2.0, "closecombat": 1.5, "liquidation": 1.0, "bodyslam": 1.0}
    expect_new = {"earthquake": 1.5, "closecombat": 1.5, "liquidation": 1.5, "bodyslam": 1.0}
    for m in moves:
        assert same[m] == pytest.approx(expect_same[m], rel=0.01), m
        assert new[m] == pytest.approx(expect_new[m], rel=0.01), m
    assert np.isfinite(list(same.values())).all()


def test_used_tera_from_the_public_log_backs_up_the_parser_flag():
    from fog.env.state_recon import _engine_mon_from_self, _pre_tera_species, _tera_from_log

    events = [["", "-terastallize", "p1a: Great Tusk", "Water"],
              ["", "-terastallize", "p2a: Ogerpon", "Water"]]
    log = _tera_from_log(_events_view(events))
    assert log == {"p1: Great Tusk": "water", "p2: Ogerpon": "water"}
    mon = _engine_mon_from_self(_own_mon(is_terastallized=False), _tables(),
                                logged_tera=log["p1: Great Tusk"])
    assert bool(mon.terastallized) and str(mon.tera_type).lower() == "water"
    assert tuple(str(t).lower() for t in mon.types) == ("ground", "fighting")
    # a Tera forme in the log maps back to the sampled (pre-Tera) forme
    assert _pre_tera_species("ogerponwellspringtera", _tables()) == "ogerponwellspring"
    assert _pre_tera_species("greattusk", _tables()) == "greattusk"
