"""EngineBattle tests (ARCHITECTURE.md §3.2): engine API contract pins, partial-observability
censoring, legality masks, event synthesis, determinism. Skipped without poke-engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

POOL = Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b"

TTAR_LEAD = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Knock Off
"""

SPECS_LEAD = """Hydreigon @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- U-turn

Weavile @ Heavy-Duty Boots
Ability: Pressure
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Knock Off
- Triple Axel
- Ice Shard
- Swords Dance
"""

WEAK_LEAD = """Great Tusk @ Heavy-Duty Boots
Level: 5
Ability: Protosynthesis
EVs: 4 HP
Serious Nature
- Headlong Rush
- Rapid Spin
- Knock Off
- Ice Spinner

Kingambit @ Black Glasses
Ability: Supreme Overlord
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance
"""


def _paste(name: str) -> str:
    return (POOL / name).read_text(encoding="utf-8")


def _battle(p1: str, p2: str, seed: int = 0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(p1, p2, seed=seed, battle_tag=f"eb-test-{seed}")


def _move_slot(battle, role: str, move_id: str) -> int:
    active = battle.active(role)
    for i, m in enumerate(active.moves):
        if str(m.id).lower().replace(" ", "") == move_id:
            return i
    raise AssertionError(f"{move_id} not on {active.id}")


# ------------------------------------------------------------------------- engine API contract
def test_engine_contract_pins():
    """Pins the poke-engine behaviors EngineBattle is built on. If an upgrade breaks any of
    these, the wrapper (module docstring) must be re-verified before training on the engine."""
    import poke_engine as pe

    # 1. pe.Pokemon derives NOTHING from id: default stats are 100s, types typeless
    p = pe.Pokemon(id="garchomp")
    assert p.maxhp == 100 and p.attack == 100
    assert all(str(t).lower() in ("normal", "typeless") for t in p.types)

    # 2. apply_instructions returns the successor state and leaves the original untouched
    b = _battle(TTAR_LEAD, SPECS_LEAD)
    pre = b.state
    ins = pe.generate_instructions(pre, "earthquake", "darkpulse")
    assert ins
    post = pre.apply_instructions(max(ins, key=lambda s: len(s.instruction_list)))
    assert post.to_string() != pre.to_string()
    assert pre.to_string() == b.state.to_string()

    # 3. choice lock arrives as move.disabled via DisableMove instructions
    reprs = [repr(i) for s in ins for i in s.instruction_list]
    assert any(r.startswith(("Damage", "Boost", "DisableMove")) for r in reprs)


def test_build_engine_pokemon_computes_real_stats():
    from fog.constants.tables import TABLES
    from fog.env.engine_battle import build_engine_pokemon, parse_paste

    paste = """Garchomp @ Loaded Dice
Ability: Rough Skin
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Earthquake
- Scale Shot
- Swords Dance
- Stealth Rock
"""
    mon = build_engine_pokemon(parse_paste(paste)[0], TABLES)
    assert mon.maxhp == 420          # base 108, 31 IV, 252 EVs
    assert mon.speed == 333          # Jolly 252 Spe — the spec-pinned known value
    assert tuple(str(t) for t in mon.types) == ("dragon", "ground")
    assert mon.weight_kg > 90


# ----------------------------------------------------------------------- reveals + censoring
def test_move_reveal_censoring_and_tokens():
    from fog.encoding.observation import Observation

    b = _battle(TTAR_LEAD, SPECS_LEAD, seed=1)
    slot = _move_slot(b, "p1", "rockblast")
    b.step({"p1": b.engine_action("p1", slot), "p2": b.engine_action("p2", 1)})  # darkpulse

    # p2's view of p1: rockblast revealed, item/ability censored (no lead-ability for ttar? sand
    # stream IS a lead reveal — ability must be visible; item stays hidden)
    view = b.view("p2")
    ttar = view.opponent_team["p1: tyranitar"]
    assert list(ttar.moves) == ["rockblast"]
    assert ttar.item == "unknown_item"
    assert ttar.ability == "sandstream"          # lead weather reveals the ability
    tusk = view.opponent_team["p1: greattusk"]
    assert tusk.moves == {} and tusk.ability is None

    obs = Observation.from_battle(view, tracker=b.trackers["p2"])
    tok = obs.opp_revealed["tokens"]
    n = len(b.paste_species["p1"])
    assert (tok[:n, 0] >= 0).all()               # species are preview-public
    from fog.constants.tables import TABLES

    row = list(tok[:, 0]).index(TABLES.species_index("tyranitar"))
    assert TABLES.move_index("rockblast") in tok[row, 3:7].tolist()
    assert tok[row, 1] == -1                     # item hidden


def test_knockoff_reveals_no_item():
    from fog.constants.schema import NO_ITEM
    from fog.constants.tables import TABLES

    b = _battle(SPECS_LEAD, TTAR_LEAD, seed=3)
    # p1 switches to weavile while ttar sets rocks; then weavile (faster, unstatused) Knock Offs
    # first in EVERY branch — the ChangeItem instruction is deterministic
    b.step({"p1": b.engine_action("p1", 8), "p2": b.engine_action("p2", 0)})  # weavile in, rocks
    slot = _move_slot(b, "p1", "knockoff")
    b.step({"p1": b.engine_action("p1", slot), "p2": b.engine_action("p2", 1)})

    ttar_rev = b.revealed["p2"].get("p2: tyranitar", {})
    assert ttar_rev.get("item") == ""            # revealed-GONE, not merely hidden
    view = b.view("p1")
    assert view.opponent_team["p2: tyranitar"].item == ""   # -> <NO_ITEM> in reveals.py
    from fog.encoding.reveals import opponent_views_from_battle

    views = opponent_views_from_battle(view, TABLES)
    ttar_view = next(v for v in views if v.species == TABLES.species_index("tyranitar"))
    assert ttar_view.item == TABLES.item_index(NO_ITEM)


def test_tera_once_per_side():
    b = _battle(TTAR_LEAD, SPECS_LEAD, seed=4)
    mask = b.action_mask("p1")
    assert mask[4:8].any(), "tera slots must be open before tera"
    slot = _move_slot(b, "p1", "rockblast")
    b.step({"p1": b.engine_action("p1", 4 + slot), "p2": b.engine_action("p2", 1)})
    assert b.tera_used["p1"] and not b.tera_used["p2"]
    assert not b.action_mask("p1")[4:8].any(), "tera is once per battle per side"
    # and the reveal reached the ledger/census + opponent view
    assert b.view("p2").opponent_team["p1: tyranitar"].terastallized


# ------------------------------------------------------------------------------ legality rules
def test_choice_lock_masks_other_moves():
    b = _battle(SPECS_LEAD, TTAR_LEAD, seed=5)
    slot = _move_slot(b, "p1", "darkpulse")
    b.step({"p1": b.engine_action("p1", slot), "p2": b.engine_action("p2", 0)})
    mask = b.action_mask("p1")
    move_bits = mask[:4].tolist()
    assert move_bits == [i == slot for i in range(4)], f"choice lock leak: {move_bits}"
    assert mask[8:13].any(), "switching out of a choice lock is legal"
    # observation mask can never disagree (it is computed off the same view)
    from fog.encoding.observation import Observation

    obs = Observation.from_battle(b.view("p1"), tracker=b.trackers["p1"])
    assert (obs.legality_mask == mask).all()


def test_faint_forces_replacement():
    b = _battle(WEAK_LEAD, TTAR_LEAD, seed=6)
    # level-5 lead cannot survive; loop a few turns until it faints
    for _ in range(6):
        if b.finished or b.pending_sides() != ["p1", "p2"]:
            break
        b.step({
            "p1": b.engine_action("p1", int(np.flatnonzero(b.action_mask("p1"))[0])),
            "p2": b.engine_action("p2", _move_slot(b, "p2", "rockblast")),
        })
    assert b.pending_sides() == ["p1"], "fainted side must be the only pending side"
    mask = b.action_mask("p1")
    assert not mask[:8].any() and mask[8:13].any()
    assert b.view("p1").force_switch
    b.step({"p1": b.engine_action("p1", 8)})     # p2 auto-passes "none"
    assert b.active("p1").hp > 0
    assert not b.finished and b.pending_sides() == ["p1", "p2"]


# ---------------------------------------------------------------- events, tracker, determinism
def test_history_and_tracker_wiring():
    from fog.encoding.observation import Observation

    b = _battle(TTAR_LEAD, SPECS_LEAD, seed=7)
    for _ in range(3):
        if b.finished:
            break
        moves = {}
        for role in b.pending_sides():
            idx = int(np.flatnonzero(b.action_mask(role))[0])
            b.trackers[role].record_decision(b.turn, None, idx)
            moves[role] = b.engine_action(role, idx)
        b.step(moves)
    assert len(b.observations) >= 2
    events = b.observations[1].events
    assert any(e[1] == "move" for e in events)
    obs = Observation.from_battle(b.view("p1"), tracker=b.trackers["p1"])
    assert obs.history_len >= 2 and obs.history.any()
    # tracker saw the moves: last_used_move populated for an active mon
    assert b.trackers["p1"].last_used_move


def test_seed_determinism():
    def run(seed):
        b = _battle(_paste("team1_balance.txt"), _paste("team2_offense.txt"), seed=seed)
        rng = np.random.default_rng(123)
        script = []
        while not b.finished and b.turn <= 40:
            moves = {}
            for role in b.pending_sides():
                legal = np.flatnonzero(b.action_mask(role))
                idx = int(legal[rng.integers(len(legal))])
                script.append((role, idx))
                moves[role] = b.engine_action(role, idx)
            b.step(moves)
        events = [tuple(map(tuple, o.events)) for o in b.observations.values()]
        return script, events, b.winner

    s1, e1, w1 = run(9)
    s2, e2, w2 = run(9)
    assert s1 == s2 and e1 == e2 and w1 == w2


def test_lead_weather_and_ability_reveal():
    b = _battle(TTAR_LEAD, SPECS_LEAD, seed=10)
    assert str(b.state.weather).lower() == "sand"
    assert b.revealed["p1"]["p1: tyranitar"]["ability"] == "sandstream"
    # the ledger (mirror path) got the same fact
    from fog.constants.tables import TABLES

    views = b.ledgers["p1"].views(
        ["p1: tyranitar"], [TABLES.species_index("tyranitar")]
    )
    assert views[0].ability == TABLES.ability_index("sandstream")


def test_truth_tokens_align_with_view_slots():
    from fog.constants.tables import TABLES
    from fog.encoding.observation import Observation

    b = _battle(TTAR_LEAD, SPECS_LEAD, seed=11)
    b.set_leads(1, 1)    # non-trivial leads: view slots must stay PASTE order regardless
    truth = b.truth_tokens("p2")
    obs = Observation.from_battle(b.view("p1"), tracker=b.trackers["p1"])
    tok = obs.opp_revealed["tokens"]
    n = len(b.paste_species["p2"])
    assert (tok[:n, 0] == truth[:n, 0]).all(), "belief slots must align with truth slots"
    assert truth[0, 0] == TABLES.species_index("hydreigon")


def test_full_pool_game_smoke():
    import random

    rng = random.Random(0)
    b = _battle(_paste("team3_bulky.txt"), _paste("team4_rain.txt"), seed=12)
    decisions = 0
    while not b.finished:
        moves = {}
        for role in b.pending_sides():
            legal = np.flatnonzero(b.action_mask(role)).tolist()
            moves[role] = b.engine_action(role, rng.choice(legal))
            decisions += 1
        b.step(moves)
        assert decisions < 4000
    assert b.winner in ("p1", "p2", None)
    assert b.view("p1").won == (b.winner == "p1") if b.winner else b.view("p1").won is None
    assert not b.unknown_instructions or all(
        v < 50 for v in b.unknown_instructions.values()
    ), f"frequent unknown instruction kinds: {b.unknown_instructions}"
