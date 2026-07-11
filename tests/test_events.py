"""Tests for the per-turn event mini-parser (encoding/events.py) on fixture protocol events."""
from __future__ import annotations

import json

import pytest

from fog.constants import schema
from fog.constants.build import build_all
from fog.constants.tables import Tables
from fog.encoding.dims import DIMS
from fog.encoding.events import (
    event_mon_key,
    event_role,
    feed_ledger_from_events,
    hp_fraction,
    parse_turn_events,
)
from fog.encoding.reveals import RevealLedger, tokens_from_views
from tests.test_build_tables import DUMP, USAGE


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("generated_events")
    build_all(DUMP, USAGE, out_dir=out_dir)
    (out_dir / "spreads.json").write_text(json.dumps({"species": {}}))
    return Tables(gen_dir=out_dir)


def test_token_helpers():
    assert event_role("p1a: Garchomp") == "p1"
    assert event_role("p2b: X") == "p2"
    assert event_role("[from] item") is None
    assert event_mon_key("p1a: Garchomp") == "p1: Garchomp"
    assert hp_fraction("163/323") == pytest.approx(163 / 323)
    assert hp_fraction("57/100 brn") == pytest.approx(0.57)
    assert hp_fraction("0 fnt") == 0.0
    assert hp_fraction("???") is None


# A hand-written turn: I (p1, Great Tusk) use Earthquake first, it's super effective, crits,
# deals 100->40 = 60% to Garchomp; Garchomp terastallizes and answers with Earthquake for
# 80% -> 45% = 35% on me; my Leftovers... opponent's item gets knocked... keep it simple:
# opponent's ability is revealed; nobody faints.
TURN = [
    ["", "move", "p1a: Great Tusk", "Earthquake", "p2a: Garchomp"],
    ["", "-supereffective", "p2a: Garchomp"],
    ["", "-crit", "p2a: Garchomp"],
    ["", "-damage", "p2a: Garchomp", "40/100"],
    ["", "-terastallize", "p2a: Garchomp", "Water"],
    ["", "move", "p2a: Garchomp", "Earthquake", "p1a: Great Tusk"],
    ["", "-ability", "p2a: Garchomp", "Rough Skin"],
    ["", "-damage", "p1a: Great Tusk", "45/100"],
]
HP_SEED = {"p2: Garchomp": 1.0, "p1: Great Tusk": 0.80}


def test_parse_turn_record(tables):
    rec = parse_turn_events(TURN, "p1", tables, my_action_idx=0, hp_seed=dict(HP_SEED))
    assert rec.shape == (DIMS.turn_dim,)
    # my action one-hot
    assert rec[30] == 1.0 and rec[30:44].sum() == 1.0
    # opp action: moved with earthquake, tera'd, not switched, not unseen
    eq = tables.move_index("earthquake")
    assert rec[44] == 1.0 and rec[45] == pytest.approx(eq / 1000)
    assert rec[46] == 0.0 and rec[48] == 1.0 and rec[49] == 0.0
    # resolution: dmg dealt 0.6, taken 0.35, my crit, eff mine=super(idx3), eff opp=neutral(idx2)
    assert rec[84] == pytest.approx(0.60, abs=1e-6)
    assert rec[85] == pytest.approx(0.35, abs=1e-6)
    assert rec[86] == 1.0 and rec[87] == 0.0
    assert rec[84 + 4 + 3] == 1.0   # eff_mine super
    assert rec[84 + 9 + 2] == 1.0   # eff_opp neutral (damage without marker)
    assert rec[84 + 17] == 1.0      # opp ability revealed
    assert rec[84 + 16] == 0.0      # no opp item reveal
    assert rec[84 + 20] == 1.0      # I moved first
    assert rec[84 + 21] == pytest.approx(round(0.60 * 255) / 255)


def test_preemption_flag_and_faint(tables):
    events = [
        ["", "move", "p1a: Great Tusk", "Earthquake", "p2a: Garchomp"],
        ["", "-damage", "p2a: Garchomp", "0 fnt"],
        ["", "faint", "p2a: Garchomp"],
    ]
    rec = parse_turn_events(events, "p1", tables, my_action_idx=0, hp_seed={"p2: Garchomp": 0.3})
    assert rec[49] == 1.0            # opponent never acted -> <UNSEEN>
    assert rec[84 + 19] == 1.0       # opp faint
    assert rec[84] == pytest.approx(0.3)


def test_retro_my_move_slot_inference(tables):
    movesets = {"p1: Great Tusk": {"earthquake": 2}}
    rec = parse_turn_events(TURN, "p1", tables, my_movesets=movesets, hp_seed=dict(HP_SEED))
    assert rec[32] == 1.0 and rec[30:44].sum() == 1.0


def test_switch_seeds_hp_and_opp_switch_encoding(tables):
    events = [
        ["", "switch", "p2a: Gabite", "Gabite, L84, M", "75/100"],
        ["", "move", "p1a: Great Tusk", "Earthquake", "p2a: Gabite"],
        ["", "-damage", "p2a: Gabite", "25/100"],
    ]
    rec = parse_turn_events(events, "p1", tables, my_action_idx=0)
    assert rec[46] == 1.0
    assert rec[47] == pytest.approx(tables.species_index("gabite") / 1000)
    assert rec[84] == pytest.approx(0.50)  # 75 -> 25, seeded by the switch condition


def test_ledger_feeding_end_to_end(tables):
    ledger = RevealLedger()
    my_turn_events = [
        ["", "move", "p1a: Great Tusk", "Stealth Rock", "p2a: Garchomp"],
        ["", "-enditem", "p1a: Great Tusk", "Booster Energy"],
        ["", "-ability", "p1a: Great Tusk", "Protosynthesis"],
        ["", "-terastallize", "p1a: Great Tusk", "Ground"],
        # opponent-side events must NOT be recorded as my reveals
        ["", "move", "p2a: Garchomp", "Earthquake", "p1a: Great Tusk"],
    ]
    feed_ledger_from_events(my_turn_events, "p1", ledger, tables)
    views = ledger.views(["p1: Great Tusk"], [tables.species_index("greattusk")])
    t = tokens_from_views(views + [None] * 0)[:1]
    assert t[0, 3] == tables.move_index("stealthrock")
    assert t[0, 1] == tables.item_index(schema.NO_ITEM)      # consumed Booster -> known-absent
    assert t[0, 2] == tables.ability_index("protosynthesis")
    assert t[0, 8] == schema.TYPE_INDEX["ground"]
    # Garchomp's move must not appear anywhere in MY ledger
    assert len(ledger.views(["p2: Garchomp"], [0])[0].moves) == 0


def test_snapshot_from_mock_battle(tables):
    from fog.encoding.events import build_snapshot

    class Status:
        name = "BRN"

    class Mon:
        def __init__(self, species, hp, status=None, boosts=None):
            self.species = species
            self.current_hp_fraction = hp
            self.status = status
            self.boosts = boosts or {}

    class Battle:
        active_pokemon = Mon("greattusk", 0.8, Status(), {"atk": 2, "spe": 1})
        opponent_active_pokemon = Mon("garchomp", 0.4)
        weather = {}
        fields = {}

    snap = build_snapshot(Battle(), tables)
    assert snap.shape == (30,)
    assert snap[2] == pytest.approx(0.8) and snap[3] == pytest.approx(0.4)
    assert snap[4 + schema.Status.BRN.value] == 1.0
    assert snap[18] == pytest.approx(3 / 12)
