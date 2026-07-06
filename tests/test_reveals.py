"""Tests for the information-state core (encoding/reveals.py) against the fixture tables."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fog.constants import schema
from fog.constants.build import build_all
from fog.constants.tables import Tables
from fog.encoding.reveals import (
    RevealLedger,
    SlotView,
    build_belief_input,
    labels_from_truth,
    masks_for_tokens,
    opponent_views_from_battle,
    tokens_from_views,
)
from tests.test_build_tables import DUMP, USAGE


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("generated_reveals")
    build_all(DUMP, USAGE, out_dir=out_dir)
    (out_dir / "spreads.json").write_text(
        json.dumps(
            {
                "species": {
                    "greattusk": [
                        {"nature": "jolly", "evs": [0, 252, 0, 0, 4, 252], "stats": [], "weight": 0.6},
                        {"nature": "impish", "evs": [252, 0, 252, 0, 4, 0], "stats": [], "weight": 0.4},
                    ]
                }
            }
        )
    )
    return Tables(gen_dir=out_dir)


def test_tokens_from_views_packs_moves_earliest_first():
    views = [SlotView(species=1, moves=[10, 3], tera=5)] + [SlotView()] * 5
    t = tokens_from_views(views)
    assert t.shape == (6, 9)
    assert t[0, 0] == 1 and t[0, 3] == 10 and t[0, 4] == 3
    assert t[0, 5] == -1 and t[0, 6] == -1  # unrevealed move slots stay hidden
    assert t[0, 8] == 5
    assert (t[1:] == -1).all()


def test_masks_clamp_species_and_restrict_attributes(tables):
    tusk = tables.species_index("greattusk")
    views = [SlotView(species=tusk)] + [SlotView()] * 5
    inp = build_belief_input(views, tables)
    m = inp["masks"]
    # slot 0: species clamped one-hot; attributes restricted by tables
    assert m["species"][0].sum() == 1 and m["species"][0, tusk]
    assert (m["move"][0] == np.asarray(tables.learnset_mask(tusk))).all()
    assert m["ability"][0].sum() == 1  # protosynthesis only
    # spread mask: 2 fixture clusters + <OTHER_SPREAD> = 3 legal
    assert m["spread"][0].sum() == 3
    # slot 1: unknown species -> uninformative
    assert m["species"][1].all() and m["move"][1].all() and m["spread"][1].all()


def test_rare_species_keeps_uninformative_masks(tables):
    rare = tables.species_index(schema.RARE_SPECIES)
    views = [SlotView(species=rare)] + [SlotView()] * 5
    m = masks_for_tokens(tokens_from_views(views), tables)
    assert m["species"][0].sum() == 1  # clamped
    assert m["move"][0].all() and m["item"][0].all()  # but attributes unrestricted


def test_ledger_flow_and_my_view(tables):
    tusk = tables.species_index("greattusk")
    chomp = tables.species_index("garchomp")
    eq = tables.move_index("earthquake")
    sr = tables.move_index("stealthrock")

    ledger = RevealLedger()
    keys = ["tusk", "chomp"] + [f"m{i}" for i in range(4)]
    species = [tusk, chomp, tusk, tusk, tusk, tusk]

    # nothing revealed yet -> only species visible
    t0 = tokens_from_views(ledger.views(keys, species))
    assert (t0[:, 0] >= 0).all() and (t0[:, 1:] == -1).all()

    ledger.record_move("tusk", eq)
    ledger.record_move("tusk", sr)
    ledger.record_move("tusk", eq)  # duplicate use must not consume a second slot
    ledger.record_item("chomp", tables.item_index(schema.NO_ITEM))  # knocked off
    ledger.record_ability("tusk", tables.ability_index("protosynthesis"))
    ledger.record_tera("chomp", 4)

    t1 = tokens_from_views(ledger.views(keys, species))
    assert t1[0, 3] == eq and t1[0, 4] == sr and t1[0, 5] == -1  # reveal order, packed
    assert t1[1, 1] == tables.item_index(schema.NO_ITEM)          # known-absent, NOT -1
    assert t1[0, 2] == tables.ability_index("protosynthesis")
    assert t1[1, 8] == 4
    assert t1[0, 7] == -1  # spreads never directly revealed

    ledger.reset()
    t2 = tokens_from_views(ledger.views(keys, species))
    assert (t2[:, 1:] == -1).all()


def test_labels_and_truth_roundtrip(tables):
    truth = np.arange(54, dtype=np.int64).reshape(6, 9)
    labels = labels_from_truth(truth)
    assert labels["species"].shape == (6,) and labels["move"].shape == (6, 4)
    assert labels["species"][0] == 0 and labels["tera"][5] == 53
    assert (labels["move"][2] == truth[2, 3:7]).all()

    views = [SlotView(species=int(truth[j, 0]) % tables.n_species) for j in range(6)]
    inp = build_belief_input(views, tables, truth_tokens=truth)
    assert "labels" in inp and (inp["labels"]["move"] == truth[:, 3:7]).all()


class _MockMon:
    def __init__(self, species, item="unknown_item", ability=None, moves=(), tera=None):
        self.species = species
        self.item = item
        self.ability = ability
        self.moves = {m: object() for m in moves}
        self.terastallized = tera is not None
        self.tera_type = tera


def test_opponent_views_from_battle(tables):
    class _Battle:
        opponent_team = {
            "p2: Great Tusk": _MockMon(
                "greattusk", item=None, ability="Protosynthesis", moves=("earthquake",)
            ),
            "p2: Garchomp": _MockMon("garchomp"),
        }
        teampreview_opponent_team = [_MockMon("greattusk"), _MockMon("garchomp"),
                                     _MockMon("gabite"), _MockMon("rarebird")]

    views = opponent_views_from_battle(_Battle(), tables)
    assert len(views) == 6
    v_tusk, v_chomp = views[0], views[1]
    assert v_tusk.species == tables.species_index("greattusk")
    assert v_tusk.item == tables.item_index(schema.NO_ITEM)      # item=None -> known-absent
    assert v_tusk.ability == tables.ability_index("protosynthesis")
    assert v_tusk.moves == [tables.move_index("earthquake")]
    assert v_chomp.item is None                                   # unknown_item -> unknown
    # preview padding: gabite known species, rarebird -> <RARE_SPECIES> fallback
    assert views[2].species == tables.species_index("gabite")
    assert views[3].species == tables.species_index(schema.RARE_SPECIES)
    assert views[4].species is None and views[5].species is None
