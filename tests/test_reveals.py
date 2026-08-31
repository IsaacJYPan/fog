"""Tests for the information-state core (encoding/reveals.py) against the fixture tables."""
from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest

from fog.constants import schema
from fog.constants.build import build_all
from fog.constants.tables import Tables
from fog.encoding.reveals import (
    RevealLedger,
    SlotView,
    align_training_truth_to_views,
    align_truth_to_views,
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
    assert t.shape == (6, 15)
    assert t[0, 0] == 1 and t[0, 3] == 10 and t[0, 4] == 3
    assert t[0, 5] == -1 and t[0, 6] == -1  # unrevealed move slots stay hidden
    assert t[0, 14] == 5
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
    # slot 1: unknown species -> uninformative
    assert m["species"][1].all() and m["move"][1].all()


def test_rare_species_keeps_uninformative_masks(tables):
    rare = tables.species_index(schema.RARE_SPECIES)
    views = [SlotView(species=rare)] + [SlotView()] * 5
    m = masks_for_tokens(tokens_from_views(views), tables)
    assert m["species"][0].sum() == 1  # clamped
    assert m["move"][0].all() and m["item"][0].all()  # but attributes unrestricted


def test_off_learnset_public_move_relaxes_illusion_sensitive_masks(tables):
    tusk = tables.species_index("greattusk")
    legal_moves = np.asarray(tables.learnset_mask(tusk))
    off_sheet = int(np.flatnonzero(~legal_moves)[0])
    tokens = tokens_from_views([SlotView(species=tusk, moves=[off_sheet])] + [SlotView()] * 5)

    masks = masks_for_tokens(tokens, tables)

    assert masks["species"][0].sum() == 1
    assert masks["move"][0].all()
    assert masks["ability"][0].all()


def test_ogerpon_tera_forms_keep_submitted_and_live_abilities_legal():
    """Belief truth is the submitted set even after Ogerpon changes public battle form."""
    from fog.constants.tables import TABLES

    cases = (
        ("ogerpontealtera", "defiant", "embodyaspectteal"),
        ("ogerponwellspringtera", "waterabsorb", "embodyaspectwellspring"),
        ("ogerponhearthflametera", "moldbreaker", "embodyaspecthearthflame"),
        ("ogerponcornerstonetera", "sturdy", "embodyaspectcornerstone"),
    )
    for public_form, submitted_ability, live_ability in cases:
        mask = TABLES.legal_abilities_mask(TABLES.species_index(public_form))
        assert mask[TABLES.ability_index(submitted_ability)]
        assert mask[TABLES.ability_index(live_ability)]


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
    assert t1[1, 14] == 4
    assert (t1[0, 7:14] == -1).all()  # submitted nature/EVs are never directly revealed

    ledger.reset()
    t2 = tokens_from_views(ledger.views(keys, species))
    assert (t2[:, 1:] == -1).all()


def test_labels_and_truth_roundtrip(tables):
    truth = np.arange(90, dtype=np.int64).reshape(6, 15)
    labels = labels_from_truth(truth)
    assert labels["species"].shape == (6,) and labels["move"].shape == (6, 4)
    assert labels["species"][0] == 0 and labels["tera"][5] == 89
    assert labels["nature"].shape == (6,) and labels["ev"].shape == (6, 6)
    assert (labels["move"][2] == truth[2, 3:7]).all()

    views = [SlotView(species=int(truth[j, 0]) % tables.n_species) for j in range(6)]
    inp = build_belief_input(views, tables, truth_tokens=truth)
    assert "labels" in inp and (inp["labels"]["move"] == truth[:, 3:7]).all()


# -------------------------------------------------------- move-order alignment (fix 2026-07-12)
def test_align_places_revealed_move_at_its_clamp_position():
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0] = [7, 2, 1, 10, 3, 8, 5, 0, 1, 2, 3, 4, 5, 6, 4]
    view = np.full((6, 15), -1, dtype=np.int64)
    view[0, 0] = 7
    view[0, 3] = 8  # revealed: paste position 2 -> clamped at subslot 0 by tokens_from_views
    aligned, ok = align_truth_to_views(truth, view)
    assert ok.all()
    assert aligned[0, 3:7].tolist() == [8, 10, 3, 5]  # reveal first, remainder in paste order
    non_move = [0, 1, 2, *range(7, 15)]
    assert (aligned[0, non_move] == truth[0, non_move]).all()
    assert (aligned[1:] == truth[1:]).all()
    assert truth[0, 3] == 10  # input not mutated


def test_align_keeps_multi_reveal_order_and_is_a_permutation():
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[2, 3:7] = [10, 3, 8, 5]
    view = np.full((6, 15), -1, dtype=np.int64)
    view[2, 3], view[2, 4] = 5, 3  # revealed in THIS order (reveal order, packed earliest)
    aligned, ok = align_truth_to_views(truth, view)
    assert ok.all()
    assert aligned[2, 3:7].tolist() == [5, 3, 10, 8]
    assert sorted(aligned[2, 3:7].tolist()) == sorted(truth[2, 3:7].tolist())


def test_align_flags_contradiction_only_when_provable():
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0, 3:7] = [10, 3, 8, 5]    # fully labeled sheet
    truth[1, 3:7] = [10, 3, -1, -1]  # gap: censored labels / short set — membership unprovable
    view = np.full((6, 15), -1, dtype=np.int64)
    view[0, 3] = 6  # not on slot 0's sheet -> genuine contradiction
    view[1, 3] = 6
    aligned, ok = align_truth_to_views(truth, view)
    assert not ok[0] and ok[1]
    # revealed move is still placed so targets always match the input clamp
    assert aligned[0, 3] == 6
    assert aligned[1, 3:7].tolist() == [6, 10, 3, -1]


def test_training_alignment_preserves_truth_for_illusion_like_move():
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0, 3:7] = [10, 3, 8, 5]
    view = np.full((6, 15), -1, dtype=np.int64)
    view[0, 3] = 6

    aligned, supervision_view, ok = align_training_truth_to_views(truth, view)

    assert not ok[0]
    assert aligned[0, 3:7].tolist() == [10, 3, 8, 5]
    assert supervision_view[0, 3:7].tolist() == [-1, -1, -1, -1]
    assert view[0, 3] == 6


def test_align_broadcasts_one_view_over_candidates():
    cands = np.full((3, 6, 15), -1, dtype=np.int64)
    cands[:, 0, 3:7] = [[10, 3, 8, 5], [10, 3, 8, 6], [3, 8, 10, 5]]
    view = np.full((6, 15), -1, dtype=np.int64)
    view[0, 3] = 5
    aligned, ok = align_truth_to_views(cands, view)
    assert aligned.shape == (3, 6, 15) and ok.shape == (3, 6)
    assert aligned[0, 0, 3:7].tolist() == [5, 10, 3, 8]
    assert not ok[1, 0]  # candidate 1 lacks move 5 on a fully-labeled sheet
    assert aligned[2, 0, 3:7].tolist() == [5, 3, 8, 10]  # remainder keeps ITS paste order


def test_align_without_reveals_is_identity():
    truth = np.arange(90, dtype=np.int64).reshape(6, 15)
    view = np.full((6, 15), -1, dtype=np.int64)
    aligned, ok = align_truth_to_views(truth, view)
    assert (aligned == truth).all() and ok.all()


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
        opponent_team: ClassVar = {
            "p2: Great Tusk": _MockMon(
                "greattusk", item=None, ability="Protosynthesis", moves=("earthquake",)
            ),
            "p2: Garchomp": _MockMon("garchomp"),
        }
        teampreview_opponent_team: ClassVar = [
            _MockMon("greattusk"), _MockMon("garchomp"),
            _MockMon("gabite"), _MockMon("rarebird"),
        ]

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


def test_view_slot_order_follows_preview_matching_paste_order(tables):
    """Slot-order guardrail (KNOWN_ISSUES 2026-07-11): candidate/truth grids are compared
    slot-by-slot against the observation, which is valid because view slots follow PREVIEW order
    — not reveal/insertion order — and Showdown emits |poke| preview in packed-team (paste)
    order. The server half of that claim can't run here; this pins OUR half: a mon revealed
    LATER than another still lands on its preview slot, matching truth_tokens_from_paste rows."""
    from fog.env.team_truth import truth_tokens_from_paste

    paste = (
        "Great Tusk\n"
        "Ability: Protosynthesis\n"
        "- Earthquake\n"
        "- Stealth Rock\n"
        "\n"
        "Garchomp\n"
        "- Earthquake\n"
    )
    truth = truth_tokens_from_paste(paste, tables)

    class _Battle:
        # insertion (reveal) order REVERSED vs preview: garchomp acted first
        opponent_team: ClassVar = {
            "p2: Garchomp": _MockMon("garchomp", moves=("earthquake",)),
            "p2: Great Tusk": _MockMon("greattusk"),
        }
        teampreview_opponent_team: ClassVar = [
            _MockMon("greattusk"), _MockMon("garchomp")
        ]

    tokens = tokens_from_views(opponent_views_from_battle(_Battle(), tables))
    assert tokens[0, 0] == truth[0, 0] == tables.species_index("greattusk")
    assert tokens[1, 0] == truth[1, 0] == tables.species_index("garchomp")
    assert tokens[1, 3] == tables.move_index("earthquake")  # reveal sits on the PREVIEW slot
