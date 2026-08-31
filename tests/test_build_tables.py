"""Fixture-based tests for constants/build.py + tables.py (§1) — no Showdown checkout needed.

A miniature dex dump (same schema as server/dump-dex.cjs emits) exercises: usage filtering, learnset
gen-9 source filtering + forme/prevo fallback, item usage cuts, sentinel indices, and the tables.py
masks/round-trips.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fog.constants import schema
from fog.constants.build import build_all, to_id, weight_bucket
from fog.constants.tables import Tables

DUMP = {
    "commit": "deadbeef",
    "gen": "gen9",
    "pokedex": {
        "greattusk": {
            "name": "Great Tusk", "num": 984,
            "types": ["Ground", "Fighting"],
            "baseStats": {"hp": 115, "atk": 131, "def": 131, "spa": 53, "spd": 53, "spe": 87},
            "abilities": {"0": "Protosynthesis"},
            "weightkg": 320.0, "tier": "OU", "isNonstandard": None,
            "baseSpecies": None, "changesFrom": None, "prevo": None,
        },
        "garchomp": {
            "name": "Garchomp", "num": 445,
            "types": ["Dragon", "Ground"],
            "baseStats": {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102},
            "abilities": {"0": "Sand Veil", "H": "Rough Skin"},
            "weightkg": 95.0, "tier": "OU", "isNonstandard": None,
            "baseSpecies": None, "changesFrom": None, "prevo": "gabite",
        },
        "gabite": {
            "name": "Gabite", "num": 444,
            "types": ["Dragon", "Ground"],
            "baseStats": {"hp": 68, "atk": 90, "def": 65, "spa": 50, "spd": 55, "spe": 82},
            "abilities": {"0": "Sand Veil", "H": "Rough Skin"},
            "weightkg": 56.0, "tier": "NFE", "isNonstandard": None,
            "baseSpecies": None, "changesFrom": None, "prevo": "gible",
        },
        "rarebird": {  # below usage cut -> excluded (species_index must fall back to <RARE_SPECIES>)
            "name": "Rarebird", "num": 999,
            "types": ["Flying"],
            "baseStats": {"hp": 50, "atk": 50, "def": 50, "spa": 50, "spd": 50, "spe": 50},
            "abilities": {"0": "Keen Eye"},
            "weightkg": 5.0, "tier": None, "isNonstandard": None,
            "baseSpecies": None, "changesFrom": None, "prevo": None,
        },
        "missingmon": {  # isNonstandard -> always excluded
            "name": "Missingmon", "num": 998,
            "types": ["Normal"],
            "baseStats": {"hp": 1, "atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1},
            "abilities": {"0": "Keen Eye"},
            "weightkg": 1.0, "tier": None, "isNonstandard": "CAP",
            "baseSpecies": None, "changesFrom": None, "prevo": None,
        },
    },
    "moves": {
        "earthquake": {
            "name": "Earthquake", "num": 89, "type": "Ground", "category": "Physical",
            "basePower": 100, "accuracy": 100, "priority": 0, "flags": {"protect": 1},
            "status": None, "boosts": None, "target": "allAdjacent", "selfSwitch": False,
            "sideCondition": None, "heal": None, "drain": None, "secondary": None,
            "isNonstandard": None,
        },
        "stealthrock": {
            "name": "Stealth Rock", "num": 446, "type": "Rock", "category": "Status",
            "basePower": 0, "accuracy": True, "priority": 0, "flags": {"reflectable": 1},
            "status": None, "boosts": None, "target": "foeSide", "selfSwitch": False,
            "sideCondition": "stealthrock", "heal": None, "drain": None, "secondary": None,
            "isNonstandard": None,
        },
        "swordsdance": {
            "name": "Swords Dance", "num": 14, "type": "Normal", "category": "Status",
            "basePower": 0, "accuracy": True, "priority": 0, "flags": {"snatch": 1},
            "status": None, "boosts": {"atk": 2}, "target": "self", "selfSwitch": False,
            "sideCondition": None, "heal": None, "drain": None, "secondary": None,
            "isNonstandard": None,
        },
        "sandtomb": {  # only a gabite (prevo) move in this fixture
            "name": "Sand Tomb", "num": 328, "type": "Ground", "category": "Physical",
            "basePower": 35, "accuracy": 85, "priority": 0, "flags": {"protect": 1},
            "status": None, "boosts": None, "target": "normal", "selfSwitch": False,
            "sideCondition": None, "heal": None, "drain": None, "secondary": None,
            "isNonstandard": None,
        },
        "oldgenmove": {  # only has a gen-8 source -> filtered out of every learnset
            "name": "Old Gen Move", "num": 997, "type": "Normal", "category": "Physical",
            "basePower": 70, "accuracy": 100, "priority": 0, "flags": {},
            "status": None, "boosts": None, "target": "normal", "selfSwitch": False,
            "sideCondition": None, "heal": None, "drain": None, "secondary": None,
            "isNonstandard": None,
        },
    },
    "items": {
        "leftovers": {"name": "Leftovers", "num": 234, "isNonstandard": None},
        "choicescarf": {"name": "Choice Scarf", "num": 264, "isNonstandard": None},
        "nichegem": {"name": "Niche Gem", "num": 996, "isNonstandard": None},
    },
    "abilities": {
        "protosynthesis": {"name": "Protosynthesis", "num": 281, "isNonstandard": None},
        "sandveil": {"name": "Sand Veil", "num": 8, "isNonstandard": None},
        "roughskin": {"name": "Rough Skin", "num": 24, "isNonstandard": None},
        "keeneye": {"name": "Keen Eye", "num": 51, "isNonstandard": None},
    },
    "learnsets": {
        "greattusk": {"earthquake": ["9L36"], "stealthrock": ["9M"], "oldgenmove": ["8L20"]},
        "garchomp": {"earthquake": ["9L48"], "swordsdance": ["9M"]},
        "gabite": {"sandtomb": ["9L24"], "earthquake": ["9M"]},
        "rarebird": {"earthquake": ["9M"]},
    },
}

USAGE = {
    "data": {
        "Great Tusk": {
            "usage": 0.40,
            "Items": {"leftovers": 800, "choicescarf": 196, "nichegem": 4},  # nichegem < 0.5%
        },
        "Garchomp": {
            "usage": 0.10,
            "Items": {"choicescarf": 60, "leftovers": 40},
        },
        "Rarebird": {"usage": 0.0001, "Items": {"leftovers": 1}},
    }
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("generated")
    paths = build_all(DUMP, USAGE, out_dir=out_dir)
    # spreads.json isn't produced by build_all; drop a stub so Tables.spreads loads too
    (out_dir / "spreads.json").write_text(json.dumps({"species": {}}))
    return out_dir, paths


@pytest.fixture(scope="module")
def tables(built):
    out_dir, _ = built
    return Tables(gen_dir=out_dir)


def test_helpers():
    assert to_id("Great Tusk") == "greattusk"
    assert to_id("Farfetch'd") == "farfetchd"
    assert weight_bucket(5.0) == 0 and weight_bucket(95.0) == 3 and weight_bucket(320.0) == 5


def test_species_filtering_and_sentinel(built, tables):
    ids = tables.species["ids"]
    assert ids[-1] == schema.RARE_SPECIES
    assert "greattusk" in ids and "garchomp" in ids
    assert "rarebird" not in ids       # below 0.1% usage
    assert "missingmon" not in ids     # nonstandard
    # unknown species falls back to <RARE_SPECIES>
    assert tables.species_index("rarebird") == tables.species_index(schema.RARE_SPECIES)
    assert tables.species_index("greattusk") != tables.species_index("garchomp")


def test_meta_records_commit(built):
    _out_dir, paths = built
    data = json.loads(paths["species"].read_text())
    assert data["_meta"]["showdown_commit"] == "deadbeef"


def test_learnset_gen_filter_and_prevo_fallback(tables):
    tusk = tables.species["features"]["greattusk"]
    assert "earthquake" in tusk["learnset"] and "stealthrock" in tusk["learnset"]
    assert "oldgenmove" not in tusk["learnset"]  # 8-only source
    chomp = tables.species["features"]["garchomp"]
    assert "sandtomb" in chomp["learnset"]       # inherited from gabite (prevo walk)
    # oldgenmove has no gen-9 source anywhere -> not in the MOVES table at all
    assert "oldgenmove" not in tables.moves["ids"]


def test_move_features(tables):
    sr = tables.moves["features"]["stealthrock"]
    assert sr["is_hazard"] and sr["bypass_accuracy"] and sr["category"] == "status"
    sd = tables.moves["features"]["swordsdance"]
    assert sd["is_setup"] and not sd["is_hazard"]
    eq = tables.moves["features"]["earthquake"]
    assert not eq["is_setup"] and eq["base_power"] == 100


def test_item_sentinels_and_usage_cut(tables):
    ids = tables.items["ids"]
    assert ids[-3:] == [schema.OTHER_ITEM, schema.NO_ITEM, schema.UNKNOWN_ITEM]
    assert "leftovers" in ids and "choicescarf" in ids
    assert "nichegem" not in ids  # 0.5% share on its best species
    assert tables.item_index("nichegem") == tables.item_index(schema.OTHER_ITEM)
    assert tables.item_index(schema.NO_ITEM) != tables.item_index(schema.UNKNOWN_ITEM)


def test_masks(tables):
    tusk = tables.species_index("greattusk")
    chomp = tables.species_index("garchomp")
    rare = tables.species_index(schema.RARE_SPECIES)

    ls = tables.learnset_mask(tusk)
    assert ls[tables.move_index("earthquake")] and ls[tables.move_index("stealthrock")]
    assert not ls[tables.move_index("swordsdance")]
    assert tables.learnset_mask(rare).all()  # uninformative prior for rare tail

    ab = tables.legal_abilities_mask(chomp)
    assert ab[tables.ability_index("sandveil")] and ab[tables.ability_index("roughskin")]
    assert not ab[tables.ability_index("protosynthesis")]
    assert ab.sum() == 2

    it = tables.usable_items_mask(tusk)
    assert it[tables.item_index("leftovers")] and it[tables.item_index("choicescarf")]
    # sentinels always legal
    assert it[tables.item_index(schema.NO_ITEM)] and it[tables.item_index(schema.UNKNOWN_ITEM)]
    assert np.asarray(it).dtype == bool


def test_move_index_unknown_raises_or_defaults(tables):
    with pytest.raises(KeyError):
        tables.move_index("notamove")
    assert tables.move_index("notamove", default=-1) == -1


def test_cardinalities_consistent(tables):
    assert tables.n_species == len(tables.species["ids"])
    assert tables._learnset_matrix.shape == (tables.n_species, tables.n_moves)
    assert tables._abilities_matrix.shape == (tables.n_species, tables.n_abilities)
    assert tables._items_matrix.shape == (tables.n_species, tables.n_items)
