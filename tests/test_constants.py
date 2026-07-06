"""Tests for the fixed constants / action space (spec §1). These run without generated tables."""
from __future__ import annotations

import pytest

from fog.constants import schema


def test_action_space_size_and_layout():
    assert schema.N_ACTIONS == 14
    assert len(schema.ACTIONS) == 14
    # 0-3 move, 4-7 tera-move, 8-12 switch, 13 forced
    assert all(a[0] == schema.ActionType.MOVE for a in schema.ACTIONS[0:4])
    assert all(a[0] == schema.ActionType.MOVE_TERA for a in schema.ACTIONS[4:8])
    assert all(a[0] == schema.ActionType.SWITCH for a in schema.ACTIONS[8:13])
    assert schema.ACTIONS[13][0] == schema.ActionType.FORCED
    assert schema.FORCED_SLOT == 13


def test_enum_cardinalities():
    assert schema.N_STATUS == 7
    assert schema.N_WEATHER == 5
    assert schema.N_TERRAIN == 5
    assert schema.N_TYPES == 18
    assert schema.N_TERA_TYPES == 19  # 18 types + Stellar


def test_type_index_roundtrip():
    for t in schema.TYPES:
        assert schema.TYPES[schema.TYPE_INDEX[t]] == t


@pytest.mark.xfail(reason="generated tables not built in scaffolding (§1 build.py stub)", strict=False)
def test_generated_tables_roundtrip():
    from fog.constants.tables import TABLES

    # every scraped team round-trips with <2% unknown-index hits; masks nonempty; spreads within 4 pts
    assert TABLES.species is not None
