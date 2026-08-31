"""Encoding shape/layout tests (spec §2): the widths every encoder and test must agree on."""
from __future__ import annotations

import numpy as np
import pytest

from fog.encoding.dims import DIMS
from fog.encoding.history import pad_or_truncate_front


def test_legality_encoding_does_not_hide_action_mapping_failures(monkeypatch):
    from fog.encoding import entities

    def fail(_battle):
        raise RuntimeError("broken request parser")

    monkeypatch.setattr(entities, "action_legality_mask", fail)
    with pytest.raises(RuntimeError, match="broken request parser"):
        entities.legality_mask(object())


def test_field_dim_matches_layout():
    assert DIMS.field_dim > 0
    # sum of the documented field blocks
    expected = (
        DIMS.field_weather
        + DIMS.field_terrain
        + DIMS.field_hazards
        + DIMS.field_screens
        + DIMS.field_trickroom
        + DIMS.field_turn
        + DIMS.field_forced_switch
        + DIMS.field_delayed      # Future Sight / Wish per side (encoding audit 2026-07)
        + DIMS.field_gravity
    )
    assert DIMS.field_dim == expected
    # delayed-effects block: per-side (FS flag, FS turns, Wish flag, Wish turns, Wish amount)
    assert DIMS.field_delayed == 10


def test_turn_dim_matches_layout():
    expected = DIMS.hist_snapshot + DIMS.hist_my_action + DIMS.hist_opp_action + DIMS.hist_resolution
    assert DIMS.turn_dim == expected
    assert DIMS.hist_my_action == 14  # 14-way one-hot action


def test_history_front_pad():
    d = DIMS.turn_dim
    short = np.ones((3, d), dtype=np.float32)
    out = pad_or_truncate_front(short, cap=DIMS.history_cap)
    assert out.shape == (DIMS.history_cap, d)
    # most recent turns preserved at the end; front zero-padded
    assert out[-1].sum() == d
    assert out[0].sum() == 0


def test_history_truncate_keeps_recent():
    d = DIMS.turn_dim
    long = np.arange(DIMS.history_cap + 10)[:, None] * np.ones((1, d), dtype=np.float32)
    out = pad_or_truncate_front(long, cap=DIMS.history_cap)
    assert out.shape == (DIMS.history_cap, d)
    # last row is the most recent (largest index), earliest 10 dropped
    assert out[-1, 0] == DIMS.history_cap + 9
    assert out[0, 0] == 10


def test_old_replay_entity_ids_gain_unknown_private_sheet_columns():
    from fog.constants.schema import N_EV_UNITS, N_NATURES
    from fog.constants.tables import TABLES
    from fog.training.dataset import _normalize_entity_ids

    old = np.zeros((2, 12, 9), dtype=np.int64)
    upgraded = _normalize_entity_ids(old)
    assert upgraded.shape == (2, 12, 17)
    assert (upgraded[..., 8] == N_NATURES).all()
    assert (upgraded[..., 9:15] == N_EV_UNITS).all()
    assert (upgraded[..., 15] == TABLES.n_items).all()
    assert (upgraded[..., 16] == TABLES.n_abilities).all()


def test_public_replay_does_not_fabricate_acting_players_private_sheet():
    from fog.constants.schema import N_EV_UNITS, N_NATURES
    from fog.constants.tables import TABLES
    from fog.encoding.entities import L
    from fog.encoding.observation import Observation
    from tests.golden_fixture import GoldenBattle

    battle = GoldenBattle()
    own_public = np.full((6, 15), -1, dtype=np.int64)
    own_public[:, 0] = [
        TABLES.species_index(mon.species) for mon in battle.team.values()
    ]
    observation = Observation.from_battle(battle, own_revealed=own_public)

    assert (observation.entity_ids[:6, 1] == TABLES.n_items).all()
    assert (observation.entity_ids[:6, 2] == TABLES.n_abilities).all()
    assert (observation.entity_ids[:6, 3:7] == TABLES.n_moves).all()
    assert (observation.entity_ids[:6, 8] == N_NATURES).all()
    assert (observation.entity_ids[:6, 9:15] == N_EV_UNITS).all()
    assert (
        observation.entities[:6, L.epistemic : L.epistemic + DIMS.epistemic_flags] == 0
    ).all()
