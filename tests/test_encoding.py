"""Encoding shape/layout tests (spec §2). Shapes are concrete even while extraction is stubbed."""
from __future__ import annotations

import numpy as np

from fog.encoding.dims import DIMS
from fog.encoding.history import pad_or_truncate_front


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
