"""Dependency tests for the value-independent set-AR generation scheduler."""
from __future__ import annotations

import torch

from fog.models.belief import (
    N_SLOTS,
    _SLOT_ABILITY_FLATS,
    _SLOT_ITEM_FLATS,
    _SLOT_MOVE_FLATS,
    _SLOT_SPREAD_FLATS,
    BeliefModel,
)

SIZES = dict(n_species=10, n_items=6, n_abilities=5, n_moves=12, n_spreads=4, n_tera=19)


def model() -> BeliefModel:
    torch.manual_seed(0)
    return BeliefModel(state_dim=32, d=16, **SIZES).eval()


def positions(route: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(route, -1)
    rows = torch.arange(len(route)).unsqueeze(1).expand_as(route)
    steps = torch.arange(route.shape[1]).unsqueeze(0).expand_as(route)
    valid = route >= 0
    result[rows[valid], route[valid]] = steps[valid]
    return result


def test_random_route_obeys_only_mechanical_dependencies():
    belief = model()
    revealed = torch.zeros(128, 54, dtype=torch.bool)
    route, length, _ = belief._route(
        revealed, "random", torch.Generator().manual_seed(1)
    )
    where = positions(route)
    assert (length == 54).all()
    for slot in range(N_SLOTS):
        species = slot * 9
        assert (where[:, _SLOT_ABILITY_FLATS[slot]] > where[:, species]).all()
        for move in _SLOT_MOVE_FLATS[slot]:
            assert (where[:, move] > where[:, species]).all()
        spread = where[:, _SLOT_SPREAD_FLATS[slot]]
        assert (spread > where[:, species]).all()
        assert (spread > where[:, _SLOT_ITEM_FLATS[slot]]).all()
        assert (spread > where[:, _SLOT_ABILITY_FLATS[slot]]).all()


def test_scheduler_is_not_canonical_or_globally_constrained():
    belief = model()
    revealed = torch.zeros(256, 54, dtype=torch.bool)
    route, _, _ = belief._route(
        revealed, "random", torch.Generator().manual_seed(2)
    )
    where = positions(route)
    # Item and Tera have no mutual dependency; both directions must occur in a real random route.
    item_before_tera = []
    for slot in range(N_SLOTS):
        item = where[:, slot * 9 + 1]
        tera = where[:, slot * 9 + 8]
        item_before_tera.append(item < tera)
    both = torch.stack(item_before_tera)
    assert both.any() and (~both).any()

    # Spread is not globally last: unrelated fields frequently follow it.
    assert any(
        (where[:, slot * 9 + 8] > where[:, _SLOT_SPREAD_FLATS[slot]]).any()
        for slot in range(N_SLOTS)
    )


def test_revealed_dependencies_are_available_from_step_zero():
    belief = model()
    revealed = torch.zeros(32, 54, dtype=torch.bool)
    for slot in range(N_SLOTS):
        revealed[:, slot * 9] = True
        revealed[:, _SLOT_ITEM_FLATS[slot]] = True
        revealed[:, _SLOT_ABILITY_FLATS[slot]] = True
    eligible = belief._eligible(revealed)
    for slot in range(N_SLOTS):
        assert eligible[:, _SLOT_MOVE_FLATS[slot][0]].all()
        assert eligible[:, _SLOT_SPREAD_FLATS[slot]].all()


def test_canonical_diagnostic_still_respects_dependencies():
    belief = model()
    revealed = torch.zeros(4, 54, dtype=torch.bool)
    route, _, _ = belief._route(revealed, "canonical", None)
    where = positions(route)
    for slot in range(N_SLOTS):
        assert (
            where[:, _SLOT_SPREAD_FLATS[slot]]
            > where[:, _SLOT_ITEM_FLATS[slot]]
        ).all()
        assert (
            where[:, _SLOT_SPREAD_FLATS[slot]]
            > where[:, _SLOT_ABILITY_FLATS[slot]]
        ).all()
