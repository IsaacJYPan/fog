"""Phase-2 EV-in-chain ordering tests (BELIEF_POLICY_PLAN §4.2).

Spread (the nature+EV cluster) is generated AFTER item and ability within each slot, so at the spread
step item/ability are pinned and each observed damage event becomes a hard constraint on the spread
alone (the precondition for the §3.4 damage-consistency mask). The constraint touches only the random
training/probe tail order; canonical order already satisfies it. The any-order prefix property is
untouched — this is about the tail order, not the prefix.
"""
from __future__ import annotations

import torch

from fog.models.belief import (
    N_SLOTS,
    _SLOT_ABILITY_FLATS,
    _SLOT_ITEM_FLATS,
    _SLOT_SPREAD_FLATS,
    BeliefModel,
)

SIZES = dict(n_species=10, n_items=6, n_abilities=5, n_moves=12, n_spreads=4, n_tera=19)
_TERA_FLATS = tuple(j * 9 + 8 for j in range(N_SLOTS))


def _model() -> BeliefModel:
    torch.manual_seed(0)
    return BeliefModel(state_dim=32, d=16, **SIZES).eval()


def _content_to_phys(model, revealed, tail_order, seed):
    gen = torch.Generator().manual_seed(seed)
    order = model._order(revealed, tail_order, gen)   # [B,55] physical pos -> content id
    return order.argsort(dim=1)                        # [B,55] content id -> physical pos


def test_spread_after_item_ability_all_hidden_random():
    """Nothing revealed -> all 54 attrs in the tail. Every slot's spread lands after its item and
    ability under the random order, across a big batch."""
    model = _model()
    b = 128
    revealed = torch.zeros(b, 54, dtype=torch.bool)
    phys = _content_to_phys(model, revealed, "random", seed=1)
    for j in range(N_SLOTS):
        assert (phys[:, _SLOT_SPREAD_FLATS[j]] > phys[:, _SLOT_ITEM_FLATS[j]]).all()
        assert (phys[:, _SLOT_SPREAD_FLATS[j]] > phys[:, _SLOT_ABILITY_FLATS[j]]).all()


def test_spread_after_item_ability_canonical():
    """Canonical order already satisfies the constraint (spread col 7 follows item col 1 / ability
    col 2)."""
    model = _model()
    revealed = torch.zeros(4, 54, dtype=torch.bool)
    phys = _content_to_phys(model, revealed, "canonical", seed=0)
    for j in range(N_SLOTS):
        assert (phys[:, _SLOT_SPREAD_FLATS[j]] > phys[:, _SLOT_ITEM_FLATS[j]]).all()
        assert (phys[:, _SLOT_SPREAD_FLATS[j]] > phys[:, _SLOT_ABILITY_FLATS[j]]).all()


def test_revealed_item_ability_still_before_spread():
    """When item/ability are REVEALED they sit in the prefix (physically before START and the whole
    tail), so the spread is still generated after them — the constraint holds trivially."""
    model = _model()
    b = 16
    revealed = torch.zeros(b, 54, dtype=torch.bool)
    for j in range(N_SLOTS):
        revealed[:, _SLOT_ITEM_FLATS[j]] = True
        revealed[:, _SLOT_ABILITY_FLATS[j]] = True
    phys = _content_to_phys(model, revealed, "random", seed=2)
    for j in range(N_SLOTS):
        assert (phys[:, _SLOT_SPREAD_FLATS[j]] > phys[:, _SLOT_ITEM_FLATS[j]]).all()
        assert (phys[:, _SLOT_SPREAD_FLATS[j]] > phys[:, _SLOT_ABILITY_FLATS[j]]).all()


def test_spread_not_globally_forced_last():
    """'Everything else stays free' (§4.2): the constraint pins spread after item/ability of the SAME
    slot only, not after every tail attr. Across a batch, tera frequently follows its slot's spread."""
    model = _model()
    b = 256
    revealed = torch.zeros(b, 54, dtype=torch.bool)
    phys = _content_to_phys(model, revealed, "random", seed=3)
    tera_after_spread_anywhere = False
    for j in range(N_SLOTS):
        if (phys[:, _TERA_FLATS[j]] > phys[:, _SLOT_SPREAD_FLATS[j]]).any():
            tera_after_spread_anywhere = True
    assert tera_after_spread_anywhere, "spread was forced globally last — over-constrained"
