"""Backbone forward-pass shape test (spec §2.4, §3). Skipped if torch isn't installed.

The backbone is fully implemented, so a random-tensor forward should produce s in R^256 and the
policy/value head should mask illegal actions to -inf log-prob.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fog.encoding.dims import DIMS  # noqa: E402
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS  # noqa: E402
from fog.models.encoder import Backbone  # noqa: E402
from fog.models.heads import PolicyValueHead  # noqa: E402


def _random_inputs(b=2, t=7):
    entities = torch.randn(b, 12, ENTITY_RAW_DIM)
    entity_ids = torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long)
    field = torch.randn(b, DIMS.field_dim)
    history = torch.randn(b, t, DIMS.turn_dim)
    active_idx = torch.zeros(b, 2, dtype=torch.long)
    active_idx[:, 1] = 6  # opponent active is slot 6
    return entities, entity_ids, field, history, active_idx


def test_backbone_produces_state_vector():
    bb = Backbone({"state_dim": 256})
    s = bb(*_random_inputs())
    assert s.shape == (2, 256)


def test_policy_masks_illegal_actions():
    head = PolicyValueHead(state_dim=256, use_belief_vec=False)
    s = torch.randn(2, 256)
    mask = torch.zeros(2, 14, dtype=torch.bool)
    mask[:, [0, 1, 8]] = True  # only a few legal
    log_probs, value = head(s, mask)
    assert log_probs.shape == (2, 14)
    assert torch.isinf(log_probs[~mask]).all()      # illegal -> -inf
    assert torch.isfinite(log_probs[mask]).all()
    assert value.shape == (2,)
    assert (value.abs() <= 1.0).all()               # tanh range
