"""Backbone forward-pass shape test (spec §2.4, §3). Skipped if torch isn't installed.

The backbone is fully implemented, so a random-tensor forward should produce the three explicit
encoder contexts and the policy/value head should mask illegal actions to -inf log-prob.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.encoding.events import H_MY_MOVE
from fog.models.encoder import Backbone
from fog.models.heads import PolicyValueHead


def _random_inputs(b=2, t=7):
    entities = torch.randn(b, 12, ENTITY_RAW_DIM)
    entity_ids = torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long)
    field = torch.randn(b, DIMS.field_dim)
    history = torch.randn(b, t, DIMS.turn_dim)
    active_idx = torch.zeros(b, 2, dtype=torch.long)
    active_idx[:, 1] = 6  # opponent active is slot 6
    return entities, entity_ids, field, history, active_idx


def test_backbone_produces_state_vector():
    bb = Backbone({"state_dim": 256, "dyn_dim": 256})
    contexts = bb(*_random_inputs())
    assert contexts.private_state.shape == (2, 256)
    assert contexts.public_state.shape == (2, 256)
    assert contexts.history.shape == (2, 256)


def test_policy_masks_illegal_actions():
    head = PolicyValueHead(state_dim=256, dyn_dim=128)
    s = torch.randn(2, 256)
    public_context = torch.randn(2, 128)
    mask = torch.zeros(2, 14, dtype=torch.bool)
    mask[:, [0, 1, 8]] = True  # only a few legal
    log_probs, value = head(s, mask, public_context)
    assert log_probs.shape == (2, 14)
    assert torch.isinf(log_probs[~mask]).all()      # illegal -> -inf
    assert torch.isfinite(log_probs[mask]).all()
    assert value.shape == (2,)
    assert (value.abs() <= 1.0).all()               # tanh range


def test_categorical_history_and_private_sheet_have_distinct_model_paths():
    bb = Backbone({"state_dim": 64, "dyn_dim": 64, "dropout": 0.0}).eval()
    entities, entity_ids, field, history, active_idx = _random_inputs(b=1, t=1)
    history_ids = torch.zeros(1, 1, DIMS.hist_id_fields, dtype=torch.long)
    history_ids[..., :2] = bb.embeds.species.num_embeddings - 1
    history_ids[..., 2:8] = bb.embeds.move.num_embeddings - 1
    history_ids[..., 8:10] = bb.embeds.item.num_embeddings - 1
    history_ids[..., 10:12] = bb.embeds.ability.num_embeddings - 1
    history_ids[..., 12:14] = 19
    history_state = torch.zeros(1, 1, DIMS.hist_state)
    lengths = torch.ones(1, dtype=torch.long)

    with torch.no_grad():
        contexts0 = bb(
            entities, entity_ids, field, history, active_idx, lengths,
            history_ids=history_ids, history_state=history_state,
        )
        changed_history = history_ids.clone()
        changed_history[..., H_MY_MOVE] = 1
        contexts1 = bb(
            entities, entity_ids, field, history, active_idx, lengths,
            history_ids=changed_history, history_state=history_state,
        )
        changed_sheet = entity_ids.clone()
        changed_sheet[..., 15] = bb.embeds.item.num_embeddings - 1
        changed_sheet[..., 16] = bb.embeds.ability.num_embeddings - 1
        changed_sheet[:, 0, 15:17] = 1
        contexts2 = bb(
            entities, changed_sheet, field, history, active_idx, lengths,
            history_ids=history_ids, history_state=history_state,
        )
    assert not torch.allclose(contexts0.history, contexts1.history)
    assert not torch.allclose(contexts0.private_state, contexts2.private_state)
