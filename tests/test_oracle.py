"""Oracle variants (§3.4): set oracle rewrites belief input to truth; action oracle augments s."""
from __future__ import annotations

import numpy as np
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.models.oracle import ActionOracleHead, apply_set_oracle


def _batch(b=2, extra=None):
    torch.manual_seed(0)
    tokens = torch.full((b, 6, 9), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, 320, (b, 6))
    legal = torch.zeros(b, 14, dtype=torch.bool)
    legal[:, [0, 1, 8]] = True
    batch = {
        "entities": torch.randn(b, 12, ENTITY_RAW_DIM),
        "entity_ids": torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.randn(b, DIMS.field_dim),
        "history": torch.randn(b, 4, DIMS.turn_dim),
        "active_idx": torch.tensor([[0, 6]] * b),
        "legality_mask": legal,
        "opp_revealed": {"tokens": tokens, "masks": None},
    }
    if extra:
        batch.update(extra)
    return batch


def test_apply_set_oracle_clamps_to_truth():
    truth = np.zeros((2, 6, 9), dtype=np.int64)
    truth[:, :, 1] = 3  # item = 3 for all slots
    out = apply_set_oracle({"tokens": torch.full((2, 6, 9), -1)}, truth)
    assert (out["tokens"][:, :, 1] == 3).all()
    assert out["masks"] == {}


def test_set_oracle_uses_truth_tokens():
    agent = FogAgent({"model": {"oracle": "set", "use_belief_vec": True}})
    truth = torch.zeros(2, 6, 9, dtype=torch.long)
    truth[:, :, 0] = 5
    batch = _batch(extra={"opp_truth_tokens": truth})
    out = agent(batch)
    # belief head should now emit truth-consistent species (clamped), loss-free forward runs
    assert out.belief is not None
    assert out.log_probs.shape == (2, 14)


def test_action_oracle_changes_head_width_and_runs():
    agent = FogAgent({"model": {"oracle": "action", "use_belief_vec": True}})
    assert agent.action_oracle is not None
    committed = torch.tensor([2, 7])
    batch = _batch(extra={"opp_committed_action": committed})
    out = agent(batch)
    assert out.log_probs.shape == (2, 14)
    # policy head input width grew by the action-embed dim
    assert agent.policy_value.policy[0].in_features == agent.backbone.state_dim + 128 + agent.action_oracle.extra_dim


def test_action_oracle_head_augment_handles_missing_action():
    head = ActionOracleHead(state_dim=32, belief_dim=16)
    s = torch.randn(3, 32)
    aug = head.augment(s, None)  # no committed action known -> sentinel embed
    assert aug.shape == (3, 32 + head.extra_dim)
    aug2 = head.augment(s, torch.tensor([0, 1, 2]))
    assert aug2.shape == (3, 32 + head.extra_dim)


def test_oracle_none_is_default():
    agent = FogAgent()
    assert agent.oracle == "none" and agent.action_oracle is None
