import numpy as np
import pytest
import torch

from fog.encoding.action_features import ACTION_FEATURE_DIM, ACTION_FEATURE_NAMES
from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.encoding.public_action_features import (
    ANSWER_DERIVED_ACTION_FEATURE_NAMES,
    PUBLIC_ACTION_FEATURE_NAMES,
    public_action_features,
)
from fog.models.agent import FogAgent


def test_public_action_features_erase_every_rule_decision_column():
    raw = np.arange(2 * 14 * ACTION_FEATURE_DIM, dtype=np.float32).reshape(
        2, 14, ACTION_FEATURE_DIM
    )
    clean = public_action_features(raw)

    for name in PUBLIC_ACTION_FEATURE_NAMES:
        index = ACTION_FEATURE_NAMES.index(name)
        assert np.array_equal(clean[..., index], raw[..., index])
    for name in ANSWER_DERIVED_ACTION_FEATURE_NAMES:
        index = ACTION_FEATURE_NAMES.index(name)
        assert not clean[..., index].any(), name


def test_public_action_features_are_invariant_to_exact_answer_and_rule_branches():
    rng = np.random.default_rng(7)
    first = rng.normal(size=(14, ACTION_FEATURE_DIM)).astype(np.float32)
    second = first.copy()
    for name in ANSWER_DERIVED_ACTION_FEATURE_NAMES:
        second[:, ACTION_FEATURE_NAMES.index(name)] = rng.normal(size=14)

    assert np.array_equal(public_action_features(first), public_action_features(second))
    assert "simple_rule_pick" in ANSWER_DERIVED_ACTION_FEATURE_NAMES
    assert "move_branch_active" in ANSWER_DERIVED_ACTION_FEATURE_NAMES
    assert "switch_branch_active" in ANSWER_DERIVED_ACTION_FEATURE_NAMES


def test_public_action_features_reject_bad_shape_and_nonfinite_values():
    with pytest.raises(ValueError, match="shape"):
        public_action_features(np.zeros((13, ACTION_FEATURE_DIM), dtype=np.float32))
    bad = np.zeros((14, ACTION_FEATURE_DIM), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        public_action_features(bad)


def test_agent_masks_answer_columns_before_the_existing_action_scorer():
    agent = FogAgent(
        {
            "encoding": {"dropout": 0.0},
            "model": {
                "use_action_features": True,
                "public_action_features_only": True,
                "simple_rule_prior_strength": 0.0,
            },
        }
    ).eval()
    scorer = agent.policy_value.action_scorer
    assert scorer is not None
    for parameter in scorer.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)

    batch_size = 2
    base = {
        "entities": torch.zeros(batch_size, 12, ENTITY_RAW_DIM),
        "entity_ids": torch.zeros(batch_size, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.zeros(batch_size, DIMS.field_dim),
        "history": torch.zeros(batch_size, 1, DIMS.turn_dim),
        "active_idx": torch.tensor([[0, 6], [0, 6]], dtype=torch.long),
        "legality_mask": torch.ones(batch_size, 14, dtype=torch.bool),
    }
    first = torch.randn(batch_size, 14, ACTION_FEATURE_DIM)
    second = first.clone()
    for name in ANSWER_DERIVED_ACTION_FEATURE_NAMES:
        second[..., ACTION_FEATURE_NAMES.index(name)] = torch.randn(batch_size, 14)

    with torch.no_grad():
        output1 = agent({**base, "action_features": first})
        output2 = agent({**base, "action_features": second})
    assert torch.equal(output1.log_probs, output2.log_probs)
    assert torch.equal(output1.value, output2.value)


def test_public_only_mode_rejects_rule_prior_and_missing_action_channel():
    with pytest.raises(ValueError, match="use_action_features"):
        FogAgent({"model": {"public_action_features_only": True}})
    with pytest.raises(ValueError, match="simple_rule_prior_strength"):
        FogAgent(
            {
                "model": {
                    "use_action_features": True,
                    "public_action_features_only": True,
                    "simple_rule_prior_strength": 1.0,
                }
            }
        )
