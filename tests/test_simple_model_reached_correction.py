import numpy as np
import pytest

from fog.encoding.action_features import ACTION_FEATURE_DIM, ACTION_FEATURE_NAMES
from fog.encoding.public_action_features import public_action_features
from fog.training.simple_model_reached_correction import assert_clean_model_reached_rows


def test_model_reached_correction_rejects_answer_derived_inputs():
    clean_cfg = {
        "model": {"use_action_features": False, "simple_rule_prior_strength": 0.0}
    }
    assert_clean_model_reached_rows(clean_cfg, [{"frag": {"entities": object()}}])
    with pytest.raises(ValueError, match="action features"):
        assert_clean_model_reached_rows(
            clean_cfg, [{"frag": {"action_features": object()}}]
        )
    with pytest.raises(ValueError, match="use_action_features"):
        assert_clean_model_reached_rows(
            {"model": {"use_action_features": True}}, []
        )


def test_model_reached_correction_accepts_only_sanitized_public_action_features():
    cfg = {
        "model": {
            "use_action_features": True,
            "public_action_features_only": True,
            "simple_rule_prior_strength": 0.0,
        }
    }
    raw = np.ones((14, ACTION_FEATURE_DIM), dtype=np.float32)
    clean = public_action_features(raw)
    assert_clean_model_reached_rows(cfg, [{"frag": {"action_features": clean}}])

    contaminated = clean.copy()
    contaminated[3, ACTION_FEATURE_NAMES.index("simple_rule_pick")] = 1.0
    with pytest.raises(ValueError, match="answer-derived"):
        assert_clean_model_reached_rows(
            cfg, [{"frag": {"action_features": contaminated}}]
        )
    with pytest.raises(ValueError, match="missing"):
        assert_clean_model_reached_rows(cfg, [{"frag": {"entities": object()}}])
