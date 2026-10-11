import math

import pytest

from fog.pretrained.evaluation import confirmation_gate, continuation_review


@pytest.mark.parametrize(
    "scores", ([1, 0, 1, 0] * 64, [1, 1, 1, 0] * 31 + [1, 0, 0, 0] * 33)
)
def test_uncertain_or_lower_point_estimate_does_not_trigger_rollback(scores):
    result = continuation_review(scores)
    assert result["decision"] == "retain"
    assert not result["significant_regression_on_development"]
    assert not result["strength_claim_confirmed"]
    assert not result["automatic_weight_change"]


def test_clear_regression_recommends_rollback():
    result = continuation_review([0, 0, 1, 0] * 64)
    assert result["decision"] == "rollback"
    assert result["vs_reference"]["high"] < 0.5


def test_exploratory_gain_does_not_bypass_final_confirmation():
    scores = [1, 1, 1, 0] * 256
    result = continuation_review(scores)
    assert result["decision"] == "retain"
    assert result["improvement_supported_on_development"]
    assert not result["strength_claim_confirmed"]
    assert not confirmation_gate(scores, attempt=1)["passed"]


@pytest.mark.parametrize(
    "scores,kwargs",
    [
        ([1, 0, 1, 0] * 63, {}),
        ([1, 0, 1, 0] * 64 + [1], {}),
        ([None, 0, 1, 0] * 64, {}),
        ([1, 0, 1, 0] * 64, {"failures": 1}),
        ([1, 0, 1, 0] * 64, {"unresolved": 1}),
    ],
)
def test_incomplete_or_failed_comparisons_require_diagnosis(scores, kwargs):
    result = continuation_review(scores, **kwargs)
    assert result["decision"] == "diagnose"
    assert result["vs_reference"] is None


@pytest.mark.parametrize("value", (math.nan, math.inf, -0.1, 1.1))
def test_invalid_scores_cannot_support_retention(value):
    with pytest.raises(ValueError, match="finite bounded"):
        continuation_review([value, 0, 1, 0] * 64)


def test_negative_failure_count_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        continuation_review([1, 0, 1, 0] * 64, failures=-1)
