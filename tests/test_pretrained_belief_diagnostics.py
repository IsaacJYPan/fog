import math

import pytest
import torch

from fog.pretrained.belief_diagnostics import DistributionMetrics
from fog.scripts.pretrained_belief_diagnose import validate_diagnostic


def test_distribution_scores_ignore_unverified_values_and_keep_exact_probabilities():
    metric = DistributionMetrics()
    logits = {"item": torch.tensor([[math.log(0.75), math.log(0.25)], [0.0, 0.0]])}
    metric.update(logits, {"item": torch.tensor([0, 99999])}, {"item": torch.tensor([True, False])})
    report = metric.report()["item"]
    assert report["fields"] == 1 and report["top1"] == 1
    assert report["nll"] == pytest.approx(-math.log(0.75))
    assert report["true_probability"] == pytest.approx(0.75)
    assert report["entropy"] == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))


def test_game_aggregation_weights_eligible_fields_instead_of_game_count():
    first, second = DistributionMetrics(), DistributionMetrics()
    first.update({"item": torch.tensor([[0.0, -math.inf]])}, {"item": torch.tensor([0])}, {"item": torch.tensor([True])})
    second.update({"item": torch.zeros(3, 2)}, {"item": torch.ones(3, dtype=torch.long)}, {"item": torch.ones(3, dtype=torch.bool)})
    first.merge(second)
    report = first.report()["item"]
    assert report["fields"] == 4 and report["top1"] == 0.25
    assert report["nll"] == pytest.approx(3 * math.log(2) / 4)
    assert second.report()["item"]["fields"] == 3


@pytest.mark.parametrize("logits,target,error", [
    ([0.0, 0.0], 2, "outside vocabulary"),
    ([0.0, -math.inf], 1, "excluded by decoder constraints"),
])
def test_diagnostic_does_not_hide_impossible_verified_targets(logits, target, error):
    with pytest.raises(ValueError, match=error):
        DistributionMetrics().update({"item": torch.tensor([logits])}, {"item": torch.tensor([target])}, {"item": torch.tensor([True])})


def diagnostic():
    return {
        "seed": 1, "queries_per_context": 16,
        "cohorts": {split: {"folder": split, "games": 1} for split in ("train", "development")},
        "checkpoints": [{"label": "joint-313", "native": "native", "native_sha256": "a" * 64, "auxiliary": "auxiliary", "update": 313}],
    }


@pytest.mark.parametrize("change", [
    {"queries_per_context": 0}, {"seed": None},
    {"cohorts": {"train": {"folder": "train", "games": 1}}},
    {"checkpoints": [{"label": "../escape", "native": "x", "auxiliary": "y", "update": 1, "native_sha256": "a" * 64}]},
])
def test_diagnostic_requires_bound_complete_cohorts_and_checkpoint_identities(change):
    config = diagnostic()
    validate_diagnostic(config)
    config.update(change)
    with pytest.raises(ValueError):
        validate_diagnostic(config)
