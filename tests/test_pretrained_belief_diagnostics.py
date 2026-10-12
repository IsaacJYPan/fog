import math

import pytest
import torch

from fog.pretrained.belief_diagnostics import DistributionMetrics, ablated_latent, permuted_context
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


def test_permutation_preserves_pov_and_phase_without_using_other_seat_in_same_game():
    latents = {
        "a": {(0, 0): torch.tensor([[1.0, 2.0]]), (1, 0): torch.tensor([[3.0, 4.0]])},
        "b": {(0, 0): torch.tensor([[5.0, 6.0]]), (1, 0): torch.tensor([[7.0, 8.0]])},
    }
    value, donor = permuted_context(latents, "a", (1, 0))
    assert donor == "b" and torch.equal(value, latents["b"][(1, 0)])
    assert not torch.equal(value, latents["a"][(0, 0)])
    zero, donor = ablated_latent("zero", latents["a"][(0, 0)])
    assert donor is None and not zero.any()
    assert torch.equal(latents["a"][(0, 0)], torch.tensor([[1.0, 2.0]]))


def test_permutation_rejects_incomplete_contexts_instead_of_borrowing_private_opponent_view():
    with pytest.raises(ValueError, match="two complete"):
        permuted_context({"a": {(0, 0): torch.ones(1, 2)}}, "a", (0, 0))
    with pytest.raises(ValueError, match="matching POV/phase"):
        permuted_context({"a": {(0, 0): torch.ones(1, 2)}, "b": {(1, 0): torch.ones(1, 2)}}, "a", (0, 0))
    with pytest.raises(ValueError, match="shape differs"):
        ablated_latent("permuted", torch.ones(1, 2), latents={"a": {(0, 0): torch.ones(1, 2)}, "b": {(0, 0): torch.ones(1, 3)}}, game="a", point=(0, 0))


@pytest.mark.parametrize("modes", [["unknown"], ["zero", "zero"], "zero", ["permuted"]])
def test_invalid_or_unpairable_ablations_fail_closed(modes):
    config = diagnostic()
    config["latent_ablations"] = modes
    with pytest.raises(ValueError):
        validate_diagnostic(config)


def test_diagnostic_allows_complete_offline_ablations():
    config = diagnostic()
    config["cohorts"] = {split: {"folder": split, "games": 2} for split in ("train", "development")}
    config["latent_ablations"] = ["zero", "permuted"]
    validate_diagnostic(config)
