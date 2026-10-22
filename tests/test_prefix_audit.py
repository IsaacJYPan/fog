"""The offline observer must not alter inference or confuse partial set support."""

import torch

from fog.models.donor_prior import DonorPriorBatch
from fog.pretrained.prefix_audit import (
    ObservedPriorBatch, PrefixObservations, compatible_sets, summarize,
)


def fixture():
    donors = torch.zeros(1, 6, 2, 15, dtype=torch.long)
    donors[..., 0] = 1
    donors[:, :, 0, 3:7] = torch.tensor([0, 1, 2, 3])
    donors[:, :, 1, 3:7] = torch.tensor([4, 5, 6, 7])
    donors[:, :, 1, 1] = 1
    truth = donors[:, :, 0].clone()
    public = torch.full_like(truth, -1)
    public[..., 0] = 1
    prior = DonorPriorBatch(donors, torch.ones(1, 6, 2) / 2, 0.001)
    obs = PrefixObservations(truth, torch.ones_like(truth, dtype=torch.bool), public,
                             [{"game": "a", "phase": "early"}], source="test")
    return prior, public.flatten(1), obs


def test_observer_preserves_logits_rng_and_gradients_with_private_truth_changed():
    prior, fed, obs = fixture()
    logits = torch.randn(1, 3, requires_grad=True)
    rng = torch.get_rng_state().clone()
    expected = prior.condition(logits, fed, 1)
    wrapped = ObservedPriorBatch(prior, obs)
    actual = wrapped.condition(logits, fed, 1)
    assert torch.equal(expected, actual) and torch.equal(rng, torch.get_rng_state())
    obs.truth.fill_(2)
    assert torch.equal(wrapped.condition(logits, fed, 1), expected)
    a = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(actual.sum(), logits)[0]
    assert torch.equal(a, b)


def test_fallback_and_contradictory_prefix_are_different_events():
    prior, fed, obs = fixture()
    wrapped = ObservedPriorBatch(prior, obs)
    wrapped.condition(torch.zeros(1, 3), fed, 7)
    fed[0, 1] = 1  # Wrong ground-truth item, but still a well-supported donor.
    wrapped.condition(torch.zeros(1, 3), fed, 7)
    fed[0, 1] = 2  # Outside the donor bank, so fallback as well.
    wrapped.condition(torch.zeros(1, 3), fed, 7)
    assert [x["fallback"] for x in obs.rows] == [False, False, True]
    assert [x["truth_prefix_compatible"] for x in obs.rows] == [True, False, False]
    assert summarize(obs.rows)["all"]["fallback_rate"] == 1 / 3


def test_unordered_move_comparison_and_remaining_target_mass():
    prior, fed, obs = fixture()
    fed[0, 3:5] = torch.tensor([2, 1])
    logits = torch.zeros(1, 8)
    logits[:, 1:3] = -torch.inf
    ObservedPriorBatch(prior, obs).condition(logits, fed, 5)
    row = obs.rows[0]
    assert row["viable_donors"] == 1 and row["truth_prefix_compatible"]
    assert row["prior_true_mass"] > 0.999 and row["truth_value_supported"]
    permuted = obs.truth[0, 0].clone()
    permuted[3:7] = permuted[3:7].flip(0)
    assert bool(compatible_sets(prior.tokens[0, 0], permuted)[0])


def test_adapter_repeated_and_selected_rows_keep_original_identity():
    prior, fed, obs = fixture()
    wrapped = ObservedPriorBatch(prior, obs).repeat_interleave(4)
    selected = torch.tensor([False, True, False, True])
    wrapped.index_select(selected).condition(torch.zeros(2, 3), fed.repeat(2, 1), 7)
    assert len(obs.rows) == 2 and all(x["game"] == "a" for x in obs.rows)


def test_observer_skips_known_target_and_does_not_treat_unverified_set_as_supported():
    prior, fed, obs = fixture()
    obs.verified[0, 0, 7] = False
    wrapped = ObservedPriorBatch(prior, obs)
    wrapped.condition(torch.zeros(1, 3), fed, 0)
    wrapped.condition(torch.zeros(1, 3), fed, 7)
    assert not obs.rows
    wrapped.condition(torch.zeros(1, 3), fed, 1)
    assert not obs.rows[0]["truth_set_in_retrieved_bank"]
