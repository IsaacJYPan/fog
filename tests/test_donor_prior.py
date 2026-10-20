"""Correlated prefix conditioning, open support and decoder probability contracts."""

import numpy as np
import pytest
import torch

from fog.models.belief import BeliefModel
from fog.models.donor_prior import DonorPrior, DonorPriorBatch
from fog.training.replay_candidates import ReplayCandidateBank


def prior_batch(batch=1):
    values = torch.zeros(batch, 6, 2, 15, dtype=torch.long)
    values[:, :, :, 0] = 1
    values[:, :, 0, 1] = 0
    values[:, :, 1, 1] = 1
    values[:, :, 0, 3:7] = torch.tensor([0, 1, 2, 3])
    values[:, :, 1, 3:7] = torch.tensor([4, 5, 6, 7])
    values[:, :, 1, 7] = 1
    return DonorPriorBatch(values, torch.ones(batch, 6, 2) / 2, 0.001)


def public_tokens(batch=1):
    public = torch.full((batch, 6, 15), -1, dtype=torch.long)
    public[:, :, 0] = 1
    return public


def test_item_assumption_selects_a_correlated_moveset_and_nature():
    p = prior_batch()
    fed = public_tokens().flatten(1)
    initial = p.condition(torch.zeros(1, 8), fed, 3).softmax(-1)
    assert torch.allclose(initial, torch.full_like(initial, 1 / 8))
    fed[0, 1] = 1
    after = p.condition(torch.zeros(1, 8), fed, 3).softmax(-1)
    assert after[0, 4:].sum() > 0.999
    assert after[0, :4].sum() > 0  # Open support, not a finite candidate-only decoder.
    nature = p.condition(torch.zeros(1, 3), fed, 7).softmax(-1)
    assert nature[0, 1] > 0.999


def test_move_prefix_is_unordered_and_never_drawn_twice():
    p = prior_batch()
    fed = public_tokens().flatten(1)
    fed[0, 3:5] = torch.tensor([5, 4])
    legal = torch.zeros(1, 8)
    legal[0, 4:6] = -torch.inf
    first = p.condition(legal, fed, 5).softmax(-1)
    fed[0, 3:5] = torch.tensor([4, 5])
    second = p.condition(legal, fed, 5).softmax(-1)
    assert torch.equal(first, second)
    assert first[0, 6:].sum() > 0.999
    assert first[0, 4:6].sum() == 0


def test_contradicted_or_missing_donors_back_off_without_masking_legal_targets():
    p = prior_batch()
    fed = public_tokens().flatten(1)
    fed[0, 1] = 2  # No donor has this item.
    logits = torch.tensor([[0.0, -torch.inf, 0.0]])
    got = p.condition(logits, fed, 7).softmax(-1)
    assert torch.equal(got, torch.tensor([[0.5, 0.0, 0.5]]))
    with pytest.raises(ValueError, match="legal value"):
        p.condition(torch.full((1, 3), -torch.inf), fed, 7)


def small_model():
    torch.manual_seed(14)
    model = BeliefModel(
        state_dim=8,
        d=16,
        n_species=3,
        n_items=5,
        n_abilities=3,
        n_moves=8,
        n_natures=3,
        n_ev_units=2,
        n_tera=3,
    ).eval()
    for head in [*model.heads.values(), *model.ev_stat_heads]:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    return model


def test_sampling_queries_and_scoring_use_the_same_prior():
    model = small_model()
    public = public_tokens()
    # Only item is hidden. This isolates the exact value trace from route/move permutations.
    public[:] = prior_batch().tokens[:, :, 0]
    public[:, 0, 1] = -1
    payload = {"tokens": public, "prior": prior_batch()}
    context = torch.zeros(1, 8)
    marginal = model(context, payload).logits["item"][0, 0].softmax(-1)
    query = model.query_field(context, payload, slot=0, field="item").softmax(-1)[0]
    assert torch.allclose(marginal, query, atol=1e-7)
    sampled = model.sample_team(
        context, payload, k=16, return_trace=True, generator=torch.Generator().manual_seed(21)
    )
    values = sampled.teams[0, :, 0, 1]
    assert torch.allclose(sampled.log_value_prob[0], marginal[values].log(), atol=1e-6)
    for i in range(16):
        score = model.path_logprob(context, payload, sampled.teams[:, i], order_mode="canonical")
        assert torch.allclose(score, sampled.log_value_prob[:, i], atol=1e-6)


def test_prior_training_keeps_target_open_and_backpropagates_residual():
    model = small_model().train()
    public = public_tokens()
    truth = prior_batch().tokens[:, :, 0].clone()
    truth[:, :, 1] = 2  # Outside the bank.
    output = model(
        torch.zeros(1, 8),
        {"tokens": public, "prior": prior_batch()},
        teacher_values=truth,
        generator=torch.Generator().manual_seed(20),
    )
    from fog.pretrained.joint import normalized_belief_loss

    loss = normalized_belief_loss(output, truth, torch.ones_like(truth, dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters())


def fake_bank():
    # Construct the same in-memory interface without file/vocabulary dependencies.
    bank = object.__new__(ReplayCandidateBank)
    bank.tokens = prior_batch().tokens[0, 0].numpy().copy()
    bank.rosters = np.array([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]])
    bank.by_species = {1: np.array([0, 1])}
    return bank


def test_retrieval_excludes_own_public_roster_before_ranking():
    bank = fake_bank()
    retrieval = DonorPrior(bank, max_donors=2)
    public = public_tokens().numpy()
    public[0, :, 0] = np.arange(1, 7)
    p = retrieval.prepare(public)
    assert p.weights[0, 0].sum() == 1
    assert p.tokens[0, 0, 0, 1] == 1
    assert p.weights[0, 0, 1] == 0
    # Retrieval is public-only: its API has no truth/verified-label argument.
    public[0, 0, 1] = 0
    no_donor = retrieval.prepare(public)
    assert no_donor.weights[0, 0].sum() == 0


def test_public_retrieval_cache_preserves_outputs_and_cannot_be_mutated_by_a_caller():
    retrieval = DonorPrior(fake_bank(), max_donors=2)
    public = public_tokens().numpy()
    cold = retrieval.prepare(public)
    expected = cold.tokens.clone(), cold.weights.clone()
    cold.tokens.fill_(0)
    cold.weights.fill_(0)
    warm = retrieval.prepare(public)
    assert torch.equal(warm.tokens, expected[0]) and torch.equal(warm.weights, expected[1])
    assert retrieval.retrieve.cache_info().hits > 0
    retrieval.retrieve.cache_clear()
    resumed = retrieval.prepare(public)
    assert torch.equal(resumed.tokens, warm.tokens) and torch.equal(resumed.weights, warm.weights)


def test_prior_slot_permutation_tracks_the_public_team():
    p = prior_batch(batch=2)
    p.tokens[:, 3, :, 7] = 2
    fed = public_tokens(batch=2)
    order = torch.tensor([3, 1, 2, 0, 4, 5])
    moved = DonorPriorBatch(p.tokens[:, order], p.weights[:, order], p.smoothing)
    before = p.condition(torch.zeros(2, 3), fed.flatten(1), 3 * 15 + 7)
    after = moved.condition(torch.zeros(2, 3), fed[:, order].flatten(1), 7)
    assert torch.equal(before, after)


@pytest.mark.parametrize("seed", [3, 83])
def test_selected_training_queries_match_full_forward_values_and_gradients(seed):
    model = small_model().eval()
    for head in [*model.heads.values(), *model.ev_stat_heads]:
        torch.nn.init.normal_(head.weight, std=0.1)
        torch.nn.init.normal_(head.bias, std=0.1)
    truth = prior_batch(batch=16).tokens[:, :, 0].clone()
    public = public_tokens(batch=16)
    context = torch.randn(16, 8)
    payload = {"tokens": public, "prior": prior_batch(batch=16)}
    from fog.pretrained.joint import normalized_belief_loss

    results, gradients = [], []
    for selected in [False, True]:
        model.zero_grad(set_to_none=True)
        output = model(
            context,
            payload,
            teacher_values=truth,
            selected_only=selected,
            generator=torch.Generator().manual_seed(seed),
        )
        loss = normalized_belief_loss(output, truth, torch.ones_like(truth, dtype=torch.bool))
        loss.backward()
        results.append(loss.detach())
        gradients.append(
            {
                k: v.grad.clone() if v.grad is not None else torch.zeros_like(v)
                for k, v in model.named_parameters()
            }
        )
    assert torch.allclose(results[0], results[1], atol=1e-6)
    assert gradients[0].keys() == gradients[1].keys()
    for key in gradients[0]:
        assert torch.allclose(gradients[0][key], gradients[1][key], atol=1e-6), key
