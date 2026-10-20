import copy

import pytest
import torch
from torch import nn

from fog.pretrained.gradient_balance import OutcomeGradientCap


def gradients(*, ratio, split=False, zero_belief=False):
    shared = nn.Parameter(torch.eye(2))
    head = nn.Parameter(torch.ones(2))
    balance = OutcomeGradientCap([shared], ratio)
    x = torch.ones(4, 2)
    for part in x.chunk(2 if split else 1):
        latent = part @ shared.T
        factor = len(part) / len(x)
        belief = latent[:, 0].mean() * (0 if zero_belief else 1)
        outcome = 100 * (latent[:, 1].mean() + head.sum())
        retention = 3 * latent.sum(1).mean()
        balance.accumulate(belief * factor, outcome * factor)
        ((belief + outcome + retention) * factor).backward()
    head_before = head.grad.clone()
    metrics = balance.apply()
    return shared.grad, head.grad, head_before, metrics


def test_cap_preserves_head_and_retention_gradients():
    gradient, head, before, metrics = gradients(ratio=0.25)
    # Belief [1,1;0,0], retained outcome [0,0;.25,.25], retention all 3.
    assert torch.allclose(gradient, torch.tensor([[4.0, 4.0], [3.25, 3.25]]))
    assert torch.equal(head, before) and torch.equal(head, torch.full_like(head, 100))
    assert metrics["native_outcome_gradient_norm_after_cap"] == pytest.approx(
        0.25 * metrics["native_belief_gradient_norm"]
    )
    assert metrics["native_retention_gradient_norm"] == pytest.approx(6.0)


def test_cap_uses_accumulated_effective_batch_not_independent_microbatch_scales():
    full = gradients(ratio=0.25)
    micro = gradients(ratio=0.25, split=True)
    assert all(torch.equal(a, b) for a, b in zip(full[:3], micro[:3]))
    assert full[3] == micro[3]


def test_zero_belief_gradient_does_not_redirect_native_training_to_outcome():
    gradient, _, _, metrics = gradients(ratio=0.25, zero_belief=True)
    assert torch.equal(gradient, torch.full_like(gradient, 3))
    assert metrics["native_outcome_gradient_scale"] == 0


def test_invalid_cap_and_frozen_shared_parameters_are_rejected():
    p = nn.Parameter(torch.ones(1))
    for invalid in (-1.0, float("nan"), float("inf"), 2.0):
        with pytest.raises(ValueError):
            OutcomeGradientCap([p], invalid)
    with pytest.raises(ValueError):
        OutcomeGradientCap([torch.ones(1)], 0.25)


def test_gradient_balance_state_does_not_modify_parameters_or_optimizer_state():
    p = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([p], lr=0.01)
    state = copy.deepcopy(optimizer.state_dict())
    balance = OutcomeGradientCap([p], 0.25)
    belief, outcome = p.sum(), p.square().sum() * 50
    balance.accumulate(belief, outcome)
    (belief + outcome).backward()
    balance.apply()
    assert torch.equal(p, torch.ones_like(p))
    assert optimizer.state_dict() == state
