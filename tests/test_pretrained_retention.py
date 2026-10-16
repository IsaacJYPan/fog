import copy

import pytest
import torch
from torch import nn

from fog.pretrained.retention import NativeStep, RetentionLimits, retention_terms


def inputs():
    p = torch.tensor([[0.25, 0.75, 0.0]], requires_grad=True)
    q = torch.tensor([[2.0, -1.0, 99.0]], requires_grad=True)
    return p, q, torch.tensor([[True, True, False]])


def test_retention_is_zero_at_reference_but_belief_gradients_need_not_be():
    p, q, legal = inputs()
    kl, value, metrics = retention_terms(p, p.detach(), q, q.detach(), legal)
    assert RetentionLimits().accepts(metrics)
    assert kl.item() == value.item() == 0
    # A zero retention gradient must not suppress the main belief update.
    (kl + value + (q[..., 0] - 3).square().mean()).backward()
    assert q.grad[0, 0] != 0


def test_illegal_values_do_not_affect_native_value_retention():
    p, q, legal = inputs()
    changed = q.detach().clone()
    changed[~legal] = -99999
    kl, value, metrics = retention_terms(p, p.detach(), q, changed, legal)
    assert value == 0 and kl == 0 and RetentionLimits().accepts(metrics)


def test_retention_rejects_large_changes_and_invalid_masks():
    p, q, legal = inputs()
    _, _, metrics = retention_terms(torch.tensor([[0.75, 0.25, 0.0]]), p.detach(), q, q.detach(), legal)
    assert not RetentionLimits().accepts(metrics)
    with pytest.raises(ValueError, match="Illegal"):
        retention_terms(torch.ones_like(p) / 3, p.detach(), q, q.detach(), legal)
    with pytest.raises(ValueError, match="normalized"):
        retention_terms(p * 2, p.detach(), q, q.detach(), legal)


def test_native_rejection_restores_parameters_and_adam_but_keeps_head_update():
    native, head = nn.Linear(2, 2), nn.Linear(2, 1)
    optimizer = torch.optim.Adam(native.parameters(), lr=0.1)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=0.1)
    x = torch.ones(1, 2)
    for _ in range(2):
        optimizer.zero_grad()
        native(x).square().sum().backward()
        optimizer.step()
    before = copy.deepcopy(native.state_dict())
    adam_before = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad()
    head_optimizer.zero_grad()
    head(native(x)).square().sum().backward()
    step = NativeStep(native.parameters(), optimizer)
    head_before = copy.deepcopy(head.state_dict())
    optimizer.step()
    head_optimizer.step()
    step.restore()
    assert all(torch.equal(v, before[k]) for k, v in native.state_dict().items())
    for key, state in optimizer.state_dict()["state"].items():
        assert all(torch.equal(v, adam_before["state"][key][name]) for name, v in state.items())
    assert any(not torch.equal(v, head_before[k]) for k, v in head.state_dict().items())


def test_limits_reject_invalid_values_and_missing_metrics():
    with pytest.raises(ValueError):
        RetentionLimits(policy_kl_mean=float("nan"))
    assert not RetentionLimits().accepts({})
