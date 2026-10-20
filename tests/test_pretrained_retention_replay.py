import copy
import random

import pytest
import torch
from torch import nn

from fog.pretrained.retention import RetentionLimits
from fog.pretrained.retention_replay import (
    OriginalReferenceRepair,
    budget_retention_terms,
    replay_specifications,
)


def measured(value):
    return {
        "policy_kl_mean": abs(value) * 0.002,
        "policy_kl_max": abs(value) * 0.002,
        "argmax_disagreement": 0.0,
        "value_relative_mean": 0.0,
        "value_relative_max": 0.0,
    }


def test_preexisting_violation_contracts_from_original_and_resets_only_native_Adam():
    native, head = nn.Parameter(torch.zeros(())), nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([native], lr=0.1)
    repair = OriginalReferenceRepair([native])
    (native - 1).square().backward()
    optimizer.step()
    with torch.no_grad():
        native.fill_(2.0)
    head_before = head.clone()
    result = repair.ensure(lambda: {"training": measured(float(native))}, optimizer)
    assert result["repaired"] and result["original_reference_scale"] == 0.25
    assert native.item() == 0.5 and not optimizer.state and torch.equal(head, head_before)
    assert repair.repairs == 1 and native.grad is None
    again = repair.ensure(lambda: {"training": measured(float(native))}, optimizer)
    assert not again["repaired"] and repair.repairs == 1


def test_invalid_original_fails_closed_and_restores_candidate_and_optimizer():
    native = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([native], lr=0.1)
    repair = OriginalReferenceRepair([native], max_backtracks=2)
    (native - 1).square().backward()
    optimizer.step()
    before = native.clone()
    state = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(FloatingPointError, match="Protected original"):
        repair.ensure(lambda: {"broken_contract": measured(1.0)}, optimizer)
    assert torch.equal(native, before) and repair.repairs == 0
    assert optimizer.state_dict()["param_groups"] == state["param_groups"]
    assert all(
        torch.equal(value, state["state"][0][key])
        for key, value in optimizer.state_dict()["state"][0].items()
    )


def test_repair_resume_requires_same_original_and_preserves_progress():
    repair = OriginalReferenceRepair([nn.Parameter(torch.zeros(2))])
    repair.repairs = 3
    restored = OriginalReferenceRepair([nn.Parameter(torch.zeros(2))])
    restored.load_state_dict(repair.state_dict())
    assert restored.state_dict() == repair.state_dict()
    with pytest.raises(ValueError, match="reference"):
        OriginalReferenceRepair([nn.Parameter(torch.ones(2))]).load_state_dict(repair.state_dict())


def test_retention_dead_zone_allows_small_motion_and_penalizes_excess():
    base = torch.tensor([[0.5, 0.5]])
    p = torch.tensor([[0.5001, 0.4999]], requires_grad=True)
    q = torch.tensor([[0.0001, -0.0001]], requires_grad=True)
    legal = torch.ones_like(p, dtype=torch.bool)
    policy, value = budget_retention_terms(
        p, base, q, torch.zeros_like(q), legal, limits=RetentionLimits(), fraction=0.25
    )
    (policy + value).backward()
    assert policy == 0 and value == 0 and p.grad.abs().max() == 0 and q.grad.abs().max() == 0
    p = torch.tensor([[0.9, 0.1]], requires_grad=True)
    q = torch.ones_like(p, requires_grad=True)
    policy, value = budget_retention_terms(
        p, base, q, torch.zeros_like(q), legal, limits=RetentionLimits(), fraction=0.25
    )
    (policy + value).backward()
    assert policy > 0 and value > 0 and p.grad.abs().sum() > 0 and q.grad.abs().sum() > 0


def test_retention_objective_has_effective_context_weighting_with_variable_legal_actions():
    p = torch.tensor([[0.7, 0.3], [1.0, 0.0]])
    base = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    legal = torch.tensor([[True, True], [True, False]])
    q = torch.tensor([[0.02, 0.05], [0.08, 100.0]])
    whole = budget_retention_terms(
        p, base, q, torch.zeros_like(q), legal, limits=RetentionLimits(), fraction=0.25
    )
    parts = [
        budget_retention_terms(
            p[i : i + 1],
            base[i : i + 1],
            q[i : i + 1],
            torch.zeros_like(q[i : i + 1]),
            legal[i : i + 1],
            limits=RetentionLimits(),
            fraction=0.25,
        )
        for i in range(2)
    ]
    assert all(
        torch.allclose(full, (parts[0][j] + parts[1][j]) / 2) for j, full in enumerate(whole)
    )


class Data:
    rows = {
        "train": [{"path": "train/a.pt", "positions": 20}],
        "development": [{"path": "development/b.pt", "positions": 20}],
    }

    @staticmethod
    def draw(rng, count):
        return [("train/a.pt", rng.randrange(20)) for _ in range(count)]


def test_bank_is_training_only_unique_and_does_not_consume_shared_random_streams():
    before = random.getstate()
    specs, signature = replay_specifications(
        Data(), count=8, seed=12, extra=[("train/a.pt", 3)] * 2
    )
    assert random.getstate() == before and len(specs) == len(set(specs)) == 8
    assert specs[0] == ("train/a.pt", 3)
    assert (specs, signature) == replay_specifications(
        Data(), count=8, seed=12, extra=[("train/a.pt", 3)]
    )
    with pytest.raises(ValueError, match="training contexts"):
        replay_specifications(Data(), count=8, seed=12, extra=[("development/b.pt", 3)])
