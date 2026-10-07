import random

import pytest
import torch

from fog.pretrained.data import MatchedMixture, match_split
from fog.pretrained.optimization import AuxiliaryGradientCap


def test_matched_data_schedule_ignores_auxiliary_rng_and_resumes():
    first = MatchedMixture(["p1", "p2"], ["f1", "f2"], 7)
    second = MatchedMixture(["p1", "p2"], ["f1", "f2"], 7)
    rows = []
    for _ in range(5):
        rows.extend(first.next_batch())
        random.random()
    assert sum(source == "published" for source, _, _ in rows) == 36
    assert rows == [row for _ in range(5) for row in second.next_batch()]
    state = first.state_dict()
    expected = first.next_batch()
    second.load_state_dict(state)
    assert second.next_batch() == expected
    with pytest.raises(ValueError, match="different dataset"):
        MatchedMixture(["other"], ["f1"], 7).load_state_dict(state)
    assert match_split("one-match") == match_split("one-match")


def test_cap_after_accumulation_handles_primary_cancellation():
    shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    head = torch.nn.Parameter(torch.tensor(1.0))
    cap = AuxiliaryGradientCap([shared])
    for sign in (1.0, -0.9):
        primary = sign * shared.sum()
        auxiliary = 100 * shared.sum() + head.square()
        cap.backward(primary, auxiliary, scale=0.5)
    metrics = cap.finish()
    assert (
        metrics["capped_auxiliary_backbone_norm"] <= 0.25 * metrics["primary_backbone_norm"] + 1e-7
    )
    torch.testing.assert_close(shared.grad, torch.tensor([0.0625, 0.0625]))
    assert head.grad == 2


def test_zero_primary_does_not_allow_auxiliary_to_update_backbone():
    shared = torch.nn.Parameter(torch.tensor(1.0))
    cap = AuxiliaryGradientCap([shared])
    cap.backward(shared * 0, shared * 100)
    metrics = cap.finish()
    assert shared.grad == 0
    assert metrics["auxiliary_scale"] == 0
