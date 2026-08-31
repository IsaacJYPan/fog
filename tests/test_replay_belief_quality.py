import torch

from fog.eval.replay_belief_quality import FamilyAccumulator


def test_family_accumulator_reports_accuracy_nll_and_calibration() -> None:
    accumulator = FamilyAccumulator()
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0]])
    targets = torch.tensor([0, 2, 1])
    accumulator.update(logits, targets, torch.tensor([True, True, False]))
    report = accumulator.report()
    assert report["count"] == 2
    assert report["top1"] == 0.5
    assert report["top3"] == 1.0
    assert report["nll"] > 0.0
    assert 0.0 <= report["top1_ece"] <= 1.0


def test_family_accumulator_skips_invalid_and_impossible_targets() -> None:
    accumulator = FamilyAccumulator()
    logits = torch.tensor([[0.0, float("-inf")], [2.0, 0.0]])
    accumulator.update(logits, torch.tensor([1, 0]), torch.tensor([True, False]))
    assert accumulator.report()["count"] == 0
