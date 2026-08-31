import math

import torch

from fog.scripts.evaluate_move_set_belief import MoveSetAccumulator


def _empty(classes: int = 10):
    logits = torch.zeros(1, 6, 4, classes)
    targets = torch.full((1, 6, 4), -1, dtype=torch.long)
    valid = torch.zeros(1, 6, 4, dtype=torch.bool)
    return logits, targets, valid


def test_move_set_metric_ignores_storage_permutation() -> None:
    logits, targets, valid = _empty()
    targets[0, 0, :2] = torch.tensor([2, 7])
    valid[0, 0, :2] = True
    logits[0, 0, 0, 7] = 8
    logits[0, 0, 1, 2] = 8
    acc = MoveSetAccumulator()
    acc.update(logits, targets, valid)
    report = acc.report()
    assert report["set_recall_at_cardinality"] == 1.0
    assert report["set_precision_at_cardinality"] == 1.0
    assert report["exact_hidden_set_at_cardinality"] == 1.0
    assert math.isfinite(report["assignment_marginalized_set_nll_per_hidden_move"])
    assert report["assignment_finite_pokemon"] == 1


def test_move_set_metric_penalizes_missing_target() -> None:
    logits, targets, valid = _empty()
    targets[0, 0, :2] = torch.tensor([2, 7])
    valid[0, 0, :2] = True
    logits[0, 0, :, 2] = 8
    logits[0, 0, :, 3] = 7
    acc = MoveSetAccumulator()
    acc.update(logits, targets, valid)
    report = acc.report()
    assert report["set_recall_at_cardinality"] == 0.5
    assert report["exact_hidden_set_at_cardinality"] == 0.0
