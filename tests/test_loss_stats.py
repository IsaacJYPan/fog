import torch

from fog.training.losses import belief_loss


def test_belief_loss_reports_nonfinite_drops_by_family():
    logits = {
        "move": torch.tensor([[[0.0, float("-inf")]]]),
        "item": torch.tensor([[[0.0, 1.0]]]),
    }
    targets = {
        "move": torch.tensor([[1]]),
        "item": torch.tensor([[1]]),
    }
    valid = {
        "move": torch.tensor([[True]]),
        "item": torch.tensor([[True]]),
    }
    stats = {}

    loss = belief_loss(logits, targets, valid, stats=stats)

    assert torch.isfinite(loss)
    assert stats["valid"] == 1
    assert stats["dropped_nonfinite"] == 1
    assert stats["by_family"]["move"]["dropped_nonfinite"] == 1
    assert stats["by_family"]["move"]["dropped_examples"] == [
        {"flat_index": 0, "target": 1}
    ]
    assert stats["by_family"]["item"]["dropped_nonfinite"] == 0
