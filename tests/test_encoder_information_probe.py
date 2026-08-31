import torch

from fog.encoding.entities import ENTITY_RAW_DIM
from fog.eval.encoder_information_probe import SharedEntityProbe, _known_ce, _majority_correct


def test_known_ce_ignores_unknown_class_ids() -> None:
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    target = torch.tensor([0, 1, 2])
    loss, correct, count = _known_ce(logits, target)
    assert torch.isfinite(loss)
    assert (correct, count) == (2, 2)


def test_shared_entity_probe_accepts_tokens_and_pooled_sources() -> None:
    probe = SharedEntityProbe(
        {"token": 8, "pooled": 16},
        {"species": 7, "item": 5, "ability": 6, "move": 9},
    )
    raw = torch.zeros(2, 6, ENTITY_RAW_DIM)
    ids = torch.zeros(2, 6, 17, dtype=torch.long)
    token_loss, _ = probe.loss_metrics("token", torch.zeros(2, 6, 8), raw, ids)
    pooled_loss, _ = probe.loss_metrics("pooled", torch.zeros(2, 16), raw, ids)
    assert torch.isfinite(token_loss)
    assert torch.isfinite(pooled_loss)


def test_majority_correct_ignores_unknown_values() -> None:
    assert _majority_correct(torch.tensor([0, 1, 1, 3, -1]), classes=3) == 2
