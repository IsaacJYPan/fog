import torch

from fog.eval.encoder_information_probe_live import _mode


def test_live_probe_mode_ignores_unknown_and_out_of_range() -> None:
    assert _mode(torch.tensor([-1, 2, 2, 4, 8]), classes=5) == 2
