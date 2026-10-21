import pytest
import torch

from fog.training.optimization import cosine_schedule


def test_cosine_schedule_keeps_group_scales_and_clamps_after_training():
    optimizer = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": 0.01},
            {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": 0.001},
        ]
    )
    schedule = cosine_schedule(optimizer, final_ratio=0.1, total_steps=4)
    rates = [schedule.get_last_lr()]
    for _ in range(6):
        optimizer.step()
        schedule.step()
        rates.append(schedule.get_last_lr())

    assert rates[0] == pytest.approx([0.01, 0.001])
    assert rates[2] == pytest.approx([0.0055, 0.00055])
    assert rates[4] == pytest.approx([0.001, 0.0001])
    assert rates[6] == pytest.approx(rates[4])
    assert all(primary == pytest.approx(10 * shared) for primary, shared in rates)
