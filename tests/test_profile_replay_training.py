from __future__ import annotations

import pytest
import torch

from fog.scripts.profile_replay_training import measure_batch_building, measure_steps


class Dataset:
    def __len__(self):
        return 6

    def batches(self, batch_size, *, seed, **_kwargs):
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(6, generator=generator)
        for start in range(0, 6, batch_size):
            indices = order[start:start + batch_size]
            yield {"x": indices.float()[:, None] / 6, "action": indices % 2}


class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.belief = torch.nn.Linear(1, 2)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, batch):
        return self.dropout(self.belief(batch["x"]))


def loss(agent, batch, _config):
    return torch.nn.functional.cross_entropy(agent(batch), batch["action"]), {}


def test_profile_hooks_preserve_twenty_updates_and_report_components():
    cfg = {"seed": 123, "bc": {"batch_size": 3, "epochs": 10}}
    torch.manual_seed(42)
    agent = Agent()
    actual = measure_steps(agent, Dataset(), cfg, warmup=2, steps=18, loss_fn=loss)
    # Independently execute the ordinary trainer update, including stochastic dropout.
    torch.manual_seed(42)
    reference = Agent().train()
    opt = torch.optim.AdamW(reference.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=3e-5)
    expected = []
    for epoch in range(10):
        for batch in Dataset().batches(3, seed=123 + epoch):
            opt.zero_grad(set_to_none=True)
            total, _ = loss(reference, batch, {})
            expected.append(float(total.detach()))
            total.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            opt.step()
            sched.step()
    assert actual["first_20_training_losses"] == expected
    for name, tensor in agent.state_dict().items():
        assert torch.equal(tensor, reference.state_dict()[name])
    seconds = actual["seconds_per_step"]
    assert sum(seconds[key] for key in (
        "batch_build", "model_forward", "backward_optimizer", "other"
    )) == pytest.approx(seconds["total"])
    assert 0 < actual["belief_share_of_model_forward"] <= 1
    assert actual["measured_rows"] == 54
    assert not agent._forward_hooks
    assert not agent.belief._forward_hooks


def test_batch_only_pass_includes_partial_batches_and_epoch_restarts():
    result = measure_batch_building(Dataset(), batch_size=4, seed=3, steps=5)
    assert result["rows"] == 16
    assert result["rows_per_second"] > 0


@pytest.mark.parametrize("warmup,steps", [(-1, 2), (0, 0)])
def test_invalid_profile_lengths(warmup, steps):
    with pytest.raises(ValueError):
        measure_steps(Agent(), Dataset(), {}, warmup=warmup, steps=steps)
