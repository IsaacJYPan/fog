import pytest
import torch

from fog.training.curriculum_pilot import ReverseSchedule
from fog.training.losses import value_loss
from fog.training.ppo import ppo_update


def test_curriculum_gate_requires_experience_and_restores_its_sampler_counters():
    schedule = ReverseSchedule()
    for _ in range(16):
        schedule.record(archived_outcome=1, score=1)
        schedule.record(archived_outcome=-1, score=0)
    assert schedule.horizon == 4 and not schedule.advance(9)
    assert schedule.advance(10) and schedule.horizon == 8
    assert not schedule.advance(11)
    schedule.state["reset_episodes"] = 17
    schedule.state["pair_record"] = 3
    restored = ReverseSchedule(schedule.state)
    assert restored.state == schedule.state
    restored.state["winning_scores"].append(1)
    assert not schedule.state["winning_scores"]


def test_failed_finishes_do_not_advance_the_curriculum():
    schedule = ReverseSchedule()
    for i in range(64):
        schedule.record(archived_outcome=1, score=float(i % 2))
    assert not schedule.advance(50) and schedule.horizon == 4


class SharedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.pi = torch.nn.Linear(4, 14)
        self.v = torch.nn.Linear(4, 1)

    def forward(self, batch):
        from fog.models.agent import AgentOutput

        shared = self.backbone(batch["obs"]).tanh()
        log_probs = self.pi(shared).masked_fill(~batch["legality_mask"], -torch.inf).log_softmax(-1)
        return AgentOutput(log_probs=log_probs, value=self.v(shared).tanh().squeeze(-1))


def test_ppo_caps_only_auxiliary_shared_gradient_and_retains_the_full_value_head():
    torch.manual_seed(19)
    agent = SharedPolicy()
    batch = {"obs": torch.randn(8, 3), "actions": torch.arange(8) % 4,
             "legality_mask": torch.tensor([[True]*4+[False]*10]*8),
             "advantages": torch.linspace(-1, 1, 8), "returns": torch.full((8,), -1.)}
    output = agent(batch)
    old = output.log_probs.gather(1, batch["actions"][:, None]).squeeze(1).detach()
    batch["old_log_probs"] = old
    adv = batch["advantages"] / batch["advantages"].std(unbiased=False)
    primary = -((output.log_probs.gather(1, batch["actions"][:, None]).squeeze(1)-old).exp()*adv).mean()
    auxiliary = 100 * value_loss(output.value, batch["returns"])
    shared = list(agent.backbone.parameters())
    p = torch.autograd.grad(primary, shared, retain_graph=True)
    a = torch.autograd.grad(auxiliary, shared, retain_graph=True)
    head = torch.autograd.grad(auxiliary, tuple(agent.v.parameters()), retain_graph=True)
    def norm(values):
        return torch.sqrt(sum(value.square().sum() for value in values))
    scale = min(1., float(.25*norm(p)/norm(a)))
    metrics = ppo_update(agent, torch.optim.SGD(agent.parameters(), lr=0), batch,
                         {"ppo": {"epochs": 1, "minibatch_size": 8, "auxiliary_gradient_cap": .25,
                                  "entropy_coef": 0, "max_grad_norm": 1e6}, "loss": {"value_coef": 100}})
    assert metrics["capped_auxiliary_primary_ratio"] == pytest.approx(.25)
    for parameter, primary_gradient, auxiliary_gradient in zip(shared, p, a):
        assert torch.allclose(parameter.grad, primary_gradient + scale*auxiliary_gradient, atol=2e-5)
    for parameter, expected in zip(agent.v.parameters(), head):
        assert torch.allclose(parameter.grad, expected, atol=2e-5)


@pytest.mark.parametrize("cap", [float("nan"), -.1, .26])
def test_invalid_auxiliary_caps_are_rejected(cap):
    agent = SharedPolicy()
    with pytest.raises(ValueError, match="auxiliary gradient cap"):
        ppo_update(agent, torch.optim.SGD(agent.parameters(), lr=0), {}, {"ppo": {"auxiliary_gradient_cap": cap}})
