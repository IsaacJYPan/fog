"""Tests for the PPO clipped update (§6) and the leakage penalty machinery (§7)."""
from __future__ import annotations

import math

import torch

from fog.training.leakage import DegenerateSolutionWatchdog, delta_information, shaped_reward
from fog.training.ppo import _masked_entropy, _masked_kl, ppo_update


class TinyPolicyAgent(torch.nn.Module):
    """Minimal stand-in exposing the FogAgent forward contract (log_probs/value/belief=None)."""

    def __init__(self, obs_dim: int = 8, n_actions: int = 14):
        super().__init__()
        self.pi = torch.nn.Linear(obs_dim, n_actions)
        self.v = torch.nn.Linear(obs_dim, 1)

    def forward(self, batch):
        from fog.models.agent import AgentOutput

        logits = self.pi(batch["obs"])
        logits = logits.masked_fill(~batch["legality_mask"], float("-inf"))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = torch.tanh(self.v(batch["obs"])).squeeze(-1)
        return AgentOutput(log_probs=log_probs, value=value)


def make_ppo_batch(n=64, obs_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=g)
    legality = torch.zeros(n, 14, dtype=torch.bool)
    legality[:, :4] = True
    actions = torch.randint(0, 4, (n,), generator=g)
    old_log_probs = torch.log(torch.full((n,), 0.25))
    advantages = torch.randn(n, generator=g)
    returns = torch.tanh(torch.randn(n, generator=g))
    return dict(
        obs=obs,
        legality_mask=legality,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
    )


def test_masked_entropy_and_kl_handle_neg_inf():
    lp = torch.log(torch.tensor([[0.5, 0.5, 0.0, 0.0]]).clamp(min=0))  # -inf on illegal
    ent = _masked_entropy(lp)
    assert torch.isfinite(ent).all()
    assert abs(ent.item() - math.log(2)) < 1e-6
    kl = _masked_kl(lp, lp)
    assert abs(kl.item()) < 1e-6


def test_ppo_update_runs_and_changes_params():
    torch.manual_seed(0)
    agent = TinyPolicyAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    batch = make_ppo_batch()
    before = [p.detach().clone() for p in agent.parameters()]
    metrics = ppo_update(agent, opt, batch, {"ppo": {"minibatch_size": 32}})
    after = list(agent.parameters())
    assert any((a - b).abs().sum() > 0 for a, b in zip(after, before))
    for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        assert key in metrics and math.isfinite(metrics[key])
    assert 0.0 <= metrics["clip_frac"] <= 1.0


def test_ppo_update_improves_advantaged_action():
    """Action 0 gets +1 advantage, actions 1-3 get -1: repeated updates must raise pi(0).

    (Advantages must be contrastive — the update normalizes them over the batch, so a constant
    advantage carries zero policy gradient by construction.)
    """
    torch.manual_seed(1)
    agent = TinyPolicyAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-3)
    batch = make_ppo_batch(n=128, seed=1)
    batch["actions"] = torch.arange(128) % 4
    batch["advantages"] = torch.where(batch["actions"] == 0, 1.0, -1.0)

    def p0():
        with torch.no_grad():
            return agent(batch).log_probs[:, 0].exp().mean().item()

    start = p0()
    for _ in range(10):
        ppo_update(agent, opt, batch, {"ppo": {"minibatch_size": 64, "entropy_coef": 0.0}})
    assert p0() > start + 0.05


def test_kl_anchor_penalizes_drift():
    torch.manual_seed(2)
    agent = TinyPolicyAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    batch = make_ppo_batch()
    with torch.no_grad():
        batch["bc_log_probs"] = agent(batch).log_probs.clone()
    metrics = ppo_update(
        agent, opt, batch, {"ppo": {"minibatch_size": 64, "kl_anchor_coef": 0.02}}
    )
    assert "kl_bc" in metrics and math.isfinite(metrics["kl_bc"])


# --------------------------------------------------------------------------------------- leakage §7

def test_shaped_reward_only_penalizes_revealing():
    di = torch.tensor([1.0, -2.0, 0.0])
    r = shaped_reward(1.0, di, beta=0.1)
    assert torch.allclose(r, torch.tensor([0.9, 1.0, 1.0]))


class ConstCEPhi:
    """Mock Φ̂ (FrozenProbe interface): leakage_ce returns a preset constant per call."""

    def __init__(self, ces):
        self.ces = list(ces)

    def leakage_ce(self, mirror_batch, my_view, **kwargs):
        return torch.tensor([self.ces.pop(0)])


def test_delta_information_sign_convention():
    phi = ConstCEPhi([3.0, 1.0])  # before=3 nats, after=1 nat -> I leaked 2 nats
    obs = {"entities": torch.zeros(1, 4), "my_view": {}}
    di = delta_information(phi, obs, obs)
    assert torch.allclose(di, torch.tensor([2.0]))


def test_watchdogs():
    wd = DegenerateSolutionWatchdog(
        baseline_win_rate=0.5, baseline_move_entropy=1.0, baseline_game_length=40.0
    )
    assert wd.check_move_diversity_spike({"win_rate": 0.3, "move_entropy": 1.5})
    assert not wd.check_move_diversity_spike({"win_rate": 0.3, "move_entropy": 1.0})  # crash alone
    assert not wd.check_move_diversity_spike({"win_rate": 0.55, "move_entropy": 2.0})  # spike alone
    assert wd.check_stale_phi_exploitation(fresh_ce=1.0, frozen_ce=2.0)
    assert not wd.check_stale_phi_exploitation(fresh_ce=2.0, frozen_ce=2.2)
    assert wd.check_stalling(80.0)
    assert not wd.check_stalling(45.0)
    flags = wd.check_all(
        {"win_rate": 0.3, "move_entropy": 1.5, "fresh_ce": 1.0, "frozen_ce": 2.0,
         "avg_game_length": 80.0}
    )
    assert flags == {
        "move_diversity_spike": True,
        "stale_phi_exploitation": True,
        "stalling": True,
    }
