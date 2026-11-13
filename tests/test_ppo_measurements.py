"""Learning-check diagnostics must describe the actual rows and update work."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fog.training.ppo import _maybe_chunk_update, ppo_update
from tests.test_ppo_update import TinyPolicyAgent, make_ppo_batch


def test_checkpointed_update_preserves_parameters_metrics_and_rng():
    from copy import deepcopy

    torch.manual_seed(42)
    ordinary = TinyPolicyAgent()
    ordinary.backbone = ordinary.pi  # exercise an actual checkpointed block
    checked = deepcopy(ordinary)
    batch = make_ppo_batch(n=10)
    options = {"epochs": 2, "minibatch_size": 4}
    rng = torch.random.get_rng_state()
    before = ppo_update(ordinary, torch.optim.Adam(ordinary.parameters()), batch,
                        {"ppo": options})
    final_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    after = ppo_update(checked, torch.optim.Adam(checked.parameters()), batch,
                       {"ppo": {**options, "activation_checkpointing": True}})
    assert before == after
    assert torch.equal(final_rng, torch.random.get_rng_state())
    for name, value in ordinary.state_dict().items():
        assert torch.equal(value, checked.state_dict()[name]), name


def test_checkpointed_forward_restores_blocks_on_failure():
    from fog.training.ppo import _checkpointed_forward

    agent = TinyPolicyAgent()
    agent.backbone = agent.pi
    original = agent.backbone.forward
    with pytest.raises(KeyError):
        _checkpointed_forward(agent, {})
    assert agent.backbone.forward == original


def test_update_counts_rows_including_short_last_minibatch():
    agent = TinyPolicyAgent()
    batch = make_ppo_batch(n=10)
    metrics = ppo_update(agent, torch.optim.SGD(agent.parameters(), lr=0), batch,
                         {"ppo": {"epochs": 3, "minibatch_size": 4}})
    assert metrics["optimizer_rows"] == 30
    assert metrics["optimizer_steps"] == 9
    assert metrics["effective_epochs"] == 3


@pytest.mark.parametrize("constant", [False, True])
def test_chunk_reports_behavior_value_variance_and_episode_returns(monkeypatch, constant):
    import time

    import fog.training.ppo as ppo

    returns = torch.tensor([1., 1., 1., 1.] if constant else [-1., 0., 1., 0.])
    values = np.array([0., 0., 1., 1.], dtype=np.float32)
    games = {i: SimpleNamespace(values=values[2*i:2*i+2], outcome=outcome, bonus=0.)
             for i, outcome in enumerate([-1., 1.])}
    batch = {"actions": torch.zeros(4, dtype=torch.long), "returns": returns}
    buf = SimpleNamespace(n_decisions=4, n_games=2, _games=games,
                          to_batch=lambda device: batch, clear=lambda: None)
    dashboard = SimpleNamespace(log=lambda step, metrics: None)
    agent = SimpleNamespace(policy_value=SimpleNamespace(simple_rule_prior_strength=0.))
    setup = SimpleNamespace(
        buf=buf, batch_decisions=4, total=100, decisions_seen=0, cfg={}, ppo_cfg={},
        rollout_tera_logit_bonus=0., entropy0=.01, entropy_decay_to=.003,
        bc_anchor=None, agent=agent, device="cpu", fresh_probe=None, fresh_opt=None,
        leak_cfg={}, opt=None, replay_regularizer=None, rule_prior0=0., rule_prior_final=0.,
        pool_sample_counts=[], chunk_t0=time.monotonic()-.1, chunk_battles=2,
        dash=dashboard, watchdog=None, beta=0., next_snapshot=100,
    )
    monkeypatch.setattr(ppo, "_fit_and_measure_probes", lambda *a, **k: {})
    monkeypatch.setattr(ppo, "_belief_accuracy_vs_turn", lambda *a, **k: {})
    monkeypatch.setattr(ppo, "ppo_update", lambda *a, **k: dict(
        loss=0., entropy=1., approx_kl=0., clip_frac=0., optimizer_rows=12.))
    assert _maybe_chunk_update(setup, SimpleNamespace(phi_hat=None), {}) == "updated"
    metrics = setup.metrics
    assert metrics["mean_episode_return"] == 0.
    expected = 1 - np.var(returns.numpy()-values)/np.var(returns.numpy()) if not constant else None
    if constant:
        assert np.isnan(metrics["value_explained_variance"])
    else:
        assert metrics["value_explained_variance"] == pytest.approx(expected)
    assert metrics["collection_seconds"] >= .1
    assert metrics["update_seconds"] > 0
    assert metrics["optimization_rows_per_sec"] > 0
