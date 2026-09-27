from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fog.training.training_state import restore_training_state, save_training_state, state_path_for


class _ReplayState:
    def __init__(self) -> None:
        self.value = {"active_epoch": 3, "batch_in_epoch": 7, "remaining": 0}

    def state_dict(self):
        return dict(self.value)

    def load_state_dict(self, state):
        self.value = dict(state)


def _setup(tmp_path: Path):
    agent = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.01)
    loss = agent(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return SimpleNamespace(
        cfg={"seed": 9, "ppo": {"total_decisions": 100}, "log": {"dir": str(tmp_path)}},
        device=torch.device("cpu"),
        agent=agent,
        opt=optimizer,
        buf=SimpleNamespace(n_decisions=0),
        decisions_seen=40,
        next_snapshot=80,
        next_refreeze=60,
        metrics={"loss": 1.25},
        pool_sample_counts=np.array([2, 4, 1], dtype=np.int64),
        league=SimpleNamespace(
            past_ckpts=[tmp_path / "ppo_20.pt"], latest_ckpt=tmp_path / "ppo_20.pt"
        ),
        rng=random.Random(77),
        backend_rng=random.Random(88),
        resume_backend_rng_state=None,
        replay_regularizer=_ReplayState(),
        fresh_probe=None,
        fresh_opt=None,
    )


def test_exact_training_state_roundtrip(tmp_path):
    setup = _setup(tmp_path)
    expected_agent = {name: value.clone() for name, value in setup.agent.state_dict().items()}
    expected_training_random = setup.rng.getstate()
    expected_backend_random = setup.backend_rng.getstate()
    weights = tmp_path / "ppo_40.pt"
    state_path = save_training_state(setup, weights)

    assert state_path == state_path_for(weights)
    assert weights.is_file() and state_path.is_file()
    with torch.no_grad():
        for parameter in setup.agent.parameters():
            parameter.zero_()
    setup.decisions_seen = 0
    setup.pool_sample_counts[:] = 0
    setup.league.past_ckpts.clear()
    setup.cfg["ppo"]["resume_ckpt"] = str(state_path)

    restore_training_state(setup, state_path)

    assert setup.decisions_seen == 40
    assert setup.next_snapshot == 80
    assert setup.next_refreeze == 60
    assert setup.metrics == {"loss": 1.25}
    assert setup.pool_sample_counts.tolist() == [2, 4, 1]
    assert setup.league.past_ckpts == [tmp_path / "ppo_20.pt"]
    assert setup.rng.getstate() == expected_training_random
    assert setup.resume_backend_rng_state == expected_backend_random
    assert setup.replay_regularizer.value["batch_in_epoch"] == 7
    for name, value in setup.agent.state_dict().items():
        assert torch.equal(value, expected_agent[name])


def test_exact_training_state_rejects_nonempty_rollout(tmp_path):
    setup = _setup(tmp_path)
    setup.buf.n_decisions = 1
    with pytest.raises(RuntimeError, match="empty rollout buffer"):
        save_training_state(setup, tmp_path / "bad.pt")


def test_exact_training_state_rejects_changed_config(tmp_path):
    setup = _setup(tmp_path)
    state_path = save_training_state(setup, tmp_path / "ppo.pt")
    setup.cfg["ppo"]["total_decisions"] = 200
    with pytest.raises(ValueError, match="configuration differs"):
        restore_training_state(setup, state_path)
