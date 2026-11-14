"""Tests for warm-start resumption wiring: ppo.init_ckpt + league.past_ckpts seeding.

Added when the g0_v2_champion run died at 163k/200k decisions and had to be finished from its
155k snapshot (configs/g0_v2_champion_resume.yaml is the consumer).
"""
from __future__ import annotations

import pytest
import torch

from fog.training.ppo import _TrainSetup


def _cfg(tmp_path, **overrides):
    cfg = {
        "device": "cpu",
        "env": {"battle_format": "gen9ou", "team_pool": "data/teams/pool_a"},
        "ppo": {"total_decisions": 100, "batch_decisions": 50},
        "log": {"dir": str(tmp_path / "run")},
    }
    for key, val in overrides.items():
        cfg.setdefault(key, {}).update(val)
    return cfg


def test_init_ckpt_loads_agent_weights(tmp_path):
    donor = _TrainSetup(_cfg(tmp_path / "a"))
    ckpt = tmp_path / "donor.pt"
    torch.save(donor.agent.state_dict(), ckpt)

    fresh = _TrainSetup(_cfg(tmp_path / "b", ppo={"init_ckpt": str(ckpt)}))
    # Different seed-path constructions would differ without the load; loaded == donor exactly.
    for name, param in fresh.agent.state_dict().items():
        assert torch.equal(param, donor.agent.state_dict()[name]), name


def test_init_ckpt_absent_starts_fresh_and_differs_from_donor(tmp_path):
    donor = _TrainSetup(_cfg(tmp_path / "a"))
    ckpt = tmp_path / "donor.pt"
    torch.save(donor.agent.state_dict(), ckpt)
    # Perturb the donor so "fresh init happens to equal donor" cannot false-pass.
    with torch.no_grad():
        next(iter(donor.agent.parameters())).add_(1.0)
    torch.save(donor.agent.state_dict(), ckpt)

    fresh = _TrainSetup(_cfg(tmp_path / "b"))
    loaded = _TrainSetup(_cfg(tmp_path / "c", ppo={"init_ckpt": str(ckpt)}))
    diffs = [
        name
        for name, param in loaded.agent.state_dict().items()
        if not torch.equal(param, fresh.agent.state_dict()[name])
    ]
    assert diffs, "init_ckpt had no effect on the agent's weights"


def test_league_past_ckpts_seeded(tmp_path):
    snap1 = tmp_path / "ppo_50.pt"
    snap2 = tmp_path / "ppo_100.pt"
    for path in (snap1, snap2):
        path.write_bytes(b"placeholder")  # existence-checked only at setup time

    setup = _TrainSetup(
        _cfg(tmp_path / "a", league={"past_ckpts": [str(snap1), str(snap2)]})
    )
    assert [p.name for p in setup.league.past_ckpts] == ["ppo_50.pt", "ppo_100.pt"]
    assert setup.league.latest_ckpt is not None
    assert setup.league.latest_ckpt.name == "ppo_100.pt"


def test_league_past_ckpts_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _TrainSetup(
            _cfg(tmp_path / "a", league={"past_ckpts": [str(tmp_path / "nope.pt")]})
        )


def test_warm_started_run_resumes_without_reloading_initial_weights(tmp_path):
    from copy import deepcopy

    from fog.training.training_state import save_training_state

    donor = _TrainSetup(_cfg(tmp_path / "donor"))
    initial = tmp_path / "initial.pt"
    torch.save(donor.agent.state_dict(), initial)
    cfg = _cfg(tmp_path / "warm", ppo={"init_ckpt": str(initial)})
    warm = _TrainSetup(cfg)
    parameter = next(warm.agent.parameters())
    parameter.grad = torch.ones_like(parameter)
    warm.opt.step()
    warm.opt.zero_grad()
    warm.decisions_seen = 40
    state = save_training_state(warm, warm.ckpt_dir / "ppo_40.pt")
    initial.unlink()  # full resume must depend only on the saved training state
    resumed_cfg = deepcopy(cfg)
    resumed_cfg["ppo"]["resume_ckpt"] = str(state)

    resumed = _TrainSetup(resumed_cfg)

    assert resumed.decisions_seen == 40
    for name, value in warm.agent.state_dict().items():
        assert torch.equal(value, resumed.agent.state_dict()[name]), name
    expected = next(iter(warm.opt.state.values()))
    actual = next(iter(resumed.opt.state.values()))
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
