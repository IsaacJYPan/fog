"""Engine-backend training dispatch tests (SIMULATION_PLAN §4/§5): `train()` must run the full
rollout->update loop with NO server, and Showdown-only configs must fail loudly. Skipped without
poke-engine."""
from __future__ import annotations

from pathlib import Path

import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

POOL_DIR = str(Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b")


def _tiny_cfg(tmp_path, **overrides):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "env": {
            "backend": "engine",
            "battle_format": "gen9ou",
            "team_pool": POOL_DIR,
            "engine_batch": 4,
        },
        "ppo": {
            "total_decisions": 120,
            "batch_decisions": 60,
            "games_per_chunk": 4,
            "epochs": 1,
            "minibatch_size": 64,
            "lr": 1e-4,
        },
        "league": {"snapshot_every_decisions": 100, "scripted": ["MaxDamage", "Random"]},
        "log": {"dir": str(tmp_path / "run")},
    }
    for key, val in overrides.items():
        cfg.setdefault(key, {}).update(val) if isinstance(val, dict) else cfg.__setitem__(key, val)
    return cfg


def test_train_engine_end_to_end(tmp_path):
    from fog.training.ppo import train

    metrics = train(_tiny_cfg(tmp_path))
    assert metrics.get("decisions", 0) >= 120
    assert "loss" in metrics and "win_rate" in metrics
    # the engine path exercises the same aux losses as the Showdown path
    assert "belief_loss" in metrics and "self_belief_loss" in metrics
    import re

    ckpts = list((tmp_path / "run" / "checkpoints").glob("*.pt"))
    assert any(p.name == "ppo_final.pt" for p in ckpts)
    assert any(re.fullmatch(r"ppo_\d+\.pt", p.name) for p in ckpts), "league snapshot expected"


def test_train_engine_rejects_showdown_only_opponents(tmp_path):
    from fog.training.ppo import train

    cfg = _tiny_cfg(tmp_path)
    cfg["league"]["scripted"] = ["MaxDamage", "SimpleHeuristics"]
    with pytest.raises(ValueError, match="SimpleHeuristics|Showdown-only"):
        train(cfg)


def test_train_engine_requires_pool(tmp_path):
    from fog.training.ppo import train

    cfg = _tiny_cfg(tmp_path)
    cfg["env"].pop("team_pool")
    cfg["env"]["battle_format"] = "gen9randombattle"
    with pytest.raises(ValueError, match="team_pool|randombattle"):
        train(cfg)


def test_train_rejects_unknown_backend(tmp_path):
    from fog.training.ppo import train

    cfg = _tiny_cfg(tmp_path)
    cfg["env"]["backend"] = "warpspeed"
    with pytest.raises(ValueError, match="backend"):
        train(cfg)


# ------------------------------------------------------------- divergence controls (plan §5)
def test_cross_sim_rejects_opponent_without_engine_port(tmp_path):
    from fog.eval.cross_sim import cross_sim_gauntlet
    from fog.models.agent import FogAgent

    cfg = _tiny_cfg(tmp_path)
    cfg["cross_sim"] = {"opponent": "SimpleHeuristics", "games": 2}
    with pytest.raises(ValueError, match="engine port"):
        cross_sim_gauntlet(FogAgent.from_tables({}), cfg)


def test_scripted_action_is_deterministic_and_legal():
    import numpy as np

    from fog.scripts.fuzz_sim import scripted_action

    mask = np.zeros(14, dtype=bool)
    mask[[0, 2, 8, 13]] = True
    picks = {scripted_action(5, "p1", 3, mask) for _ in range(10)}
    assert len(picks) == 1 and picks.pop() in (0, 2, 8, 13)
    # different (seed, role, turn, mask) -> independent draws; same inputs -> same action
    assert scripted_action(5, "p1", 3, mask) == scripted_action(5, "p1", 3, mask)
    mask2 = mask.copy()
    mask2[1] = True
    draws = {scripted_action(s, "p2", 9, mask2) for s in range(40)}
    assert draws <= {0, 1, 2, 8, 13} and len(draws) > 1
