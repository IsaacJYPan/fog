"""Tests for RolloutBuffer: reward placement, GAE, leakage shaping, nested stacking."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fog.training.ppo import ppo_update
from fog.training.rollout import RolloutBuffer, _stack_histories


def _frag(x: float = 0.0, t: int = 3):
    return {
        "obs": np.full(4, x, dtype=np.float32),
        "legality_mask": np.array([True] * 4 + [False] * 10),
        "history": np.full((t, 2), x, dtype=np.float32),
        "opp_revealed": {
            "tokens": np.full((6, 15), -1, dtype=np.int64),
            "masks": {"item": np.ones((6, 3), dtype=bool)},
        },
    }


def test_terminal_reward_and_gae_lambda1():
    """With gamma=1, lam=1, zero values: advantage_t = outcome for every t; returns likewise."""
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    g = buf.start_game()
    for i in range(4):
        buf.add_decision(g, _frag(i), action=i % 4, log_prob=-1.0, value=0.0)
    buf.end_game(g, outcome=+1.0)
    batch = buf.to_batch()
    assert torch.allclose(batch["advantages"], torch.ones(4))
    assert torch.allclose(batch["returns"], torch.ones(4))
    assert torch.allclose(batch["game_outcomes"], torch.ones(4))
    assert batch["actions"].tolist() == [0, 1, 2, 3]


def test_value_baseline_subtracts():
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    g = buf.start_game()
    buf.add_decision(g, _frag(), action=0, log_prob=-1.0, value=0.5)
    buf.end_game(g, outcome=-1.0)
    batch = buf.to_batch()
    # single step: adv = r - v = -1 - 0.5
    assert torch.allclose(batch["advantages"], torch.tensor([-1.5]))
    assert torch.allclose(batch["returns"], torch.tensor([-1.0]))


def test_leakage_shaping_only_penalizes_positive_delta():
    buf = RolloutBuffer(gamma=1.0, lam=1.0, beta=0.1)
    g = buf.start_game()
    for dI in (2.0, -3.0, 0.0):
        buf.add_decision(g, _frag(), action=0, log_prob=-1.0, value=0.0, leak_delta=dI)
    buf.end_game(g, outcome=+1.0)
    batch = buf.to_batch()
    # rewards: [-0.2, 0, 0+1]; gamma=lam=1 => returns_t = sum of future rewards
    assert torch.allclose(batch["returns"], torch.tensor([0.8, 1.0, 1.0]))


def test_multi_game_flatten_and_no_cross_game_leak():
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    g1, g2 = buf.start_game(), buf.start_game()
    for _ in range(2):
        buf.add_decision(g1, _frag(1.0), action=0, log_prob=-1.0, value=0.0)
    buf.add_decision(g2, _frag(2.0), action=1, log_prob=-1.0, value=0.0)
    buf.end_game(g1, outcome=+1.0)
    buf.end_game(g2, outcome=-1.0)
    batch = buf.to_batch()
    assert batch["advantages"].shape == (3,)
    # g1's win must not bleed into g2's advantage
    assert torch.allclose(batch["advantages"], torch.tensor([1.0, 1.0, -1.0]))
    assert torch.allclose(batch["game_outcomes"], torch.tensor([1.0, 1.0, -1.0]))
    assert batch["obs"].shape == (3, 4)
    assert batch["opp_revealed"]["tokens"].shape == (3, 6, 15)
    assert batch["opp_revealed"]["masks"]["item"].shape == (3, 6, 3)


def test_unfinished_game_raises_and_post_end_add_raises():
    buf = RolloutBuffer()
    g = buf.start_game()
    buf.add_decision(g, _frag(), action=0, log_prob=-1.0, value=0.0)
    with pytest.raises(RuntimeError, match="not ended"):
        buf.to_batch()
    buf.end_game(g, outcome=1.0)
    with pytest.raises(RuntimeError, match="already ended"):
        buf.add_decision(g, _frag(), action=0, log_prob=-1.0, value=0.0)
    with pytest.raises(RuntimeError, match="ended twice"):
        buf.end_game(g, outcome=1.0)


def test_zero_decision_game_dropped():
    buf = RolloutBuffer()
    g = buf.start_game()
    buf.end_game(g, outcome=1.0)
    assert buf.n_decisions == 0


def test_history_front_padding():
    out = _stack_histories([np.ones((2, 3)), np.full((5, 3), 2.0)])
    assert out.shape == (2, 5, 3)
    assert (out[0, :3] == 0).all() and (out[0, 3:] == 1).all()  # front-padded, recent last
    assert (out[1] == 2.0).all()


def test_faints_bonus_lands_on_last_step():
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    g = buf.start_game()
    buf.add_decision(g, _frag(), action=0, log_prob=-1.0, value=0.0)
    buf.end_game(g, outcome=1.0, faints_bonus=0.04)
    assert torch.allclose(buf.to_batch()["returns"], torch.tensor([1.04]))


def test_progress_reward_telescopes_and_cannot_be_farmed_by_waiting():
    buf = RolloutBuffer(gamma=1.0, lam=1.0, progress_coef=0.25)
    game = buf.start_game()
    # Repeating 0.2 for an extra decision earns nothing.  Only the net 0.0 -> 0.5 progress counts.
    for score in (0.0, 0.2, 0.2):
        buf.add_decision(
            game, _frag(), action=0, log_prob=-1.0, value=0.0, progress_score=score
        )
    buf.end_game(game, outcome=-1.0, final_progress_score=0.5)
    batch = buf.to_batch()
    assert "step_rewards" in batch
    assert torch.isfinite(batch["step_rewards"]).all()
    assert float(batch["step_rewards"].abs().sum()) > 0.0
    assert torch.allclose(batch["returns"], torch.tensor([-0.875, -0.925, -0.925]))
    assert np.isclose(float(batch["returns"][0]), -1.0 + 0.25 * 0.5)


def test_progress_reward_requires_complete_measurements_when_enabled():
    buf = RolloutBuffer(progress_coef=0.25)
    game = buf.start_game()
    buf.add_decision(game, _frag(), action=0, log_prob=-1.0, value=0.0)
    buf.end_game(game, outcome=-1.0, final_progress_score=-1.0)
    with pytest.raises(RuntimeError, match="no progress score"):
        buf.to_batch()


def test_progress_reward_is_capped_below_terminal_result():
    with pytest.raises(ValueError, match="win/loss remains dominant"):
        RolloutBuffer(progress_coef=0.75)


def test_batch_feeds_ppo_update_end_to_end():
    """Integration: buffer output must be directly consumable by ppo_update (contract C4)."""
    from tests.test_ppo_update import TinyPolicyAgent

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    buf = RolloutBuffer()
    for _ in range(6):
        g = buf.start_game()
        for _ in range(rng.integers(1, 5)):
            frag = {
                "obs": rng.normal(size=8).astype(np.float32),
                "legality_mask": np.array([True] * 4 + [False] * 10),
            }
            buf.add_decision(
                g, frag, action=int(rng.integers(0, 4)), log_prob=float(np.log(0.25)),
                value=float(rng.normal() * 0.1),
            )
        buf.end_game(g, outcome=float(rng.choice([-1.0, 1.0])))
    batch = buf.to_batch()
    agent = TinyPolicyAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    metrics = ppo_update(agent, opt, batch, {"ppo": {"minibatch_size": 8}})
    assert np.isfinite(metrics["loss"])
    buf.clear()
    assert buf.n_decisions == 0
