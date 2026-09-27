"""Tests for GAE. The full ppo_update path is covered in test_ppo_update.py."""
from __future__ import annotations

import numpy as np

from fog.training.ppo import compute_gae


def test_gae_terminal_single_step():
    # one-step episode, γ=1, λ=0.95: adv = r + γ*V(next)*(1-done) - V(cur)
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.2, 0.0], dtype=np.float32)  # V(cur)=0.2, bootstrap 0
    dones = np.array([1.0], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=0.95)
    assert abs(adv[0] - (1.0 - 0.2)) < 1e-6
    assert abs(ret[0] - (adv[0] + 0.2)) < 1e-6


def test_gae_shapes():
    t = 5
    rewards = np.zeros(t, dtype=np.float32)
    values = np.zeros(t + 1, dtype=np.float32)
    dones = np.zeros(t, dtype=np.float32)
    dones[-1] = 1.0
    adv, ret = compute_gae(rewards, values, dones)
    assert adv.shape == (t,) and ret.shape == (t,)
