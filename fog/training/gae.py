"""Trajectory-return utilities shared by every rollout backend."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantages and bootstrapped returns for one trajectory."""
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    accumulator = 0.0
    for index in reversed(range(steps)):
        nonterminal = 1.0 - dones[index]
        delta = (
            rewards[index]
            + gamma * values[index + 1] * nonterminal
            - values[index]
        )
        accumulator = delta + gamma * lam * nonterminal * accumulator
        advantages[index] = accumulator
    return advantages, advantages + values[:steps]
