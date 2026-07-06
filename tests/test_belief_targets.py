"""Belief-target supervision path: rollout buffer -> C2 dict -> ppo_update belief loss."""
from __future__ import annotations

import numpy as np
import torch

from fog.training.rollout import RolloutBuffer


def _frag():
    return {
        "obs": np.zeros(4, dtype=np.float32),
        "legality_mask": np.array([True] * 4 + [False] * 10),
    }


def test_belief_targets_broadcast_and_split():
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    truth = np.arange(54, dtype=np.int64).reshape(6, 9) % 5
    g = buf.start_game()
    for _ in range(3):
        buf.add_decision(g, _frag(), action=0, log_prob=-1.0, value=0.0)
    buf.end_game(g, outcome=1.0, belief_targets=truth)

    # second game with NO belief truth -> those rows must be censored (-1 -> invalid)
    g2 = buf.start_game()
    buf.add_decision(g2, _frag(), action=1, log_prob=-1.0, value=0.0)
    buf.end_game(g2, outcome=-1.0)

    batch = buf.to_batch()
    assert "belief_targets" in batch and "belief_valid" in batch
    tg, vd = batch["belief_targets"], batch["belief_valid"]
    assert tg["species"].shape == (4, 6)
    assert tg["move"].shape == (4, 6, 4)
    # first 3 rows valid (truth broadcast), last row invalid (no truth)
    assert vd["species"][:3].all()
    assert not vd["species"][3].any()
    # species targets match the broadcast truth for game 1
    assert (tg["species"][0] == torch.as_tensor(truth[:, 0])).all()


def test_belief_loss_fires_in_ppo_update():
    from fog.models.agent import FogAgent

    torch.manual_seed(0)
    # tiny agent via placeholder sizes is fine here; we only need belief.logits to exist
    agent = FogAgent({"model": {"use_belief_vec": False}})
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    truth = np.zeros((6, 9), dtype=np.int64)  # all class-0 labels
    truth[:, 0] = np.arange(6) % 3
    for _ in range(2):
        g = buf.start_game()
        frag = {
            "entities": np.zeros((12, ENTITY_RAW_DIM), dtype=np.float32),
            "entity_ids": np.zeros((12, N_ID_FIELDS), dtype=np.int64),
            "field": np.zeros(DIMS.field_dim, dtype=np.float32),
            "history": np.zeros((3, DIMS.turn_dim), dtype=np.float32),
            "active_idx": np.array([0, 6], dtype=np.int64),
            "legality_mask": np.array([True, True] + [False] * 12),
            "opp_revealed": {"tokens": np.full((6, 9), -1, dtype=np.int64), "masks": {}},
        }
        buf.add_decision(g, frag, action=0, log_prob=float(np.log(0.5)), value=0.0)
        buf.end_game(g, outcome=1.0, belief_targets=truth)

    batch = buf.to_batch()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    metrics = ppo_update_ref(agent, opt, batch)
    assert "belief_loss" in metrics and np.isfinite(metrics["belief_loss"])
    assert metrics["belief_loss"] > 0


def ppo_update_ref(agent, opt, batch):
    from fog.training.ppo import ppo_update

    return ppo_update(agent, opt, batch, {"ppo": {"minibatch_size": 2}, "loss": {"belief_coef": 1.0}})
