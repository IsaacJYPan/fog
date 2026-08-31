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
    truth = np.arange(90, dtype=np.int64).reshape(6, 15) % 5
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


def test_belief_targets_aligned_to_each_decisions_clamp():
    """Targets follow the INPUT packing convention per decision (fix 2026-07-12): a revealed
    move — clamped into subslot 0 of the observation — whose paste position is 2 must become the
    subslot-0 target for THAT decision, while an unrevealed decision of the same game keeps
    paste order (the truth is broadcast per game, the reveal state is per decision)."""
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0] = [3, 1, 0, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 2]

    def frag_with_view(view):
        f = dict(_frag())
        f["opp_revealed"] = {"tokens": view, "masks": {}}
        return f

    revealed = np.full((6, 15), -1, dtype=np.int64)
    revealed[0, 0] = 3
    revealed[0, 3] = 12  # paste position 2, clamped at subslot 0
    g = buf.start_game()
    buf.add_decision(g, frag_with_view(revealed), action=0, log_prob=-1.0, value=0.0)
    buf.add_decision(g, frag_with_view(np.full((6, 15), -1, dtype=np.int64)),
                     action=0, log_prob=-1.0, value=0.0)
    buf.end_game(g, outcome=1.0, belief_targets=truth)

    batch = buf.to_batch()
    assert batch["belief_targets"]["move"][0, 0].tolist() == [12, 10, 11, 13]
    assert batch["belief_targets"]["move"][1, 0].tolist() == [10, 11, 12, 13]


def test_illusion_contradiction_censors_ambiguous_slot_supervision():
    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0] = [3, 1, 0, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 2]
    view = np.full((6, 15), -1, dtype=np.int64)
    view[0, 0] = 3
    view[0, 3:7] = [20, 21, 22, 23]
    frag = dict(_frag())
    frag["opp_revealed"] = {"tokens": view, "masks": {}}
    game = buf.start_game()
    buf.add_decision(game, frag, action=0, log_prob=-1.0, value=0.0)
    buf.end_game(game, outcome=1.0, belief_targets=truth)

    valid = buf.to_batch()["belief_valid"]

    assert all(not bool(mask[0, 0].any()) for mask in valid.values())


def test_no_item_current_state_keeps_original_item_target_valid():
    from fog.constants.schema import NO_ITEM
    from fog.constants.tables import TABLES

    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    truth = np.full((6, 15), -1, dtype=np.int64)
    truth[0, 0] = 0
    truth[0, 1] = TABLES.item_index("leftovers")
    view = np.full((6, 15), -1, dtype=np.int64)
    view[0, 0] = 0
    view[0, 1] = TABLES.item_index(NO_ITEM)
    frag = dict(_frag())
    frag["opp_revealed"] = {"tokens": view, "masks": {}}
    game = buf.start_game()
    buf.add_decision(game, frag, action=0, log_prob=-1.0, value=0.0)
    buf.end_game(game, outcome=1.0, belief_targets=truth)

    batch = buf.to_batch()
    assert batch["belief_valid"]["item"][0, 0]
    assert int(batch["belief_targets"]["item"][0, 0]) == truth[0, 1]


def test_masked_cross_entropy_dropped_entry_counter():
    from fog.training.losses import masked_cross_entropy

    logits = torch.zeros(4, 3)
    logits[1, 2] = -float("inf")  # target 2 sits on a legality-masked class -> dropped
    target = torch.tensor([0, 2, 1, -1])
    valid = torch.ones(4, dtype=torch.bool)
    stats: dict = {}
    loss = masked_cross_entropy(logits, target, valid, stats=stats)
    assert torch.isfinite(loss)
    assert stats == {
        "dropped_nonfinite": 1,
        "dropped_examples": [{"flat_index": 1, "target": 2}],
        "valid": 2,
    }


def test_all_censored_cross_entropy_is_a_backward_safe_zero():
    from fog.training.losses import masked_cross_entropy

    logits = torch.tensor(
        [[0.0, -float("inf"), 1.0]], requires_grad=True
    )
    loss = masked_cross_entropy(
        logits,
        torch.tensor([-1]),
        torch.tensor([False]),
    )

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_belief_loss_fires_in_ppo_update():
    from fog.models.agent import FogAgent

    torch.manual_seed(0)
    # tiny agent via placeholder sizes is fine here; we only need belief.logits to exist
    agent = FogAgent({"model": {}})
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

    buf = RolloutBuffer(gamma=1.0, lam=1.0)
    truth = np.zeros((6, 15), dtype=np.int64)  # all class-0 labels
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
            "opp_revealed": {"tokens": np.full((6, 15), -1, dtype=np.int64), "masks": {}},
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
