"""Behavior contracts for the post-review correctness pass.

These tests intentionally describe the behavior we want *before* the implementation changes land.
Contracts that the current code does not yet satisfy are strict expected failures.  Keeping them
here makes the review concrete without making the ordinary suite unusable; each xfail must be
removed as its corresponding fix is implemented.

The public-replay contract reflects the chosen scope: public replays may pretrain public-state
representations, value estimates, and censored opponent beliefs, but must not imitate human policy
actions that depended on unavailable private requests/team sheets.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import AgentOutput, FogAgent
from fog.models.belief import BeliefModel
from fog.models.encoder import Backbone
from fog.training.bc import bc_loss
from fog.training.rollout import RolloutBuffer


def _small_backbone(*, dropout: float = 0.0) -> Backbone:
    """Small, fast backbone with the same history/entity paths as the real agent."""
    return Backbone(
        {
            "entity_hidden": 16,
            "entity_heads": 4,
            "entity_transformer_layers": 1,
            "history_gru_hidden": 16,
            "state_dim": 16,
            "dropout": dropout,
        },
        sizes={
            "n_species": 8,
            "n_items": 8,
            "n_abilities": 8,
            "n_moves": 16,
            "n_natures": 5,
            "n_ev_units": 8,
        },
    )


def _backbone_inputs(history: torch.Tensor) -> dict:
    b = history.shape[0]
    return {
        "entities": torch.zeros(b, 12, ENTITY_RAW_DIM),
        "entity_ids": torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.zeros(b, DIMS.field_dim),
        "history": history,
        "active_idx": torch.tensor([[0, 6]] * b, dtype=torch.long),
    }


def test_history_representation_ignores_batch_padding():
    """Adding blank batch-padding before the same real turns must not change the state."""
    torch.manual_seed(10)
    backbone = _small_backbone().eval()  # isolate history padding from dropout
    real_turns = torch.randn(1, 2, DIMS.turn_dim)
    padded = torch.cat([torch.zeros(1, 3, DIMS.turn_dim), real_turns], dim=1)

    with torch.no_grad():
        short_state = backbone(**_backbone_inputs(real_turns))
        padded_state = backbone(**_backbone_inputs(padded))

    assert torch.allclose(short_state.private_state, padded_state.private_state, atol=1e-6)
    assert torch.allclose(short_state.public_state, padded_state.public_state, atol=1e-6)
    assert torch.allclose(short_state.history, padded_state.history, atol=1e-6)


def test_history_sidecars_ignore_front_padding():
    torch.manual_seed(101)
    backbone = _small_backbone().eval()
    real = torch.randn(1, 2, DIMS.turn_dim)
    ids = torch.randint(0, 3, (1, 2, DIMS.hist_id_fields))
    state = torch.randn(1, 2, DIMS.hist_state).clamp(-1, 1)
    short = _backbone_inputs(real)
    short.update(history_len=torch.tensor([2]), history_ids=ids, history_state=state)
    padded = _backbone_inputs(torch.cat([torch.zeros(1, 3, DIMS.turn_dim), real], dim=1))
    padded.update(
        history_len=torch.tensor([2]),
        history_ids=torch.cat([torch.zeros(1, 3, DIMS.hist_id_fields, dtype=torch.long), ids], dim=1),
        history_state=torch.cat([torch.zeros(1, 3, DIMS.hist_state), state], dim=1),
    )

    with torch.no_grad():
        short_state = backbone(**short)
        padded_state = backbone(**padded)
    assert torch.allclose(short_state.private_state, padded_state.private_state, atol=1e-6)
    assert torch.allclose(short_state.public_state, padded_state.public_state, atol=1e-6)
    assert torch.allclose(short_state.history, padded_state.history, atol=1e-6)


def test_explicit_zero_dropout_is_deterministic_in_training_mode():
    """With dropout explicitly zero, identical inputs/weights must give identical outputs."""
    torch.manual_seed(11)
    backbone = _small_backbone(dropout=0.0).train()
    inputs = _backbone_inputs(torch.randn(2, 3, DIMS.turn_dim))

    first = backbone(**inputs)
    second = backbone(**inputs)

    assert torch.equal(first.private_state, second.private_state)
    assert torch.equal(first.public_state, second.public_state)
    assert torch.equal(first.history, second.history)


@pytest.mark.xfail(
    strict=True,
    reason="BeliefModel.teacher_forced has not been implemented",
)
def test_belief_teacher_forcing_lets_earlier_truth_affect_later_predictions():
    """A true earlier item should be available when training the later move conditional."""
    torch.manual_seed(12)
    model = BeliefModel(
        state_dim=16,
        d=16,
        n_species=8,
        n_items=6,
        n_abilities=5,
        n_moves=12,
        n_natures=5,
        n_ev_units=8,
    ).eval()
    state = torch.randn(1, 16)
    revealed = torch.full((1, 6, 15), -1, dtype=torch.long)
    revealed[:, :, 0] = 1  # standard preview: species is public
    view = {"tokens": revealed, "masks": {}}

    truth_a = revealed.clone()
    truth_b = revealed.clone()
    truth_a[0, 0, 1] = 2  # same visible state, different true hidden item
    truth_b[0, 0, 1] = 4

    out_a = model.teacher_forced(state, view, truth_a)
    out_b = model.teacher_forced(state, view, truth_b)

    # move0 is later in the chain than item, so its training distribution should depend on item.
    assert not torch.allclose(out_a.logits["move"][0, 0, 0], out_b.logits["move"][0, 0, 0])


def _agent_batch() -> dict:
    tokens = torch.full((1, 6, 15), -1, dtype=torch.long)
    tokens[:, :, 0] = 1
    return {
        **_backbone_inputs(torch.zeros(1, 2, DIMS.turn_dim)),
        "legality_mask": torch.ones(1, 14, dtype=torch.bool),
        "opp_revealed": {"tokens": tokens, "masks": {}},
    }


def test_teacher_tokens_never_change_policy_or_visible_belief_summary():
    """Hidden truth may teach the belief loss, but it must never become a policy input."""
    torch.manual_seed(13)
    agent = FogAgent().eval()
    batch_a = _agent_batch()
    batch_b = copy.deepcopy(batch_a)
    batch_a["belief_teacher_tokens"] = torch.zeros(1, 6, 15, dtype=torch.long)
    batch_b["belief_teacher_tokens"] = torch.ones(1, 6, 15, dtype=torch.long)

    with torch.no_grad():
        out_a = agent(batch_a)
        out_b = agent(batch_b)

    assert torch.equal(out_a.log_probs, out_b.log_probs)
    assert torch.equal(out_a.value, out_b.value)
    # the visible belief summary is now the marginal logits (b_vec is gone, §3.6) — still teacher-free
    for fam in out_a.belief.logits:
        assert torch.equal(out_a.belief.logits[fam], out_b.belief.logits[fam])


class _FixedPublicStateAgent:
    """Tiny stand-in whose action probabilities are deliberately non-uniform."""

    def __call__(self, batch):
        b = batch["action"].shape[0]
        logits = torch.arange(14, dtype=torch.float32).expand(b, -1)
        return AgentOutput(
            log_probs=torch.log_softmax(logits, dim=-1),
            value=torch.zeros(b),
            belief=None,
            self_belief=None,
        )


def test_replay_policy_loss_ignores_unaligned_action_labels():
    agent = _FixedPublicStateAgent()
    base = {
        "action": torch.tensor([0, 13]),
        "policy_valid": torch.tensor([True, False]),
        "outcome": torch.tensor([0.0, 0.0]),
    }
    changed_invalid = {**base, "action": torch.tensor([0, 1])}
    cfg = {
        "action_coef": 1.0,
        "belief_coef": 0.0,
        "belief_chain_coef": 0.0,
        "self_belief_coef": 0.0,
        "value_coef": 0.0,
    }

    loss_a, metrics_a = bc_loss(agent, base, cfg)
    loss_b, metrics_b = bc_loss(agent, changed_invalid, cfg)

    assert torch.equal(loss_a, loss_b)
    assert metrics_a["policy_valid_count"] == 1
    assert metrics_b["policy_valid_count"] == 1


@pytest.mark.xfail(
    strict=True,
    reason="public-state mode does not yet disable the human policy-action loss",
)
def test_public_state_pretraining_does_not_learn_from_human_action_slots():
    """Changing an unavailable/private human action label must not change public-state loss."""
    agent = _FixedPublicStateAgent()
    batch_a = {"action": torch.tensor([0]), "outcome": torch.tensor([1.0])}
    batch_b = {"action": torch.tensor([13]), "outcome": torch.tensor([1.0])}
    cfg = {"policy_coef": 0.0, "belief_coef": 0.0, "self_belief_coef": 0.0, "value_coef": 1.0}

    loss_a, _ = bc_loss(agent, batch_a, cfg)
    loss_b, _ = bc_loss(agent, batch_b, cfg)

    assert torch.equal(loss_a, loss_b)


@pytest.mark.xfail(
    strict=True,
    reason="rollout batches do not yet retain a per-row game id",
)
def test_rollout_batch_retains_game_ids_for_held_out_evaluation():
    """Readability rows need their battle identity so whole battles can be held out."""
    buf = RolloutBuffer()
    for expected_gid in range(2):
        gid = buf.start_game()
        for turn in range(2):
            buf.add_decision(
                gid,
                {"feature": np.array([turn], dtype=np.float32)},
                action=0,
                log_prob=0.0,
                value=0.0,
            )
        buf.end_game(gid, outcome=1.0)

    batch = buf.to_batch()

    assert batch["game_id"].tolist() == [0, 0, 1, 1]


@pytest.mark.xfail(
    strict=True,
    reason="readability split_probe_games has not been implemented",
)
def test_readability_split_never_places_one_game_in_multiple_splits():
    """Probe training, validation, and test rows must come from disjoint battles."""
    from fog.eval.readability import split_probe_games

    game_ids = torch.arange(10).repeat_interleave(3)
    batch = {"game_id": game_ids, "feature": torch.arange(30)}
    train, validation, test = split_probe_games(
        batch, train_fraction=0.6, validation_fraction=0.2, seed=7
    )

    groups = [set(part["game_id"].tolist()) for part in (train, validation, test)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(range(10))


@pytest.mark.xfail(
    strict=True,
    reason="full-state training checkpoint save/load helpers have not been implemented",
)
def test_training_checkpoint_restores_model_optimizer_progress_and_rng(tmp_path):
    """A resumed run should continue from the same training state, not merely reload weights."""
    from fog.training.checkpoint import load_training_checkpoint, save_training_checkpoint

    torch.manual_seed(14)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()  # populate Adam's running state

    saved_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path / "training.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        decisions=1234,
        config={"seed": 14, "env": {"backend": "engine"}},
    )
    expected_next_random = torch.rand(4)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(100.0)
    metadata = load_training_checkpoint(path, model=model, optimizer=optimizer)
    actual_next_random = torch.rand(4)

    assert metadata["decisions"] == 1234
    assert metadata["config_hash"]
    assert all(torch.equal(model.state_dict()[k], v) for k, v in saved_weights.items())
    assert optimizer.state  # Adam state was restored, not recreated empty
    assert torch.equal(actual_next_random, expected_next_random)
