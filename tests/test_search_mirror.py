"""Tests for determinized search (§8), the mirror/opponent-view (§3.3), and config overrides."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fog.config import apply_overrides
from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.search.determinized import determinized_action


def _batch(b=1, legal=(0, 1, 8)):
    torch.manual_seed(0)
    tokens = torch.full((b, 6, 9), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, 320, (b, 6))
    mask = torch.zeros(b, 14, dtype=torch.bool)
    for a in legal:
        mask[:, a] = True
    return {
        "entities": torch.randn(b, 12, ENTITY_RAW_DIM),
        "entity_ids": torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.randn(b, DIMS.field_dim),
        "history": torch.randn(b, 4, DIMS.turn_dim),
        "active_idx": torch.tensor([[0, 6]] * b),
        "legality_mask": mask,
        "opp_revealed": {"tokens": tokens, "masks": None},
    }


def _fog_observation(history_len=3):
    """A zero fog Observation with correct shapes — stands in for a simulated leaf."""
    from fog.encoding.observation import Observation

    return Observation(
        entities=np.zeros((12, ENTITY_RAW_DIM), np.float32),
        entity_ids=np.zeros((12, N_ID_FIELDS), np.int64),
        field=np.zeros(DIMS.field_dim, np.float32),
        history=np.zeros((history_len, DIMS.turn_dim), np.float32),
        history_len=history_len,
        legality_mask=np.array([True] + [False] * 13),
        active_idx=np.array([0, 6], dtype=np.int64),
        opp_revealed={"tokens": np.full((6, 9), -1, np.int64), "masks": {}},
    )


def test_determinized_requires_step_fn():
    """The old step_fn=None fallback scored every action identically — now a hard error."""
    agent = FogAgent()
    with pytest.raises(ValueError, match="step_fn"):
        determinized_action(agent, _batch(legal=(0, 1, 8)), step_fn=None, k=2)


def test_determinized_action_returns_legal():
    agent = FogAgent()
    a = determinized_action(agent, _batch(legal=(0, 1, 8)), step_fn=lambda a_, w: 0.0, k=4)
    assert a in (0, 1, 8)


def test_determinized_single_legal_shortcuts():
    agent = FogAgent()
    a = determinized_action(agent, _batch(legal=(3,)), step_fn=lambda a_, w: 0.0, k=8)
    assert a == 3


def test_determinized_prefers_winning_action():
    """With the prior off, the action whose simulated worlds all win must be chosen."""
    agent = FogAgent()
    a = determinized_action(
        agent, _batch(legal=(0, 1, 8)),
        step_fn=lambda a_, w: 1.0 if a_ == 8 else -1.0, k=4, prior_weight=0.0,
    )
    assert a == 8


def test_determinized_prior_blend_dominates_at_large_lambda():
    """When every simulated outcome ties, a large λ must recover the policy argmax (fix 2)."""
    agent = FogAgent()
    agent.eval()
    batch = _batch(legal=(0, 1, 8))
    with torch.no_grad():
        prior_argmax = int(agent(batch).log_probs[0].argmax())
    a = determinized_action(
        agent, batch, step_fn=lambda a_, w: 0.0, k=4, prior_weight=1000.0
    )
    assert a == prior_argmax


def test_determinized_scores_fog_observation_leaves():
    """Observation leaves (fix 1) batch through the value net; mixed history lengths pad."""
    agent = FogAgent()
    calls = []

    def step_fn(a_, w):
        calls.append(a_)
        if a_ == 0:
            return 0.5  # terminal
        return _fog_observation(history_len=2 + a_ % 3)

    a = determinized_action(agent, _batch(legal=(0, 1, 8)), step_fn=step_fn, k=3)
    assert a in (0, 1, 8)
    assert sorted(set(calls)) == [0, 1, 8]  # every legal action was simulated


def test_determinized_failed_samples_fall_back_to_root_value():
    """step_fn returning None must skip the sample, not crash or bury the action."""
    agent = FogAgent()
    a = determinized_action(
        agent, _batch(legal=(0, 1, 8)),
        step_fn=lambda a_, w: None if a_ == 1 else 0.0, k=2,
    )
    assert a in (0, 1, 8)


def test_config_overrides():
    cfg = {"leakage": {"beta": 0.0}, "ppo": {"lr": 1e-4}}
    apply_overrides(cfg, ["leakage.beta=0.03", "ppo.lr=0.0005", "model.oracle=set"])
    assert cfg["leakage"]["beta"] == 0.03
    assert cfg["ppo"]["lr"] == 0.0005
    assert cfg["model"]["oracle"] == "set"


def test_all_milestone_configs_load():
    from fog.config import load_config

    for name in ("m0_plumbing", "m1_poc", "m1_5_belief", "m2_oracle", "m3_leakage",
                 "bc_ou", "train_ou", "m4_ou_selfplay", "eval_ou"):
        cfg = load_config(f"{name}.yaml")
        assert "experiment" in cfg
        assert cfg["pins"]["usage_month"] == "2026-06"  # inherited from default


def test_mirror_opponent_view_shapes():
    from fog.constants.tables import TABLES
    from fog.encoding.mirror import build_opponent_view
    from fog.encoding.observation import Observation

    # a zero observation with correct shapes (no live battle needed)
    ent = np.zeros((12, ENTITY_RAW_DIM), dtype=np.float32)
    eid = np.zeros((12, N_ID_FIELDS), dtype=np.int64)
    obs = Observation(
        entities=ent, entity_ids=eid, field=np.zeros(DIMS.field_dim, np.float32),
        history=np.zeros((3, DIMS.turn_dim), np.float32), history_len=3,
        legality_mask=np.array([True] + [False] * 13), active_idx=np.array([0, 6]),
        opp_revealed={"tokens": np.full((6, 9), -1, np.int64), "masks": {}},
    )
    my_view = np.full((6, 9), -1, np.int64)
    my_view[:, 0] = 1  # species seen
    truth = np.zeros((6, 9), np.int64)
    batch, mv = build_opponent_view(obs, my_view, truth, TABLES)
    assert batch["entities"].shape == (1, 12, ENTITY_RAW_DIM)
    assert mv["tokens"].shape == (1, 6, 9)
    assert "labels" in mv and mv["labels"]["species"].shape == (1, 6)
