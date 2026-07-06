"""Tests for determinized search (§8), the mirror/opponent-view (§3.3), and config overrides."""
from __future__ import annotations

import numpy as np
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


def test_determinized_action_returns_legal():
    agent = FogAgent()
    a = determinized_action(agent, _batch(legal=(0, 1, 8)), k=4)
    assert a in (0, 1, 8)


def test_determinized_single_legal_shortcuts():
    agent = FogAgent()
    a = determinized_action(agent, _batch(legal=(3,)), k=8)
    assert a == 3


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
