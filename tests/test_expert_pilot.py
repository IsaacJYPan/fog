import json

import numpy as np
import pytest
import torch

from fog.config import load_config
from fog.training.expert_data import ExpertData
from fog.training.expert_pilot import capped_update, recipe_digest, restore_rng, rng_state, validate_recipe


def recipe():
    return {"max_updates": 3, "games_per_batch": 2, "rows_per_game": 4,
            "microbatch": 8, "warmup_updates": 1, "save_every": 1,
            "evaluate_every": 3, "learning_rate": 0.0003, "gradient_clip": 1.5,
            "auxiliary_gradient_cap": 0.25, "training_seconds": 3600}


def test_independent_recipe_rejects_inherited_checkpoint_and_answer_features():
    config = load_config("configs/gen9ou_typed_token_teacher.yaml")
    validate_recipe(recipe(), config)
    config["bc"]["init_ckpt"] = "old.pt"
    with pytest.raises(ValueError, match="warm start"):
        validate_recipe(recipe(), config)
    config["bc"]["init_ckpt"] = None
    config["model"]["public_action_features_only"] = False
    with pytest.raises(ValueError, match="public action"):
        validate_recipe(recipe(), config)


def test_recipe_identity_binds_sampling_and_optimizer_settings():
    first = recipe()
    old = recipe_digest(first, {})
    first["microbatch"] = 4
    assert recipe_digest(first, {}) != old


def test_rng_roundtrip_preserves_next_sampler_and_torch_draws():
    sampler = np.random.default_rng(13)
    state = rng_state(sampler)
    expected_sampler, expected_torch = sampler.integers(1000), torch.rand(3)
    restore_rng(state, sampler)
    assert sampler.integers(1000) == expected_sampler
    assert torch.equal(torch.rand(3), expected_torch)


def test_shared_cap_preserves_full_auxiliary_head_gradient(monkeypatch):
    class Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1, bias=False)
            self.auxiliary = torch.nn.Parameter(torch.ones(()))
            self.backbone.weight.data.fill_(1)

    class Data:
        def batch(self, *_, **__):
            return None

    def loss(agent, batch, config, **_):
        primary = agent.backbone.weight.sum()
        auxiliary = 100 * agent.backbone.weight.sum() * agent.auxiliary
        return primary + auxiliary, {"policy_valid_count": 1, "belief_dropped_count": 0,
                                     "belief_chain_dropped_count": 0}, {"action": primary, "belief": auxiliary}

    monkeypatch.setattr("fog.training.bc.bc_loss", loss)
    agent = Agent()
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)
    settings = dict(recipe(), microbatch=1, gradient_clip=1000)
    metrics = capped_update(agent, optimizer, Data(), [None, None], {"loss": {}}, settings, "cpu")
    assert metrics["capped_auxiliary_primary_ratio"] == pytest.approx(0.25)
    assert agent.backbone.weight.item() == pytest.approx(0.875)
    assert agent.auxiliary.item() == pytest.approx(-9.0)


def test_portable_data_rejects_path_escape_and_composition_overlap(tmp_path):
    def save(records):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"format": "fog.teacher-replay-data", "version": 1,
                                   "complete": True, "records": records}))
        return path

    rows = [{"id": name, "split": name, "compositions": [name],
             "shard": "../outside.npz", "shard_sha256": "x", "policy_rows": 1}
            for name in ("train", "development")]
    with pytest.raises(ValueError, match="escaped"):
        ExpertData(save(rows))
    rows[1]["compositions"] = ["train"]
    with pytest.raises(ValueError, match="composition-disjoint"):
        ExpertData(save(rows))
