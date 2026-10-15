from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fog.models.agent import AgentOutput, FogAgent
from fog.training import curriculum_pilot as pilot
from fog.training.ppo import ppo_update


class TinyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.pi = torch.nn.Linear(4, 14)
        self.v = torch.nn.Linear(4, 1)

    def forward(self, batch):
        hidden = self.backbone(batch["obs"]).tanh()
        logits = self.pi(hidden).masked_fill(~batch["legality_mask"], -torch.inf)
        return AgentOutput(log_probs=logits.log_softmax(-1), value=self.v(hidden).tanh().squeeze(-1))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(FogAgent, "from_tables", lambda config: TinyAgent())
    monkeypatch.setattr(pilot, "probability_contract", lambda *args: {"passed": True})

    def collect(client, actor, anchor, bank, sampler, schedule, recipe):
        n = recipe["positions_per_batch"]
        batch = {"obs": torch.randn(n, 3), "legality_mask": torch.tensor([[True]*4+[False]*10]*n)}
        with torch.no_grad():
            output = actor(batch)
            actions = torch.multinomial(output.log_probs.exp(), 1).squeeze(1)
            batch.update(actions=actions, old_log_probs=output.log_probs.gather(1, actions[:, None]).squeeze(1),
                         bc_log_probs=anchor(batch).log_probs)
        batch.update(advantages=torch.randn(n), returns=torch.tensor(sampler.choice([-1., 1.], n), dtype=torch.float32))
        batch["game_outcomes"] = batch["returns"].clone()
        schedule.state["episodes"] += 1
        schedule.state["full_episodes"] += 1
        return batch, [], n

    def update(*args):
        result = ppo_update(*args)
        result.setdefault("belief_dropped_rate", 0.)
        return result

    monkeypatch.setattr(pilot, "collect", collect)
    monkeypatch.setattr("fog.training.ppo.ppo_update", update)
    initial = tmp_path/"initial.pt"
    torch.manual_seed(8)
    torch.save(TinyAgent().state_dict(), initial)
    config = {"model": {"oracle": "none", "aux_into_trunk": True, "public_action_features_only": True}}
    recipe = {"condition": "ordinary", "max_batches": 2, "positions_per_batch": 4, "minibatch_size": 2,
              "epochs": 1, "warmup_batches": 1, "save_every": 1, "learning_rate": 1e-5,
              "auxiliary_gradient_cap": .25, "seed": 19, "training_seconds": 120}
    bank = SimpleNamespace(sha256="a"*64, pool_sha256="b"*64, contract_batch=lambda: {})
    parent = tmp_path/"parent"
    pilot.train(config, recipe, bank, initial, parent)
    saved = torch.load(parent/"latest.state.pt", weights_only=False)
    child = dict(recipe, parent_state_sha256=pilot.digest(parent/"latest.state.pt"), parent_identity=saved["identity"],
                 parent_batches=2, parent_optimizer_steps=4)
    return config, child, bank, initial, parent


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_parent_continuation_preserves_adam_rng_anchor_and_exact_child_resume(tmp_path, setup):
    config, recipe, bank, initial, parent = setup
    options = {"parent": parent/"latest.state.pt", "anchor_checkpoint": initial}
    continuous, resumed = tmp_path/"continuous", tmp_path/"resumed"
    result = pilot.train(config, recipe, bank, parent/"latest.pt", continuous, **options)
    pilot.train(config, recipe, bank, parent/"latest.pt", resumed, stop_after_batches=1, **options)
    pilot.train(config, recipe, bank, parent/"latest.pt", resumed, resume=resumed/"latest.state.pt", **options)
    a = torch.load(continuous/"latest.state.pt", weights_only=False)
    b = torch.load(resumed/"latest.state.pt", weights_only=False)
    assert all(equal(a[key], b[key]) for key in ("agent", "optimizer", "random", "schedule", "lineage"))
    assert result["total_optimizer_steps"] == 8 and result["total_optimized_positions"] == 16
    assert max(int(state["step"]) for state in a["optimizer"]["state"].values()) == 8
    assert result["anchor_sha256"] == pilot.digest(initial)
    assert result["lineage"]["parent_state_sha256"] == pilot.digest(parent/"latest.state.pt")
    assert a["schedule"]["episodes"] == 4


@pytest.mark.parametrize("field,value", [("parent_state_sha256", "0"*64), ("parent_identity", "0"*64),
                                         ("parent_batches", 3), ("parent_optimizer_steps", 5)])
def test_changed_parent_provenance_or_exposure_is_rejected(tmp_path, setup, field, value):
    config, recipe, bank, initial, parent = setup
    recipe = dict(recipe, **{field: value})
    with pytest.raises(ValueError, match="Parent"):
        pilot.train(config, recipe, bank, parent/"latest.pt", tmp_path/"bad", parent=parent/"latest.state.pt",
                    anchor_checkpoint=initial)


def test_parent_weights_cannot_be_silently_replaced(tmp_path, setup):
    config, recipe, bank, initial, parent = setup
    with pytest.raises(ValueError, match="Starting weights"):
        pilot.train(config, recipe, bank, initial, tmp_path/"bad", parent=parent/"latest.state.pt",
                    anchor_checkpoint=initial)


def test_cuda_actor_cannot_silently_run_on_cpu(tmp_path, setup):
    config, recipe, bank, initial, parent = setup
    recipe = deepcopy(recipe)
    recipe["actor_device"] = "cuda"
    with pytest.raises(ValueError, match="CUDA actor"):
        pilot.train(config, recipe, bank, parent/"latest.pt", tmp_path/"bad", parent=parent/"latest.state.pt",
                    anchor_checkpoint=initial)


def test_parent_cannot_implicitly_change_the_policy_anchor(tmp_path, setup):
    config, recipe, bank, initial, parent = setup
    with pytest.raises(ValueError, match="explicit frozen policy anchor"):
        pilot.train(config, recipe, bank, parent/"latest.pt", tmp_path/"bad", parent=parent/"latest.state.pt")
