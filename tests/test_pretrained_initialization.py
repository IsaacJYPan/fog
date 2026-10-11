import copy
import hashlib
import json

import pytest
import torch

from fog.pretrained.checkpoints import export_candidates
from fog.pretrained.data import MatchedMixture
from fog.pretrained.initialization import (
    initialize_auxiliary, set_learning_rates, starting_checkpoint,
    training_optimizer, validate_warm_start,
)
from fog.pretrained.kakuna import sha256
from fog.pretrained.state import load_training_state, save_training_state


class Native(torch.nn.Linear):
    @property
    def trainable_params(self):
        return self.parameters()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = Native(2, 1)
        self.outcome = torch.nn.Linear(2, 1)

    def trainable_parameters(self):
        return list(self.parameters())


def sources(tmp_path):
    samples = tmp_path / "samples.pt"
    torch.save([torch.ones(2)], samples)
    warm = {"version": 1, "optimizer": "reset", "sampler": "reset", "sources": {}}
    for condition, value in (("ordinary", 2.0), ("joint", 3.0)):
        model = Model()
        with torch.no_grad():
            model.native.weight.fill_(value)
            model.outcome.bias.fill_(value + 5)
        config = {"condition": condition, "updates": 4, "seed": 17}
        identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        out = tmp_path / condition
        export_candidates(model, out, update=4, identity=identity, config=config)
        native, auxiliary = out / "native_000004.pt", out / "auxiliary_000004.pt"
        reference = out / "reference.pt"
        torch.save({"checkpoint_sha256": sha256(native), "samples_sha256": sha256(samples),
                    "probabilities": [torch.tensor([0.3, 0.7])]}, reference)
        warm["sources"][condition] = {
            "native": str(native), "native_sha256": sha256(native),
            "auxiliary": str(auxiliary), "auxiliary_sha256": sha256(auxiliary),
            "reference": str(reference), "reference_sha256": sha256(reference), "update": 4,
        }
    return {"warm_start": warm, "condition": "joint"}, samples


def test_weight_warm_start_restores_the_matching_heads_and_preserves_native(tmp_path):
    config, samples = sources(tmp_path)
    metadata = validate_warm_start(config, samples=samples)
    assert metadata["joint"]["source_update"] == 4
    model = Model()
    model.native.load_state_dict(torch.load(starting_checkpoint(config), weights_only=True))
    native = copy.deepcopy(model.native.state_dict())
    initialize_auxiliary(model, config)
    assert model.outcome.bias.item() == 8
    assert all(torch.equal(value, native[key]) for key, value in model.native.state_dict().items())
    config["condition"] = "ordinary"
    assert starting_checkpoint(config).parent.name == "ordinary"


@pytest.mark.parametrize("name", ["native", "auxiliary", "reference"])
def test_changed_warm_start_inputs_fail_before_training(tmp_path, name):
    config, samples = sources(tmp_path)
    source = config["warm_start"]["sources"]["joint"]
    with open(source[name], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="input hash changed"):
        validate_warm_start(config, samples=samples)


def test_unmatched_parent_recipes_and_wrong_probability_inputs_are_rejected(tmp_path):
    config, samples = sources(tmp_path)
    source = config["warm_start"]["sources"]["joint"]
    state = torch.load(source["auxiliary"], weights_only=True)
    state["config"]["seed"] += 1
    state["identity"] = hashlib.sha256(json.dumps(state["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    torch.save(state, source["auxiliary"])
    source["auxiliary_sha256"] = sha256(starting_checkpoint(config).with_name("auxiliary_000004.pt"))
    with pytest.raises(ValueError, match="matched parent recipes"):
        validate_warm_start(config, samples=samples)
    samples.write_bytes(b"wrong input batch")
    with pytest.raises(ValueError, match="probability reference differs"):
        validate_warm_start(config, samples=samples)


def test_separate_head_rate_preserves_native_rate_and_ordinary_head_weights():
    config = {"condition": "joint", "learning_rate": 2e-5, "head_learning_rate": 2e-4, "warmup_updates": 2}
    model = Model()
    optimizer, parameters = training_optimizer(model, config)
    assert len(parameters) == 4
    assert set_learning_rates(optimizer, config, 0) == {"native": 1e-5, "auxiliary": 1e-4}
    native, head = model.native.weight.detach().clone(), model.outcome.weight.detach().clone()
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    torch.testing.assert_close((head - model.outcome.weight) / (native - model.native.weight),
                               torch.full_like(head, 10), atol=0.03, rtol=0)
    config["condition"] = "ordinary"
    optimizer, parameters = training_optimizer(model, config)
    before = model.outcome.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    assert torch.equal(before, model.outcome.weight)


def test_two_rate_optimizer_full_state_resume_is_exact_and_recipe_is_immutable(tmp_path):
    model = Model()
    config = {"condition": "joint", "learning_rate": 2e-5, "head_learning_rate": 2e-4, "warmup_updates": 2}
    optimizer, _ = training_optimizer(model, config)
    sampler = MatchedMixture(["p1"], ["f1"], 7)
    generator = torch.Generator().manual_seed(9)

    def step(update):
        schedule = sampler.next_batch()
        query = torch.rand(2, generator=generator)
        noise = torch.randn(2)
        optimizer.zero_grad(set_to_none=True)
        (model.native(noise).square().sum() + model.outcome(query).square().sum()).backward()
        rates = set_learning_rates(optimizer, config, update)
        optimizer.step()
        return schedule, query, noise, rates

    step(0)
    path = tmp_path / "state.pt"
    kwargs = dict(model=model, optimizer=optimizer, sampler=sampler,
                  route_generator=generator, identity="new-recipe", config=config)
    save_training_state(path, update=1, elapsed_seconds=3, **kwargs)
    expected = step(1)
    expected_state = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    load_training_state(path, **kwargs)
    actual = step(1)
    assert expected[0] == actual[0] and expected[3] == actual[3]
    assert torch.equal(expected[1], actual[1]) and torch.equal(expected[2], actual[2])
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected_state[key])
    assert optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, expected_optimizer["state"][key][name])
    with pytest.raises(ValueError, match="immutable recipe"):
        load_training_state(path, **{**kwargs, "config": {**config, "head_learning_rate": 1e-3}})


def test_old_joint_optimizer_layout_still_has_one_group():
    optimizer, _ = training_optimizer(Model(), {"condition": "joint", "learning_rate": 2e-5})
    assert len(optimizer.param_groups) == 1
