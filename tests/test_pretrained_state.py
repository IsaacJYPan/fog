import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fog.pretrained.data import MatchedMixture
from fog.pretrained.kakuna import KakunaRuntime
from fog.pretrained.state import save_training_state, load_training_state


def test_full_resume_restores_optimizer_sampler_and_all_rngs(tmp_path):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sampler = MatchedMixture(["p1", "p2"], ["f1", "f2"], 123)
    route = torch.Generator().manual_seed(78)
    recipe = {"warmup_updates": 500}

    def update():
        sampled = sampler.next_batch()
        noise = torch.rand(1, 3) + float(np.random.rand())
        route_value = torch.randint(100, (1,), generator=route)
        optimizer.zero_grad()
        model(noise).square().sum().backward()
        optimizer.step()
        return sampled, noise, route_value

    update()
    path = tmp_path / "state.pt"
    save_training_state(
        path,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        route_generator=route,
        update=1,
        identity="frozen",
        config=recipe,
        elapsed_seconds=7,
    )
    expected = update()
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    load_training_state(
        path,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        route_generator=route,
        identity="frozen",
        config=recipe,
    )
    actual = update()
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1]) and torch.equal(actual[2], expected[2])
    assert all(torch.equal(expected_model[key], value) for key, value in model.state_dict().items())
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, expected_optimizer["state"][key][field])
    with pytest.raises(ValueError, match="identity"):
        load_training_state(
            path,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            route_generator=route,
            identity="other",
            config=recipe,
        )


def test_probability_floor_is_removed_by_common_deployment_mask():
    class Native:
        pass_obs_keys_to_actor = ["illegal_actions"]

        def get_state_embedding(self, **kwargs):
            return kwargs["rl2s"], None

        def actor(self, latent, **kwargs):
            return SimpleNamespace(probs=torch.tensor([[[[0.8, 0.1, 0.1]]]]))

    runtime = KakunaRuntime(Native(), None, None, None, "weights")
    probabilities, _, _ = runtime.distribution(
        {"illegal_actions": torch.tensor([[[False, True, False]]])},
        torch.zeros(1, 1, 4),
        torch.zeros(1, 1, 1),
    )
    torch.testing.assert_close(probabilities, torch.tensor([[[8 / 9, 0.0, 1 / 9]]]))
    with pytest.raises(ValueError, match="no legal probability"):
        runtime.distribution(
            {"illegal_actions": torch.ones(1, 1, 3, dtype=torch.bool)},
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 1),
        )
