from types import SimpleNamespace
import random

import pytest
import torch

from fog.training.heuristic_guidance import (
    attach_guidance,
    guidance_loss,
    guidance_weight,
    validate_guidance,
)
from fog.training.ppo import ppo_update
from tests.test_ppo_update import TinyPolicyAgent, make_ppo_batch


def test_teacher_schedule_tapers_to_zero_at_the_recorded_decision_budget():
    settings = {"heuristic_coef": .3, "heuristic_coef_final": 0,
                "heuristic_decay_decisions": 300000}
    assert [guidance_weight(settings, n, 600000) for n in (0, 150000, 300000, 600000)] == [
        .3, .15, 0., 0.]


@pytest.mark.parametrize("settings", [
    {"heuristic_coef": float('nan')}, {"heuristic_coef": -.1},
    {"heuristic_coef_final": float('inf')}, {"heuristic_decay_decisions": 0},
    {"heuristic_decay_decisions": 1.5}, {"heuristic_decay_decisions": True},
])
def test_bad_guidance_schedule_fails_before_training(settings):
    with pytest.raises(ValueError):
        guidance_weight(settings, 0, 100)


def test_guidance_requires_a_supported_teacher_and_authoritative_backend():
    with pytest.raises(ValueError, match='requires ppo.heuristic_teacher'):
        validate_guidance({'ppo': {'heuristic_coef': .3}})
    with pytest.raises(ValueError, match='showdown_direct'):
        validate_guidance({'env': {'backend': 'engine'}, 'ppo': {'heuristic_teacher': 'simpleheuristics'}})
    with pytest.raises(ValueError, match='simpleheuristics or maxdamage'):
        validate_guidance({'ppo': {'heuristic_teacher': 'unknown'}})
    validate_guidance({'ppo': {'total_decisions': 0}})


def test_disabled_guidance_does_not_attach_a_label_provider():
    adapter = SimpleNamespace(supervised_label_provider=None)
    assert attach_guidance(adapter, {}) is None
    assert adapter.supervised_label_provider is None


def test_teacher_stream_is_reproducible_and_leaves_global_sampling_alone(monkeypatch):
    import fog.env.actions
    import fog.env.showdown_local_runner

    monkeypatch.setattr(fog.env.showdown_local_runner, '_scripted_order',
                        lambda _name, _battle, rng: rng.randrange(4))
    monkeypatch.setattr(fog.env.actions, 'map_order_to_action', lambda _battle, order: order)
    adapter = SimpleNamespace(supervised_label_provider=None)
    teacher = attach_guidance(adapter, {'env': {'backend': 'showdown_direct'},
        'ppo': {'heuristic_teacher': 'simpleheuristics', 'heuristic_coef': .3}})
    observation = SimpleNamespace(legality_mask=[True] * 14)
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    teacher.start_game(123)
    first = [teacher(None, observation, None) for _ in range(20)]
    teacher.start_game(123)
    assert first == [teacher(None, observation, None) for _ in range(20)]
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_teacher_loss_uses_only_real_choices_and_refuses_illegal_targets():
    logits = torch.zeros(2, 14, requires_grad=True)
    legal = torch.zeros(2, 14, dtype=torch.bool)
    legal[0, :4] = True
    legal[1, 2] = True
    log_probs = logits.masked_fill(~legal, float('-inf')).log_softmax(-1)
    loss, metrics = guidance_loss(log_probs, torch.tensor([0, 2]), legal)
    assert float(loss.detach()) == pytest.approx(float(torch.log(torch.tensor(4.))))
    assert metrics['heuristic_choice_rows'] == 1
    loss.backward()
    assert bool(logits.grad[0].abs().sum() > 0)
    assert bool(logits.grad[1].abs().sum() == 0)
    with pytest.raises(ValueError, match='illegal'):
        guidance_loss(log_probs, torch.tensor([0, 3]), legal)


def test_guidance_can_train_on_learner_states_without_changing_ppo_targets():
    torch.manual_seed(79)
    agent = TinyPolicyAgent()
    batch = make_ppo_batch(n=32, seed=18)
    batch['advantages'].zero_()
    batch['supervised_label'] = torch.zeros(32, dtype=torch.long)
    advantages, actions = batch['advantages'].clone(), batch['actions'].clone()
    with torch.no_grad():
        before = agent(batch).log_probs[:, 0].mean().item()
        batch['old_log_probs'] = agent(batch).log_probs.gather(1, batch['actions'][:, None]).squeeze(1)
    metrics = ppo_update(agent, torch.optim.Adam(agent.parameters(), lr=.01), batch,
        {'ppo': {'minibatch_size': 32, 'epochs': 1, 'heuristic_coef': .3, 'entropy_coef': 0},
         'loss': {'value_coef': 0}})
    with torch.no_grad():
        assert agent(batch).log_probs[:, 0].mean().item() > before
    assert metrics['heuristic_coef'] == .3
    assert metrics['heuristic_choice_rows'] == 32
    assert metrics['clip_frac'] == 0
    assert torch.equal(actions, batch['actions']) and torch.equal(advantages, batch['advantages'])


def test_enabled_guidance_refuses_a_rollout_without_teacher_labels():
    agent = TinyPolicyAgent()
    with pytest.raises(ValueError, match='labels are missing'):
        ppo_update(agent, torch.optim.Adam(agent.parameters()), make_ppo_batch(),
                   {'ppo': {'heuristic_coef': .3}})


def test_zero_guidance_has_identical_weights_to_ordinary_ppo():
    torch.manual_seed(19)
    baseline = TinyPolicyAgent()
    candidate = TinyPolicyAgent()
    candidate.load_state_dict(baseline.state_dict())
    batch = make_ppo_batch(n=32, seed=14)
    tagged = {**batch, 'supervised_label': torch.zeros(32, dtype=torch.long)}
    cfg = {'ppo': {'minibatch_size': 16, 'epochs': 2, 'heuristic_coef': 0}}
    torch.manual_seed(77)
    ppo_update(baseline, torch.optim.Adam(baseline.parameters()), batch, cfg)
    torch.manual_seed(77)
    ppo_update(candidate, torch.optim.Adam(candidate.parameters()), tagged, cfg)
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in baseline.state_dict().items())


def test_teacher_metadata_is_not_an_input_to_the_typed_acting_policy():
    from tests.test_shared_latent import _batch, _typed_contextual_agent

    agent = _typed_contextual_agent(entity_dim=32).eval()
    batch = _batch()
    with torch.no_grad():
        first = agent({**batch, 'supervised_label': torch.tensor([0, 1])})
        second = agent({**batch, 'supervised_label': torch.tensor([2, 3])})
    assert torch.equal(first.log_probs, second.log_probs)
    assert torch.equal(first.value, second.value)
