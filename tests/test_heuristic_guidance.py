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


def test_cloning_updates_ignore_executed_actions_advantages_and_gae_returns():
    torch.manual_seed(31)
    first = TinyPolicyAgent()
    second = TinyPolicyAgent()
    second.load_state_dict(first.state_dict())
    batch = make_ppo_batch(n=32, seed=15)
    batch['supervised_label'] = torch.zeros(32, dtype=torch.long)
    batch['game_outcomes'] = torch.ones(32)
    alternate = {**batch, 'actions': (batch['actions'] + 1) % 4,
                 'advantages': -100 * batch['advantages'], 'returns': -batch['returns'],
                 'old_log_probs': batch['old_log_probs'] - 2}
    cfg = {'ppo': {'policy_objective': 'heuristic_bc', 'value_target': 'outcome',
                   'epochs': 3, 'minibatch_size': 16, 'entropy_coef': 0, 'target_kl': None},
           'loss': {'value_coef': .25}}
    with torch.no_grad():
        before = first(batch).log_probs[:, 0].mean().item()
    for agent, rows in ((first, batch), (second, alternate)):
        torch.manual_seed(51)
        metrics = ppo_update(agent, torch.optim.Adam(agent.parameters(), lr=.01), rows, cfg)
        assert metrics['optimizer_steps'] == 6
        assert metrics['policy_objective_heuristic_bc'] == 1
        assert metrics['value_target_outcome'] == 1
        assert metrics['stopped_for_kl'] == 0
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
    with torch.no_grad():
        assert first(batch).log_probs[:, 0].mean().item() > before


@pytest.mark.parametrize('settings', [
    {'policy_objective': 'unknown'}, {'value_target': 'unknown'},
    {'policy_objective': 'heuristic_bc', 'target_kl': .01},
    {'policy_objective': 'heuristic_bc', 'heuristic_coef': .3},
    {'policy_objective': 'heuristic_bc', 'heuristic_coef_final': .3},
    {'heuristic_voluntary_switch_weight': 0},
    {'heuristic_voluntary_switch_weight': float('nan')},
])
def test_invalid_cloning_recipe_fails_before_training(settings):
    with pytest.raises(ValueError):
        validate_guidance({'env': {'backend': 'showdown_direct'},
                           'ppo': {'heuristic_teacher': 'simpleheuristics', **settings}})


def test_cloning_requires_teacher_labels_and_outcome_value_requires_completed_games():
    agent = TinyPolicyAgent()
    batch = make_ppo_batch()
    optimizer = torch.optim.Adam(agent.parameters())
    with pytest.raises(ValueError, match='requires ppo.heuristic_teacher'):
        validate_guidance({'ppo': {'policy_objective': 'heuristic_bc'}})
    with pytest.raises(ValueError, match='labels are missing'):
        ppo_update(agent, optimizer, batch, {'ppo': {'policy_objective': 'heuristic_bc'}})
    with pytest.raises(ValueError, match='completed-game outcome'):
        ppo_update(agent, optimizer, batch, {'ppo': {'value_target': 'outcome'}})
    with pytest.raises(ValueError, match='finite and in'):
        ppo_update(agent, optimizer, {**batch, 'game_outcomes': torch.full((64,), float('nan'))},
                   {'ppo': {'value_target': 'outcome'}})
    with pytest.raises(ValueError, match='required belief supervision'):
        ppo_update(agent, optimizer, batch, {'loss': {'require_belief_labels': True}})
    empty_labels = {key: {'item': torch.zeros(64, 6, dtype=torch.long if key.endswith('targets') else torch.bool)}
                    for key in ('belief_targets', 'belief_valid', 'self_belief_targets', 'self_belief_valid')}
    with pytest.raises(ValueError, match='no eligible targets'):
        ppo_update(agent, optimizer, {**batch, **empty_labels}, {'loss': {'require_belief_labels': True}})


def test_switch_weight_does_not_upweight_forced_switches_and_nll_stays_comparable():
    logits = torch.zeros(3, 14, requires_grad=True)
    legal = torch.zeros(3, 14, dtype=torch.bool)
    legal[0, [0, 8]] = True
    legal[1, [8, 9]] = True
    legal[2, :4] = True
    log_probs = logits.masked_fill(~legal, float('-inf')).log_softmax(-1)
    labels = torch.tensor([8, 8, 0])
    loss, metrics = guidance_loss(log_probs, labels, legal, voluntary_switch_weight=8.)
    expected = (9 * torch.log(torch.tensor(2.)) + torch.log(torch.tensor(4.))) / 10
    assert loss.detach() == pytest.approx(expected.item())
    assert metrics['heuristic_nll'] == pytest.approx((torch.log(torch.tensor(2.)) * 2 + torch.log(torch.tensor(4.))).item() / 3)
    assert metrics['heuristic_voluntary_switch_rows'] == 1
    assert metrics['heuristic_forced_switch_rows'] == 1
    loss.backward()
    assert logits.grad[0, 8].abs() == pytest.approx(logits.grad[1, 8].abs().item() * 8)
