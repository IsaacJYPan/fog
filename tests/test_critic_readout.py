import pytest
import torch

from fog.eval.critic_readout import fit_readout, game_weighted_mse


def test_game_weighting_does_not_reward_long_trajectories():
    p = torch.cat([torch.ones(100), torch.ones(1)])
    y = torch.cat([torch.ones(100), -torch.ones(1)])
    ids = torch.cat([torch.zeros(100), torch.ones(1)])
    assert game_weighted_mse(p, y, ids) == 2


def test_readout_learns_fixed_features_without_changing_original_head():
    head = torch.nn.Linear(1, 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    x = torch.tensor([[-1.], [1.], [-1.], [1.]])
    fit = dict(features=x, outcome=x[:, 0], game_id=torch.arange(4))
    dev = dict(features=x, outcome=x[:, 0], game_id=torch.arange(4, 8))
    result = fit_readout(head, fit, dev, seed=1, lr=.1, max_epochs=20, patience=5, min_delta=0.)
    assert result['best_development_mse'] < .1
    assert torch.equal(head.weight, torch.zeros_like(head.weight))
    assert all(p.grad is None for p in head.parameters())


def test_epoch_zero_is_retained_when_outcome_fit_does_not_generalize():
    head = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(head.weight)
    fit = dict(features=torch.ones(4, 1), outcome=torch.ones(4), game_id=torch.arange(4))
    dev = dict(features=torch.ones(4, 1), outcome=-torch.ones(4), game_id=torch.arange(4, 8))
    result = fit_readout(head, fit, dev, seed=1, lr=.1, max_epochs=5, patience=2)
    assert result['best_epoch'] == 0
    assert result['best_development_mse'] == 1


def test_readout_refuses_shared_gradients_and_overlapping_development_games():
    head = torch.nn.Linear(1, 1)
    fit = dict(features=torch.ones(2, 1), outcome=torch.ones(2), game_id=torch.arange(2))
    with pytest.raises(ValueError, match='identities overlap'):
        fit_readout(head, fit, fit, seed=1)
    dev = {**fit, 'game_id': torch.arange(2, 4)}
    with pytest.raises(ValueError, match='detached'):
        fit_readout(head, {**fit, 'features': fit['features'].requires_grad_()}, dev, seed=1)
