import pytest
import torch

from fog.eval.learning_signal import gradient_geometry, outcome_metrics, probability_contract


def test_contract_reports_changed_probability_and_refuses_changed_support():
    old = torch.tensor([[.2, .8, 0.], [.5, .5, 0.]]).log()
    actions = torch.tensor([0, 1])
    recorded = old[torch.arange(2), actions]
    result = probability_contract(old, old, actions, recorded)
    assert result['ratio_min'] == result['ratio_max'] == 1
    assert result['categorical_kl_max'] == result['clip_fraction'] == 0
    new = torch.tensor([[.4, .6, 0.], [.5, .5, 0.]]).log()
    result = probability_contract(old, new, actions, recorded)
    assert result['ratio_max'] == pytest.approx(2)
    assert result['clip_fraction'] == .5
    new[0, 2] = -5
    with pytest.raises(ValueError, match='support'):
        probability_contract(old, new, actions, recorded)


def test_outcome_metrics_weight_games_equally_and_handle_constant_outcomes():
    report = outcome_metrics([{'values': [1.] * 100, 'outcome': 1},
                              {'values': [1.], 'outcome': -1}], baseline=0.)
    assert report['value_mse']['mean'] == 2
    assert report['baseline_mse']['mean'] == 1
    assert report['explained_variance'] == 0
    constant = outcome_metrics([{'values': [0.], 'outcome': 1}], baseline=1.)
    assert constant['explained_variance'] is None


def test_contract_rejects_recorded_probability_broadcasting_and_invalid_actions():
    lp = torch.tensor([[.2, .8], [.5, .5]]).log()
    actions = torch.tensor([0, 1])
    with pytest.raises(ValueError, match='shapes'):
        probability_contract(lp, lp, actions, torch.zeros(2, 1))
    with pytest.raises(ValueError, match='shapes'):
        probability_contract(lp[:0], lp[:0], actions[:0], torch.empty(0))
    with pytest.raises(ValueError, match='indices'):
        probability_contract(lp, lp, torch.tensor([0, 2]), torch.zeros(2))
    with pytest.raises(ValueError, match='indices'):
        probability_contract(lp, lp, actions.float(), torch.zeros(2))


def test_gradient_geometry_detects_cancellation_without_modifying_gradients():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    unused = torch.nn.Parameter(torch.tensor([3.]))
    p.grad = torch.tensor([7., 8.])
    result = gradient_geometry({'policy': 3*p[0] + 4*p[1], 'value': -6*p[0] - 8*p[1]},
                               [('shared', p), ('unused', unused)], {'all': [0, 1]})['all']
    assert result['norms'] == {'policy': 5., 'value': 10.}
    assert result['combined_norm'] == 5
    assert result['combined_policy_cosine'] == -1
    assert result['cosine_with_policy']['value'] == -1
    assert p.grad.tolist() == [7., 8.]
    assert unused.grad is None
