from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fog.config import load_config
from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.scripts.parse_replays import parse_replay, rows_to_arrays
from fog.training.bc import _average_metrics, bc_loss


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _batch():
    g = torch.Generator().manual_seed(183)
    count = 4
    return {
        'entities': torch.rand(count, 12, ENTITY_RAW_DIM, generator=g),
        'entity_ids': torch.zeros(count, 12, N_ID_FIELDS, dtype=torch.long),
        'field': torch.rand(count, DIMS.field_dim, generator=g),
        'history': torch.rand(count, 4, DIMS.turn_dim, generator=g),
        'history_ids': torch.zeros(count, 4, DIMS.hist_id_fields, dtype=torch.long),
        'history_state': torch.zeros(count, 4, DIMS.hist_state),
        'history_len': torch.full((count,), 4),
        'active_idx': torch.tensor([[0, 6]] * count),
        'legality_mask': torch.ones(count, 14, dtype=torch.bool),
        'policy_valid': torch.ones(count, dtype=torch.bool),
        'action': torch.arange(count),
        'outcome': torch.tensor([1., -1., 0., 1.]),
        'terminal_alive': torch.randint(2, (count, 12), generator=g).float(),
        'terminal_alive_mask': torch.ones(count, 12, dtype=torch.bool),
        'terminal_hp_difference': torch.tensor([.2, -.2, 0., .1]),
        'terminal_valid': torch.ones(count, dtype=torch.bool),
    }


def _config(aux=.1):
    cfg = load_config('configs/replay_value_cpu.yaml')
    cfg['loss'].update(outcome_aux_coef=aux, belief_coef=0., belief_chain_coef=0.)
    return cfg


def test_public_terminal_labels_match_damage_faint_and_unused_preview(tmp_path):
    replay = {'players': ['A', 'B'], 'log': '''|gametype|singles
|player|p1|A|1|1600
|player|p2|B|2|1600
|teamsize|p1|2
|teamsize|p2|2
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp|
|poke|p1|Rotom-Wash|
|poke|p2|Corviknight|
|poke|p2|Pikachu|
|teampreview
|start
|switch|p1a: Chomp|Garchomp|100/100
|switch|p2a: Bird|Corviknight|100/100
|turn|1
|move|p1a: Chomp|Flamethrower|p2a: Bird
|-damage|p2a: Bird|50/100
|move|p2a: Bird|Brave Bird|p1a: Chomp
|-damage|p1a: Chomp|25/100
|turn|2
|switch|p2a: Mouse|Pikachu|100/100
|move|p1a: Chomp|Earthquake|p2a: Mouse
|-damage|p2a: Mouse|0 fnt
|faint|p2a: Mouse
|win|A'''}
    rows = parse_replay(replay)['rows']
    assert rows
    for row in rows:
        own_alive = [1, 1, 0, 0, 0, 0]
        opp_alive = [1, 0, 0, 0, 0, 0]
        expected = own_alive + opp_alive if row['role'] == 'p1' else opp_alive + own_alive
        np.testing.assert_array_equal(row['terminal_alive'], expected)
        assert row['terminal_hp_difference'] == pytest.approx(.125 if row['role'] == 'p1' else -.125)
        np.testing.assert_array_equal(row['terminal_alive_mask'],
                                      [True, True, False, False, False, False] * 2)
        assert row['terminal_valid']
        assert not hasattr(row['obs'], 'terminal_alive')
    arrays = rows_to_arrays(rows)
    assert arrays['terminal_alive'].shape == (len(rows), 12)
    assert arrays['terminal_valid'].all()
    import polars as pl
    from fog.training.dataset import ReplayDataset, StreamingReplayDataset
    shard = tmp_path/'fixture.npz'
    np.savez_compressed(shard, **arrays)
    manifest = tmp_path/'manifest.parquet'
    pl.DataFrame({'id':['fixture'], 'shard':[str(shard)], 'n_rows':[len(rows)]}).write_parquet(manifest)
    for dataset in (ReplayDataset, StreamingReplayDataset):
        batch = next(dataset(manifest).batches(len(rows), shuffle=False))
        torch.testing.assert_close(batch['terminal_alive'], torch.from_numpy(arrays['terminal_alive']))
        torch.testing.assert_close(batch['terminal_hp_difference'], torch.from_numpy(arrays['terminal_hp_difference']))
        assert batch['terminal_valid'].all()
    np.savez_compressed(shard, **{k:v for k,v in arrays.items() if not k.startswith('terminal_')})
    for dataset in (ReplayDataset, StreamingReplayDataset):
        batch = next(dataset(manifest).batches(len(rows), shuffle=False))
        assert not batch['terminal_valid'].any()


def test_logistic_value_uses_unsaturated_logit_and_soft_draw_target():
    z = torch.tensor([-50., 50., 0.], requires_grad=True)
    output = SimpleNamespace(log_probs=torch.zeros(3, 14).log_softmax(-1),
                             value=z.tanh(), value_logit=z,
                             belief=None, belief_chain=None, self_belief=None)
    batch = {'action': torch.zeros(3, dtype=torch.long), 'outcome': torch.tensor([1., -1., 0.])}
    total, metrics = bc_loss(lambda _: output, batch, {'action_coef': 0., 'value_coef': 1.,
                                                     'value_loss': 'logistic'})
    expected = torch.nn.functional.binary_cross_entropy_with_logits(2*z, (batch['outcome']+1)/2)
    torch.testing.assert_close(total, expected)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(z.grad).all()
    torch.testing.assert_close(z.grad, torch.tensor([-2/3, 2/3, 0.]))
    assert metrics['value_brier'] == pytest.approx(2/3)


def test_auxiliary_heads_get_gradients_and_targets_cannot_change_predictions():
    cfg = _config()
    torch.manual_seed(17)
    agent = FogAgent.from_tables(cfg).eval()
    batch = _batch()
    before = agent(batch, need_belief=False)
    torch.testing.assert_close(before.value, before.value_logit.tanh(), atol=0, rtol=0)
    torch.testing.assert_close((before.value + 1) / 2, (2 * before.value_logit).sigmoid())
    changed = {**batch, 'terminal_alive': 1-batch['terminal_alive'],
               'terminal_hp_difference': -batch['terminal_hp_difference']}
    after = agent(changed, need_belief=False)
    torch.testing.assert_close(before.log_probs, after.log_probs, atol=0, rtol=0)
    torch.testing.assert_close(before.value, after.value, atol=0, rtol=0)
    total, metrics = bc_loss(agent, batch, cfg['loss'])
    total.backward()
    for head in (agent.outcome_alive, agent.outcome_hp):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    assert metrics['outcome_alive'] > 0 and metrics['outcome_hp'] > 0
    without_targets = {k:v for k,v in batch.items() if not k.startswith('terminal_')}
    playing = agent(without_targets, need_belief=False)
    assert playing.terminal_alive_logits is None and playing.terminal_hp_difference is None
    torch.testing.assert_close(before.log_probs, playing.log_probs, atol=0, rtol=0)
    torch.testing.assert_close(before.value, playing.value, atol=0, rtol=0)


def test_zero_coefficient_has_no_heads_and_preserves_policy_and_value():
    torch.manual_seed(17)
    enabled = FogAgent.from_tables(_config(.1)).eval()
    torch.manual_seed(17)
    disabled = FogAgent.from_tables(_config(0.)).eval()
    assert disabled.outcome_alive is None and disabled.outcome_hp is None
    with torch.no_grad():
        left = enabled(_batch(), need_belief=False)
        right = disabled(_batch(), need_belief=False)
    torch.testing.assert_close(left.log_probs, right.log_probs, atol=0, rtol=0)
    torch.testing.assert_close(left.value, right.value, atol=0, rtol=0)


def test_missing_terminal_targets_are_rejected_instead_of_becoming_zero_labels():
    cfg = _config()
    batch = _batch()
    batch['terminal_valid'][0] = False
    with pytest.raises(ValueError, match='reconvert'):
        bc_loss(FogAgent.from_tables(cfg).eval(), batch, cfg['loss'])


def test_ece_combines_bins_before_taking_absolute_errors():
    left = {'value_rows_count': 1., **{f'value_{kind}_bin{i}_count': 0.
             for i in range(10) for kind in ('pred', 'target')}}
    right = left.copy()
    left['value_pred_bin4_count'] = .5
    right['value_pred_bin4_count'] = .5
    right['value_target_bin4_count'] = 1.
    assert _average_metrics([(1,left),(1,right)])['value_ece'] == 0.


def test_public_terminal_health_agrees_across_views_after_a_tera_form_change():
    replay = {'players': ['A', 'B'], 'log': '''|gametype|singles
|player|p1|A|1|1600
|player|p2|B|2|1600
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Ogerpon-Cornerstone|
|poke|p2|Corviknight|
|teampreview
|start
|switch|p1a: Mask|Ogerpon-Cornerstone|100/100
|switch|p2a: Bird|Corviknight|100/100
|turn|1
|move|p1a: Mask|Horn Leech|p2a: Bird
|-damage|p2a: Bird|50/100
|move|p2a: Bird|Brave Bird|p1a: Mask
|-damage|p1a: Mask|25/100
|-terastallize|p1a: Mask|Rock
|detailschange|p1a: Mask|Ogerpon-Cornerstone-Tera
|turn|2
|move|p1a: Mask|Horn Leech|p2a: Bird
|-damage|p2a: Bird|0 fnt
|faint|p2a: Bird
|win|A'''}
    rows = parse_replay(replay)['rows']
    assert {row['role'] for row in rows} == {'p1', 'p2'}
    for row in rows:
        expected = 1/24 if row['role'] == 'p1' else -1/24
        assert row['terminal_hp_difference'] == pytest.approx(expected)



def test_epoch_calibration_matches_the_holdout_report():
    from fog.eval.replay_holdout_report import value_calibration
    values = torch.tensor([-.2, .4, .9, -1.])
    outcome = torch.tensor([-1., 1., -1., 0.])
    metrics = []
    for part in (slice(0, 2), slice(2, 4)):
        output = SimpleNamespace(log_probs=torch.zeros(2, 14).log_softmax(-1),
                                 value=values[part], belief=None, belief_chain=None, self_belief=None)
        batch = {'action':torch.zeros(2, dtype=torch.long), 'outcome':outcome[part]}
        _, measured = bc_loss(lambda _: output, batch, {'_value_diagnostics':True})
        metrics.append((2, measured))
    observed = _average_metrics(metrics)
    expected = value_calibration(values.numpy(), outcome.numpy())
    assert observed['value_brier'] == pytest.approx(expected['brier'], abs=1e-12)
    assert observed['value_ece'] == pytest.approx(expected['ece'], abs=1e-12)



def test_outcome_auxiliaries_respect_the_existing_trunk_gradient_switch():
    cfg = _config()
    cfg['model']['aux_into_trunk'] = False
    agent = FogAgent.from_tables(cfg).eval()
    _, _, components = bc_loss(agent, _batch(), cfg['loss'], return_components=True)
    components['outcome_aux'].backward()
    assert agent.outcome_alive.weight.grad.abs().sum() > 0
    assert agent.outcome_hp.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.shared_latent.parameters())
    assert all(p.grad is None for p in agent.shared_to_value.parameters())
    assert all(p.grad is None for p in agent.backbone.parameters())
