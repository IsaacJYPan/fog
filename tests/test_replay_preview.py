from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from fog.config import load_config
from fog.constants.tables import TABLES
from fog.encoding.entities import L
from fog.models.agent import FogAgent
from fog.scripts.parse_replays import parse_replay_policy_versions, rows_to_arrays
from fog.training import bc
from fog.training.dataset import PreviewDataset
from fog.training.replay_policy_data import complete_private_team
from fog.training.replay_preview import preview_arrays, public_lead_labels
from tests.test_bc_resume import _assert_state_equal, _interrupt_after_epoch
from tests.test_order_blind_model import observations as observations
from tests.test_replay_order import _replay


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _private(replay):
    result = {}
    for role in ('p1', 'p2'):
        species = [line.split('|')[3].split(',')[0] for line in replay['log'].splitlines()
                   if line.startswith('|poke|'+role+'|')]
        sheet = np.full((6, 15), -1, dtype=np.int64)
        for i, name in enumerate(species):
            sheet[i, 0] = TABLES.species_index(''.join(x for x in name.lower() if x.isalnum()))
            sheet[i, 3] = TABLES.move_index('protect')
        result[role] = complete_private_team(sheet, seed=7)[0]
    return result


def test_prefix_inputs_ignore_lead_future_and_opponent_private_sheet():
    replay = _replay()
    private = _private(replay)
    first = preview_arrays(replay, private)
    changed = copy.deepcopy(replay)
    changed['log'] = replay['log'].split('|start')[0] + '''|start
|switch|p1a: Bird|Corviknight|100/100
|switch|p2a: Chomp|Garchomp|100/100
|-ability|p2a: Chomp|Rough Skin
|-damage|p1a: Bird|1/100
|win|Bob'''
    later = preview_arrays(changed, private)
    for key in ('entities', 'entity_ids', 'field'):
        np.testing.assert_array_equal(first['preview_'+key], later['preview_'+key])
    assert not np.array_equal(first['preview_label'], later['preview_label'])
    private['p2'] = private['p2'].copy()
    private['p2'][:, 3:7] = TABLES.move_index('splash')
    other = preview_arrays(replay, private)
    for key in ('entities', 'entity_ids', 'field'):
        np.testing.assert_array_equal(first['preview_'+key][0], other['preview_'+key][0])
    assert not np.array_equal(first['preview_entity_ids'][1], other['preview_entity_ids'][1])
    assert (first['preview_entities'][..., L.hp] == 1).all()
    assert (first['preview_entities'][..., L.is_active] == 0).all()
    for col, unknown in [(1, TABLES.n_items),
                         *[(c, TABLES.n_moves) for c in range(3, 8)]]:
        assert (first['preview_entity_ids'][:, 6:, col] == unknown).all()
    from poke_env.data import GenData

    for row in range(2):
        opponent_species = first['preview_species'][1-row]
        for slot, species in enumerate(opponent_species):
            ability = first['preview_entity_ids'][row, 6+slot, 2]
            if ability != TABLES.n_abilities:
                # A species with only one ability has no hidden ability choice.
                choices = set(GenData.from_gen(9).pokedex[species]['abilities'].values())
                assert len(choices) == 1
                name = ''.join(x for x in choices.pop().lower() if x.isalnum())
                assert ability == TABLES.ability_index(name)
    assert not any(k.startswith('terminal') for k in first)


def test_public_identity_handles_form_and_illusion_without_guessing():
    replay = _replay()
    assert public_lead_labels(replay)[0] == {'p1': 0, 'p2': 0}
    form = copy.deepcopy(replay)
    form['log'] = form['log'].replace('|poke|p1|Garchomp|', '|poke|p1|Dudunsparce|').replace(
        '|switch|p1a: Chomp|Garchomp|', '|switch|p1a: Chomp|Dudunsparce-Three-Segment|')
    assert public_lead_labels(form)[0]['p1'] == 0
    illusion = copy.deepcopy(replay)
    illusion['log'] = illusion['log'].replace('|poke|p1|Corviknight|', '|poke|p1|Zoroark-Hisui|')
    assert public_lead_labels(illusion)[0]['p1'] == -1
    illusion['log'] = illusion['log'].replace('|turn|1', '|replace|p1a: Chomp|Zoroark-Hisui|100/100\n|turn|1')
    assert public_lead_labels(illusion)[0]['p1'] == 1
    late = copy.deepcopy(replay)
    late['log'] = late['log'].replace('|poke|p1|Corviknight|', '|poke|p1|Zoroark-Hisui|')
    late['log'] += '\n|replace|p1a: Bird|Zoroark-Hisui|100/100'
    assert public_lead_labels(late)[0]['p1'] == -1


def test_two_preview_records_only_first_imputation_and_ordinary_rows_unchanged(monkeypatch):
    replay = _replay()
    result = parse_replay_policy_versions(replay, replay_id='fixture', imputations=3, seed=4)
    preview = result['preview_arrays']
    assert len(preview['preview_label']) == 2
    assert preview['preview_imputation_id'].tolist() == [0, 0]
    assert preview['preview_role'].tolist() == ['p1', 'p2']
    assert preview['preview_valid'].all()
    for i, label in enumerate(preview['preview_label']):
        assert preview['preview_species'][i, label] in {'garchomp', 'dragapult'}
    expected = rows_to_arrays(result['rows'])
    monkeypatch.setattr('fog.training.replay_preview.preview_arrays', lambda *_: {})
    before = parse_replay_policy_versions(replay, replay_id='fixture', imputations=3, seed=4)
    actual = rows_to_arrays(before['rows'])
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)


def test_preview_logits_are_equivariant_with_noisy_weights(observations):
    cfg = load_config('configs/replay_preview_cpu.yaml')
    torch.manual_seed(11)
    agent = FogAgent.from_tables(cfg).eval()
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.add_(.02*torch.randn_like(parameter))
    batch = {k: observations[k].clone() for k in ('entities', 'entity_ids', 'field')}
    with torch.no_grad():
        original = agent.preview_logits(batch)
        for side in (0, 1):
            shuffled = {k:v.clone() for k,v in batch.items()}
            perm = torch.rand(64, 6).argsort(-1)
            for key in ('entities', 'entity_ids'):
                value = batch[key][:, 6*side:6*side+6]
                shuffled[key][:, 6*side:6*side+6] = value.gather(1,perm[..., None].expand(-1,-1,value.shape[-1]))
            shuffled['entities'][:, 6*side:6*side+6, L.party_slot:L.party_slot+6] = torch.eye(6)
            expected = original.gather(1,perm) if side == 0 else original
            torch.testing.assert_close(agent.preview_logits(shuffled), expected, atol=1e-5, rtol=0)
        shuffled = {k:v.clone() for k,v in batch.items()}
        perm = torch.rand(64,12,4).argsort(-1)
        shuffled['entity_ids'][..., 3:7] = batch['entity_ids'][..., 3:7].gather(-1,perm)
        for offset in (L.move_pp, L.epistemic+2):
            shuffled['entities'][..., offset:offset+4] = batch['entities'][..., offset:offset+4].gather(-1,perm)
        torch.testing.assert_close(agent.preview_logits(shuffled), original, atol=1e-5, rtol=0)
        # Labels and ordinary training-only targets cannot enter the preview path.
        altered = {**batch, 'preview_action':torch.arange(64)%6,
                   'terminal_hp_difference':torch.randn(64)}
        torch.testing.assert_close(agent.preview_logits(altered),original,atol=0,rtol=0)
        agent.backbone.order_blind_preview = False
        off = agent(observations, need_belief=False)
        agent.backbone.order_blind_preview = True
        on = agent(observations, need_belief=False)
        torch.testing.assert_close(off.log_probs,on.log_probs,atol=0,rtol=0)
        torch.testing.assert_close(off.value,on.value,atol=0,rtol=0)


def _manifest(tmp_path):
    converted = parse_replay_policy_versions(_replay(),replay_id='fixture',imputations=1,seed=4)
    shard = tmp_path/'fixture.npz'
    arrays = {**rows_to_arrays(converted['rows']), **converted['preview_arrays']}
    np.savez_compressed(shard,**arrays)
    manifest = tmp_path/'manifest.parquet'
    pl.DataFrame({'id':['fixture'],'shard':[str(shard)],'n_rows':[len(converted['rows'])]}).write_parquet(manifest)
    return manifest, shard


def test_preview_scope_gradients_dataset_filtering_and_native_resume(tmp_path, monkeypatch):
    manifest, shard = _manifest(tmp_path)
    ds = PreviewDataset(manifest)
    assert len(ds) == 2
    assert set(next(ds.batches(2))) == {'entities','entity_ids','field','action','preview_action'}
    cfg = load_config('configs/replay_preview_cpu.yaml')
    cfg['bc'].update(manifest=str(manifest),holdout_manifest=str(manifest),init_ckpt=None,
                     epochs=3,batch_size=2,torch_threads=1,lr=.003,lr_final=.0003)
    cfg['log']['dir'] = str(tmp_path/'full')
    agent = FogAgent.from_tables(cfg).eval()
    chosen = bc._select_trainable_parameters(agent, 'preview')
    total, metrics = bc.bc_loss(agent,next(ds.batches(2)),{**cfg['loss'],'_preview_only':True})
    total.backward()
    assert 'value' not in metrics and 'belief' not in metrics
    for module in (agent.preview,agent.backbone.embeds,agent.backbone.preview_context):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    allowed = {id(p) for p in chosen}
    assert all(p.grad is None for p in agent.parameters() if id(p) not in allowed)
    bc.train(cfg)
    resumed = copy.deepcopy(cfg)
    resumed['log']['dir'] = str(tmp_path/'resumed')
    with monkeypatch.context() as stop:
        _interrupt_after_epoch(stop,1)
        with pytest.raises(KeyboardInterrupt):
            bc.train(resumed)
    state = Path(resumed['log']['dir'])/'checkpoints/bc_state.pt'
    assert torch.load(state,weights_only=False)['epoch'] == 1
    torch.save({},state.parent/'bc_best.pt')
    bc.train(resumed)
    full = torch.load(tmp_path/'full/checkpoints/bc_state.pt',weights_only=False)
    after = torch.load(state,weights_only=False)
    for key in ('model','optimizer','scheduler','rng','best_model','best_epoch'):
        _assert_state_equal(full[key],after[key])
    with np.load(shard) as z:
        arrays = dict(z)
    arrays['preview_valid'][1] = False
    np.savez_compressed(shard,**arrays)
    assert len(PreviewDataset(manifest)) == 1
    arrays.pop('preview_valid')
    np.savez_compressed(shard,**arrays)
    with pytest.raises(ValueError, match='reconvert'):
        PreviewDataset(manifest)


def test_preview_only_rejects_value_selection_and_missing_coef(tmp_path):
    cfg = load_config('configs/replay_preview_cpu.yaml')
    cfg['log']['dir'] = str(tmp_path)
    cfg['bc']['save_best_value'] = True
    with pytest.raises(ValueError,match='save_best_value'):
        bc.train(cfg)
    cfg['bc']['save_best_value'] = False
    cfg['loss']['preview_coef'] = 0
    with pytest.raises(ValueError,match='positive'):
        bc.train(cfg)
    with pytest.raises(ValueError,match='order_blind'):
        FogAgent.from_tables({'model':{'order_blind_preview':True}})


def test_training_frequency_baseline_uses_identity_and_ignores_order():
    from fog.eval.replay_preview_report import frequency_predictions

    species = np.array([['garchomp','corviknight','dragapult','kingambit','greattusk','ironvaliant']])
    counts = {'garchomp':7,'corviknight':7,'kingambit':3}
    assert frequency_predictions(species,counts).tolist() == [1]
    perm = np.array([3,2,5,0,1,4])
    answer = frequency_predictions(species[:,perm],counts)[0]
    assert perm[answer] == 1
