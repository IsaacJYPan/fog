from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from fog.config import load_config
from fog.constants.tables import TABLES
from fog.encoding.action_features import ACTION_FEATURE_DIM, ACTION_FEATURE_NAMES
from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.models.mechanical_features import MECHANICAL_ACTION_FEATURE_NAMES, MechanicalMoveFeatures


def _batch():
    g = torch.Generator().manual_seed(184)
    return {'entities':torch.rand(4, 12, ENTITY_RAW_DIM, generator=g),
            'entity_ids':torch.zeros(4, 12, N_ID_FIELDS, dtype=torch.long),
            'field':torch.rand(4, DIMS.field_dim, generator=g),
            'history':torch.zeros(4, 1, DIMS.turn_dim),
            'history_len':torch.zeros(4,dtype=torch.long),
            'active_idx':torch.tensor([[0,6]]*4),
            'legality_mask':torch.ones(4,14,dtype=torch.bool),
            'action_features':torch.rand(4,14,ACTION_FEATURE_DIM,generator=g)}


def test_move_facts_come_from_tables_and_unknown_move_is_zero():
    features = MechanicalMoveFeatures(TABLES)
    earth = features(torch.tensor(TABLES.move_index('earthquake')))
    for name, expected in [('type:ground',1.),('category:physical',1.),
                           ('base_power',.4),('accuracy',1.),('priority',0.),('flag:protect',1.)]:
        torch.testing.assert_close(earth[features.names.index(name)],torch.tensor(expected))
    assert not features(torch.tensor(TABLES.n_moves)).any()
    moves = copy.deepcopy(TABLES.moves)
    moves['features']['earthquake']['new_public_boolean'] = True
    moves['features']['earthquake']['flags'].append('new_public_flag')
    extended = MechanicalMoveFeatures(SimpleNamespace(moves=moves))
    earth = extended(torch.tensor(TABLES.move_index('earthquake')))
    assert earth[extended.names.index('boolean:new_public_boolean')] == 1
    assert earth[extended.names.index('flag:new_public_flag')] == 1
    assert not any(name == 'values' for name in extended.state_dict())


def test_exact_candidate_columns_exclude_every_other_column_and_switches():
    expected = ('move_power','move_accuracy','expected_hits','physical','special','status_move',
                'same_type_bonus','type_effectiveness','physical_stat_ratio','special_stat_ratio',
                'estimated_damage')
    assert MECHANICAL_ACTION_FEATURE_NAMES == expected
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(27)
        agent = FogAgent.from_tables(load_config('configs/replay_mechanical_cpu.yaml')).eval()
        with torch.no_grad():
            for parameter in agent.parameters():
                parameter.add_(.02*torch.randn_like(parameter))
        batch = _batch()
        columns = [ACTION_FEATURE_NAMES.index(name) for name in expected]
        excluded = [i for i in range(ACTION_FEATURE_DIM) if i not in columns]
        descriptors = []
        def capture(module,args):
            descriptors.append(args[0].detach().clone())
        handle = agent.policy_value.action_query_input.register_forward_pre_hook(capture)
        with torch.no_grad():
            baseline = agent(batch, need_belief=False)
        handle.remove()
        torch.testing.assert_close(descriptors[0][:,:8,-11:],batch['action_features'][:,:8,columns])
        assert not descriptors[0][:,8:,-11:].any()
        changed = dict(batch)
        changed['action_features'] = batch['action_features'].clone()
        changed['action_features'][...,excluded] = float('nan')
        with torch.no_grad():
            actual = agent(changed,need_belief=False)
        torch.testing.assert_close(baseline.log_probs,actual.log_probs,atol=0,rtol=0)
        torch.testing.assert_close(baseline.value,actual.value,atol=0,rtol=0)
        features = batch['action_features'].clone().requires_grad_()
        output = agent({**batch,'action_features':features},need_belief=False)
        (-output.log_probs[:,0].mean()).backward()
        assert not features.grad[...,excluded].any()
        assert not features.grad[:,8:].any()
        assert features.grad[:,:8,columns].abs().sum() > 0
    finally:
        torch.set_num_threads(old_threads)
