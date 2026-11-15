import torch

from fog.encoding.entities import N_ID_FIELDS
from fog.eval.replay_move_frequency_report import frequency_bucket, labelled_move_ids


def test_frequency_counts_move_identity_across_tera_slots_and_policy_masks():
    batch = {'action':torch.tensor([0,5,8,2]),
             'policy_valid':torch.tensor([True,True,True,False]),
             'active_idx':torch.tensor([[1,6],[4,6],[0,6],[3,6]]),
             'entity_ids':torch.zeros(4,12,N_ID_FIELDS,dtype=torch.long)}
    batch['entity_ids'][0,1,3] = 10
    batch['entity_ids'][1,4,4] = 10
    batch['entity_ids'][2,0,3] = 12
    batch['entity_ids'][3,3,5] = 13
    selected, ids = labelled_move_ids(batch,{})
    assert ids[selected].tolist() == [10,10]
    assert torch.bincount(ids[selected],minlength=14)[10] == 2
    assert frequency_bucket(torch.tensor([0,1,19,20])).tolist() == [0,1,1,2]
