from types import SimpleNamespace

import torch

from fog.scripts.check_policy_contract import check


class Policy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, batch):
        return SimpleNamespace(log_probs=self.logits.log_softmax(-1).expand(len(batch['actions']), -1))


def test_gate_accepts_matching_actor_and_rejects_changed_policy_without_updating():
    model = Policy(torch.tensor([0., 1., float('-inf')]))
    old = model.logits.detach().log_softmax(-1).expand(40, -1)
    batch = dict(actions=torch.zeros(40, dtype=torch.long), behavior_log_probs=old.clone(),
                 old_log_probs=old[:, 0].clone())
    assert check(model, batch, device='cpu', temperature=1., bonus=0.)['passed']
    with torch.no_grad():
        model.logits[0] += .1
    result = check(model, batch, device='cpu', temperature=1., bonus=0.)
    assert not result['passed'] and result['weights_unchanged']
    assert model.logits.grad is None
