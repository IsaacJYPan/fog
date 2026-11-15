"""`loss.skip_unweighted_belief` leaves out the belief passes only when nothing weights them."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fog.training import bc

_ZERO = {"belief_coef": 0.0, "belief_chain_coef": 0.0, "self_belief_coef": 0.0,
         "self_belief_chain_coef": 0.0}


class _Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(14))
        self.calls: list[dict] = []
        self.seen: list[bool] = []

    def forward(self, batch, **kwargs):
        self.calls.append(kwargs)
        self.seen.append(bool(batch.get("compute_belief_chain", False)))
        rows = batch["action"].shape[0]
        log_probs = torch.log_softmax(self.weight, dim=0).expand(rows, -1)
        return SimpleNamespace(
            log_probs=log_probs, value=self.weight[:1].expand(rows), belief=None,
            belief_chain=None, self_belief=None, shared_state=None,
        )


def _batch():
    return {"action": torch.zeros(3, dtype=torch.long), "outcome": torch.ones(3)}


@pytest.mark.parametrize(
    ("loss_cfg", "skipped"),
    [
        ({**_ZERO, "skip_unweighted_belief": True}, True),
        ({**_ZERO}, False),                                   # key off: today's behaviour
        ({**_ZERO, "skip_unweighted_belief": False}, False),
        ({**_ZERO, "belief_chain_coef": 0.05, "skip_unweighted_belief": True}, False),
        ({"skip_unweighted_belief": True, "belief_chain_coef": 0.0,
          "self_belief_coef": 0.0}, False),                   # belief_coef defaults to 1.0
    ],
)
def test_belief_passes_are_skipped_only_when_asked_and_unweighted(loss_cfg, skipped):
    agent = _Agent()
    batch = {**_batch(), "compute_belief_chain": True}        # as the holdout evaluation sets it
    total, metrics = bc.bc_loss(agent, batch, loss_cfg)
    assert agent.calls == ([{"need_belief": False}] if skipped else [{}])
    assert agent.seen[-1] is (not skipped)
    assert batch["compute_belief_chain"] is True              # the caller's batch is untouched
    assert bc.belief_forward_needed(loss_cfg) is (not skipped)
    assert torch.isfinite(total)
    assert metrics["belief"] == 0.0 and metrics["belief_chain"] == 0.0
