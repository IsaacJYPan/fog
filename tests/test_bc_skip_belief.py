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


def test_outcome_auxiliary_heads_survive_the_belief_skip():
    """Brief 13 trains with both: the skip must not drop the dense value targets."""
    from fog.config import load_config
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
    from fog.models.agent import FogAgent

    cfg = load_config("configs/replay_value_cpu.yaml")
    cfg["loss"].update(_ZERO, outcome_aux_coef=0.1)
    torch.manual_seed(17)
    agent = FogAgent.from_tables(cfg).eval()
    generator = torch.Generator().manual_seed(183)
    rows = 4
    batch = {
        "entities": torch.rand(rows, 12, ENTITY_RAW_DIM, generator=generator),
        "entity_ids": torch.zeros(rows, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.rand(rows, DIMS.field_dim, generator=generator),
        "history": torch.rand(rows, 4, DIMS.turn_dim, generator=generator),
        "history_ids": torch.zeros(rows, 4, DIMS.hist_id_fields, dtype=torch.long),
        "history_state": torch.zeros(rows, 4, DIMS.hist_state),
        "history_len": torch.full((rows,), 4),
        "active_idx": torch.tensor([[0, 6]] * rows),
        "legality_mask": torch.ones(rows, 14, dtype=torch.bool),
        "policy_valid": torch.ones(rows, dtype=torch.bool),
        "action": torch.arange(rows),
        "outcome": torch.tensor([1.0, -1.0, 0.0, 1.0]),
        "terminal_alive": torch.randint(2, (rows, 12), generator=generator).float(),
        "terminal_alive_mask": torch.ones(rows, 12, dtype=torch.bool),
        "terminal_hp_difference": torch.tensor([0.2, -0.2, 0.0, 0.1]),
        "terminal_valid": torch.ones(rows, dtype=torch.bool),
        "compute_belief_chain": True,
    }
    calls: list[dict] = []
    forward = agent.forward

    def recording_forward(batch, **kwargs):
        calls.append(kwargs)
        return forward(batch, **kwargs)

    agent.forward = recording_forward
    with_belief, _, parts_with = bc.bc_loss(agent, batch, cfg["loss"], return_components=True)
    skipping = {**cfg["loss"], "skip_unweighted_belief": True}
    total, metrics, parts = bc.bc_loss(agent, batch, skipping, return_components=True)
    assert calls == [{}, {"need_belief": False}]
    assert metrics["outcome_alive"] > 0 and metrics["outcome_hp"] > 0
    assert float(parts["outcome_aux"].detach()) > 0
    torch.testing.assert_close(parts["outcome_aux"], parts_with["outcome_aux"])
    torch.testing.assert_close(total, with_belief)
    total.backward()
    for head in (agent.outcome_alive, agent.outcome_hp):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
