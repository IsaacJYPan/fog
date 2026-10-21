"""Belief consistency refresh: only the belief nets train, play is untouched,
and the whole collect->retrain loop runs on real engine self-play."""
from __future__ import annotations

import pytest
import torch

# ------------------------------------------------------------------------------- engine-backed
from fog.env.engine_battle import HAVE_ENGINE
from fog.models.agent import FogAgent
from tests.test_joint_search import P_ME, P_ME_VARIANT, P_OPP

engine = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

POOL = [P_ME, P_ME_VARIANT, P_OPP]


def test_refresh_requires_belief_nets():
    from fog.training.belief_refresh import refresh_beliefs

    agent = FogAgent({})
    agent.belief = None
    with pytest.raises(ValueError, match="belief"):
        refresh_beliefs(agent, POOL)


def test_refresh_rejects_unknown_scope(monkeypatch):
    from fog.training.belief_refresh import refresh_beliefs

    agent = FogAgent.from_tables({})
    with pytest.raises(ValueError, match="scope"):
        refresh_beliefs(agent, POOL, scope="policy")


@engine
def test_refresh_trains_only_beliefs_and_improves_train_ce():
    """End-to-end on real self-play: belief + self-belief move, policy/value/trunk/preview are
    byte-identical (play unchanged), labels exist, report is finite."""
    from fog.training.belief_refresh import refresh_beliefs

    torch.manual_seed(0)
    agent = FogAgent.from_tables({})

    def snapshot(module):
        return [p.detach().clone() for p in module.parameters()]

    frozen_before = {m: snapshot(getattr(agent, m))
                     for m in ("backbone", "policy_value", "preview")}
    belief_before = {m: snapshot(getattr(agent, m)) for m in ("belief", "self_belief")}

    report = refresh_beliefs(agent, POOL, games=3, holdout_games=2, epochs=2, lr=1e-3,
                             minibatch_size=32, engine_batch=2, seed=0)

    for m, before in frozen_before.items():
        for a, b in zip(before, snapshot(getattr(agent, m))):
            assert torch.equal(a, b), f"{m} must not move — play must be byte-identical"
    assert any(
        not torch.equal(a, b)
        for m, before in belief_before.items()
        for a, b in zip(before, snapshot(getattr(agent, m)))
    ), "belief nets must actually train"

    assert report["n_train"] > 0 and report["n_holdout"] > 0
    assert report["train_loss_last"] < report["train_loss_first"], (
        "two epochs on the same small batch must reduce the train loss"
    )
    for k in ("belief_ce_before", "belief_ce_after",
              "self_belief_ce_before", "self_belief_ce_after",
              "belief_chain_ce_before", "belief_chain_ce_after",
              "self_belief_chain_ce_before", "self_belief_chain_ce_after"):
        assert report[k] == report[k] and report[k] >= 0.0  # finite, non-NaN
    assert isinstance(report["improved"], bool)


@engine
def test_self_belief_scope_preserves_forward_belief():
    from fog.training.belief_refresh import refresh_beliefs

    torch.manual_seed(1)
    agent = FogAgent.from_tables({})
    belief_before = [p.detach().clone() for p in agent.belief.parameters()]
    self_before = [p.detach().clone() for p in agent.self_belief.parameters()]
    report = refresh_beliefs(
        agent, POOL, games=3, holdout_games=2, epochs=1, lr=1e-3,
        minibatch_size=32, engine_batch=2, seed=1, scope="self_belief",
    )
    assert all(torch.equal(a, b) for a, b in zip(belief_before, agent.belief.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(self_before, agent.self_belief.parameters()))
    assert report["scope"] == "self_belief"
