"""Policy and belief inputs through the explicit encoder contexts.

The policy consumes private state, public state, and history. Belief prediction
adds the observer's projected private state to public context. Tests check that
these inputs affect outputs without pooling belief predictions into the policy.
"""
from __future__ import annotations

import dataclasses

import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.models.belief import BeliefOutput


def _agent(seed: int = 0) -> FogAgent:
    torch.manual_seed(seed)
    agent = FogAgent({"model": {}})
    for mod in agent.modules():  # deterministic eval forward
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    return agent.eval()


def _batch(b: int = 2, t: int = 6, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.full((b, 6, 15), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, 300, (b, 6), generator=g)  # opp species revealed
    legality = torch.zeros(b, 14, dtype=torch.bool)
    legality[:, [0, 1, 8]] = True
    return {
        "entities": torch.randn(b, 12, ENTITY_RAW_DIM, generator=g),
        "entity_ids": torch.randint(0, 10, (b, 12, N_ID_FIELDS), generator=g),
        "field": torch.randn(b, DIMS.field_dim, generator=g),
        "history": torch.randn(b, t, DIMS.turn_dim, generator=g),
        "active_idx": torch.tensor([[0, 6]] * b),
        "legality_mask": legality,
        "opp_revealed": {"tokens": tokens, "masks": None},
    }


def _legal_logits(out) -> torch.Tensor:
    """log_probs at the legal actions only (illegal are -inf and would spuriously 'match')."""
    return out.log_probs[:, [0, 1, 8]]


# -------------------------------------------------------------- policy depends on turn history
def test_policy_depends_on_history():
    """Perturbing only history changes policy through the explicit history encoder."""
    agent = _agent()
    base = _batch()
    perturbed = dict(base, history=base["history"] + 3.0)
    with torch.no_grad():
        a = agent(base)
        b = agent(perturbed)
    assert not torch.allclose(_legal_logits(a), _legal_logits(b), atol=1e-4)


# ------------------------------------------------------ policy responds to current entity state
def test_policy_depends_on_current_state():
    """A current-entity perturbation changes the action distribution.

    The three-encoder contracts separately prove that private-only changes enter the private vector
    while public-current changes enter the public vector.
    """
    agent = _agent()
    base = _batch()
    perturbed = dict(base, entities=base["entities"] + 3.0)
    with torch.no_grad():
        a = agent(base)
        b = agent(perturbed)
    assert not torch.allclose(_legal_logits(a), _legal_logits(b), atol=1e-4)


# ---------------------------------------------------------- signaling: my private state
def test_signaling_channel_my_hidden_attrs_change_action():
    """Varying MY OWN mons' hidden attributes (their embedded ids in slots 0-5, exactly what
    `fork_for_own_world` varies across own-worlds) must change the policy's action distribution.
    Without this the K per-world policies are identical, the posterior never moves, and joint search
    collapses to perfect-information search."""
    agent = _agent()
    base = _batch()
    variant = dict(base, entity_ids=base["entity_ids"].clone())
    # perturb only MY team (slots 0-5); leave the opponent slots 6-11 untouched
    variant["entity_ids"][:, 0:6, :] = (variant["entity_ids"][:, 0:6, :] + 5) % 10
    with torch.no_grad():
        a = agent(base)
        b = agent(variant)
    assert not torch.allclose(_legal_logits(a), _legal_logits(b), atol=1e-4)


# --------------------------------------- explicit contexts feed chain and policy consistently
def test_explicit_contexts_are_shared_by_chain_and_policy():
    """Explicit backbone contexts reproduce the agent's policy and belief outputs."""
    agent = _agent()
    batch = _batch()
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        public_context = agent.public_context(contexts)
        belief_context = agent.belief_context(contexts)
        out = agent(batch)
        lp, v = agent.policy_value(
            contexts.private_state, batch["legality_mask"], public_context
        )
        assert torch.equal(lp, out.log_probs)
        assert torch.equal(v, out.value)
        bel = agent.belief(belief_context, batch["opp_revealed"])
        for fam, logits in bel.logits.items():
            assert torch.equal(logits, out.belief.logits[fam])


# -------------------------------------------------------- no belief mean-pool on the policy path
def test_no_belief_vec_anywhere():
    """BeliefOutput has no pooled belief vector; route-loss metadata never enters the policy."""
    assert {f.name for f in dataclasses.fields(BeliefOutput)} == {
        "logits", "loss_valid", "loss_targets", "order", "context_count", "route_length",
    }
    agent = _agent()
    assert not hasattr(agent.policy_value, "use_belief_vec")
    assert not hasattr(agent, "use_belief_vec")
    out_board, out_dyn = agent.head_inputs(_batch())
    assert out_board.shape[-1] == agent.backbone.state_dim
    assert out_dyn.shape[-1] == agent.backbone.dyn_dim
