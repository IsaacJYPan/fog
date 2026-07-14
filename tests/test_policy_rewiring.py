"""Phase-1 policy-rewiring tests (BELIEF_POLICY_PLAN §7).

The V2 policy takes exactly three inputs and NO belief summary: (a) revealed/public info + (b) my own
private info, both carried by s_board, and (c) the dynamics vector c_dyn. These tests pin:
  * the policy responds to c_dyn (input (c)) and to s_board (inputs (a)+(b));
  * the SIGNALING CHANNEL (§1): varying MY hidden attributes changes the action distribution — the
    property joint search needs (without it every own-world policy is identical and search is inert);
  * c_dyn is literally the SAME tensor fed to the belief chain and the policy head;
  * no code path resurrects a belief mean-pool (b_vec) on the policy path.
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
    tokens = torch.full((b, 6, 9), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, 300, (b, 6), generator=g)  # opp species revealed
    legality = torch.zeros(b, 14, dtype=torch.bool)
    legality[:, [0, 1, 8]] = True
    return dict(
        entities=torch.randn(b, 12, ENTITY_RAW_DIM, generator=g),
        entity_ids=torch.randint(0, 10, (b, 12, N_ID_FIELDS), generator=g),
        field=torch.randn(b, DIMS.field_dim, generator=g),
        history=torch.randn(b, t, DIMS.turn_dim, generator=g),
        active_idx=torch.tensor([[0, 6]] * b),
        legality_mask=legality,
        opp_revealed={"tokens": tokens, "masks": None},
    )


def _legal_logits(out) -> torch.Tensor:
    """log_probs at the legal actions only (illegal are -inf and would spuriously 'match')."""
    return out.log_probs[:, [0, 1, 8]]


# --------------------------------------------------------------- (c): policy depends on c_dyn
def test_policy_depends_on_c_dyn():
    """History feeds ONLY c_dyn (dyn_head), never s_board. Perturbing just the history therefore
    isolates input (c): if the action distribution moves, the policy reads c_dyn."""
    agent = _agent()
    base = _batch()
    perturbed = dict(base, history=base["history"] + 3.0)
    with torch.no_grad():
        a = agent(base)
        b = agent(perturbed)
    assert not torch.allclose(_legal_logits(a), _legal_logits(b), atol=1e-4)


# --------------------------------------------------- (a)+(b): policy depends on s_board (entities)
def test_policy_depends_on_s_board():
    """Entity records feed ONLY s_board (the entity transformer + pooling), never c_dyn. Perturbing
    the entity features isolates inputs (a)+(b)."""
    agent = _agent()
    base = _batch()
    perturbed = dict(base, entities=base["entities"] + 3.0)
    with torch.no_grad():
        a = agent(base)
        b = agent(perturbed)
    assert not torch.allclose(_legal_logits(a), _legal_logits(b), atol=1e-4)


# ------------------------------------------------------ §1 signaling channel: my private state
def test_signaling_channel_my_hidden_attrs_change_action():
    """Varying MY OWN mons' hidden attributes (their embedded ids in slots 0-5, exactly what
    `fork_for_own_world` varies across own-worlds) must change the policy's action distribution.
    Without this the K per-world policies are identical, the posterior never moves, and joint search
    collapses to perfect-info search (§1)."""
    agent = _agent()
    base = _batch()
    variant = dict(base, entity_ids=base["entity_ids"].clone())
    # perturb only MY team (slots 0-5); leave the opponent slots 6-11 untouched
    variant["entity_ids"][:, 0:6, :] = (variant["entity_ids"][:, 0:6, :] + 5) % 10
    with torch.no_grad():
        a = agent(base)
        b = agent(variant)
    assert not torch.allclose(_legal_logits(a), _legal_logits(b), atol=1e-4)


# --------------------------------------------- c_dyn is the same tensor for the chain and the policy
def test_c_dyn_is_shared_by_chain_and_policy():
    """The backbone's single c_dyn is what BOTH the belief chain and the policy head consume (§3.1,
    'one vector, two consumers'). Feeding the backbone's own (s_board, c_dyn) back through each
    consumer reproduces the agent's outputs exactly."""
    agent = _agent()
    batch = _batch()
    with torch.no_grad():
        s_board, c_dyn = agent._backbone_pass(batch)
        out = agent(batch)
        # policy consumes (s_board, c_dyn)
        lp, v = agent.policy_value(s_board, batch["legality_mask"], c_dyn)
        assert torch.equal(lp, out.log_probs)
        assert torch.equal(v, out.value)
        # belief chain consumes the SAME c_dyn
        bel = agent.belief(c_dyn, batch["opp_revealed"])
        for fam, logits in bel.logits.items():
            assert torch.equal(logits, out.belief.logits[fam])


# -------------------------------------------------------- no belief mean-pool on the policy path
def test_no_belief_vec_anywhere():
    """The mean-pooled b_vec is gone (§3.6): BeliefOutput carries only logits, the head has no
    use_belief_vec/belief_vec, and head_inputs returns (s_board, c_dyn) with no belief forward."""
    assert {f.name for f in dataclasses.fields(BeliefOutput)} == {"logits"}
    agent = _agent()
    assert not hasattr(agent.policy_value, "use_belief_vec")
    assert not hasattr(agent, "use_belief_vec")
    out_board, out_dyn = agent.head_inputs(_batch())
    assert out_board.shape[-1] == agent.backbone.state_dim
    assert out_dyn.shape[-1] == agent.backbone.dyn_dim
