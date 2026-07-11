"""Tests for the autoregressive belief model (§3.2), self-belief predictor (§3.3), and agent wiring."""
from __future__ import annotations

import math

import pytest
import torch

from fog.models.agent import FogAgent
from fog.models.belief import N_SLOTS, SLOT_ATTRS, BeliefModel
from fog.models.self_belief import (
    SelfBeliefPredictor,
    labels_from_tokens,
    still_hidden_from_tokens,
)
from fog.training.losses import belief_loss

SIZES = dict(n_species=10, n_items=6, n_abilities=5, n_moves=12, n_spreads=4, n_tera=19)
STATE_DIM = 32
D = 16
B = 3


def make_model(cls=BeliefModel) -> BeliefModel:
    torch.manual_seed(0)
    return cls(state_dim=STATE_DIM, d=D, **SIZES)


def make_inputs(reveal_all: bool = False, with_masks: bool = True):
    torch.manual_seed(1)
    s = torch.randn(B, STATE_DIM)
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    if reveal_all:
        highs = [SIZES["n_species"], SIZES["n_items"], SIZES["n_abilities"],
                 SIZES["n_moves"], SIZES["n_moves"], SIZES["n_moves"], SIZES["n_moves"],
                 SIZES["n_spreads"], SIZES["n_tera"]]
        for a, hi in enumerate(highs):
            tokens[:, :, a] = torch.randint(0, hi, (B, N_SLOTS))
        # make moves distinct within each slot
        for bi in range(B):
            for j in range(N_SLOTS):
                tokens[bi, j, 3:7] = torch.randperm(SIZES["n_moves"])[:4]
    else:
        # species always revealed (standard OU preview); one move revealed on slot 0
        tokens[:, :, 0] = torch.randint(0, SIZES["n_species"], (B, N_SLOTS))
        tokens[:, 0, 3] = 5
    masks = None
    if with_masks:
        masks = {
            "species": torch.ones(B, N_SLOTS, SIZES["n_species"], dtype=torch.bool),
            "item": torch.ones(B, N_SLOTS, SIZES["n_items"], dtype=torch.bool),
            "ability": torch.ones(B, N_SLOTS, SIZES["n_abilities"], dtype=torch.bool),
            "move": torch.ones(B, N_SLOTS, SIZES["n_moves"], dtype=torch.bool),
            "spread": torch.ones(B, N_SLOTS, SIZES["n_spreads"], dtype=torch.bool),
        }
        masks["item"][:, :, -2:] = False       # last two items illegal everywhere
        masks["move"][:, :, 0] = False         # move 0 not in any learnset
    return s, {"tokens": tokens, "masks": masks}


def test_forward_shapes():
    model = make_model()
    s, opp = make_inputs()
    out = model(s, opp)
    assert out.b_vec.shape == (B, D)
    assert out.logits["species"].shape == (B, 6, SIZES["n_species"])
    assert out.logits["item"].shape == (B, 6, SIZES["n_items"])
    assert out.logits["ability"].shape == (B, 6, SIZES["n_abilities"])
    assert out.logits["move"].shape == (B, 6, 4, SIZES["n_moves"])
    assert out.logits["spread"].shape == (B, 6, SIZES["n_spreads"])
    assert out.logits["tera"].shape == (B, 6, 19)
    for fam in ("species", "spread", "tera"):
        assert torch.isfinite(out.logits[fam]).any()


def test_legality_masks_applied():
    model = make_model()
    s, opp = make_inputs()
    out = model(s, opp)
    assert (out.logits["item"][:, :, -2:] == -math.inf).all()
    assert (out.logits["move"][..., 0] == -math.inf).all()
    assert torch.isfinite(out.logits["item"][:, :, 0]).all()


def test_move_distinctness_in_teacher_forcing():
    model = make_model()
    s, opp = make_inputs()
    out = model(s, opp)
    # slot 0 move0 revealed as 5 -> later move subslots of slot 0 must exclude 5
    for sub in (1, 2, 3):
        assert (out.logits["move"][:, 0, sub, 5] == -math.inf).all()
    # slot 1 has no revealed moves -> 5 stays legal there
    assert torch.isfinite(out.logits["move"][:, 1, 1, 5]).all()


def test_sample_team_shapes_legality_and_clamping():
    model = make_model()
    s, opp = make_inputs()
    k = 4
    g = torch.Generator().manual_seed(7)
    teams = model.sample_team(s, opp, k=k, generator=g)
    assert teams.shape == (B, k, 6, 9)
    assert (teams >= 0).all()
    # revealed attrs are clamped
    assert (teams[:, :, :, 0] == opp["tokens"][:, None, :, 0]).all()
    assert (teams[:, 0, 0, 3] == 5).all() and (teams[:, -1, 0, 3] == 5).all()
    # legality: illegal items/moves never sampled
    assert (teams[:, :, :, 1] < SIZES["n_items"] - 2).all()
    assert (teams[:, :, :, 3:7] != 0).all()
    # moves distinct within each slot
    mv = teams[..., 3:7]
    for bi in range(B):
        for ki in range(k):
            for j in range(6):
                assert len(set(mv[bi, ki, j].tolist())) == 4
    # ranges
    assert (teams[..., 2] < SIZES["n_abilities"]).all()
    assert (teams[..., 7] < SIZES["n_spreads"]).all()
    assert (teams[..., 8] < 19).all()


def test_sample_team_deterministic_with_seed():
    model = make_model()
    s, opp = make_inputs()
    t1 = model.sample_team(s, opp, k=3, generator=torch.Generator().manual_seed(11))
    t2 = model.sample_team(s, opp, k=3, generator=torch.Generator().manual_seed(11))
    assert (t1 == t2).all()


def test_belief_loss_backprops():
    model = make_model()
    s, opp = make_inputs()
    out = model(s, opp)
    targets = {
        "species": torch.randint(0, SIZES["n_species"], (B, 6)),
        "item": torch.randint(0, SIZES["n_items"] - 2, (B, 6)),
        "ability": torch.randint(0, SIZES["n_abilities"], (B, 6)),
        "move": torch.randint(1, SIZES["n_moves"], (B, 6, 4)),
        "spread": torch.randint(0, SIZES["n_spreads"], (B, 6)),
        "tera": torch.randint(0, 19, (B, 6)),
    }
    valid = {k: torch.ones_like(v, dtype=torch.bool) for k, v in targets.items()}
    # censor: move targets may collide with the distinctness mask -> only move0 valid
    valid["move"][:, :, 1:] = False
    loss = belief_loss(out.logits, targets, valid)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_leakage_ce_and_frozen_copy():
    model = make_model(SelfBeliefPredictor)
    s, my_view = make_inputs()
    truth_tokens = my_view["tokens"].clone()
    # fill full truth (I know my own team)
    s_full, full_view = make_inputs(reveal_all=True, with_masks=False)
    labels = labels_from_tokens(full_view["tokens"])
    my_view["labels"] = labels

    model.eval()  # eval mode: dropout off, so the frozen-copy comparison below is deterministic
    ce = model.leakage_ce(s, my_view)
    assert ce.shape == (B,)
    assert (ce > 0).all()  # plenty of hidden attributes

    # nothing hidden -> zero leakage signal
    all_known_view = {"tokens": full_view["tokens"], "masks": None, "labels": labels}
    ce0 = model.leakage_ce(s, all_known_view)
    assert (ce0 == 0).all()

    phi = model.frozen_copy()
    assert all(not p.requires_grad for p in phi.parameters())
    ce_phi = phi.leakage_ce(s, my_view)
    assert torch.allclose(ce_phi, ce.detach(), atol=1e-5)
    assert truth_tokens.equal(my_view["tokens"])  # no in-place surprises


def test_still_hidden_from_tokens():
    tokens = torch.full((2, 6, 9), -1, dtype=torch.long)
    tokens[:, :, 0] = 1  # species seen
    hidden = still_hidden_from_tokens(tokens)
    assert not hidden["species"].any()
    assert hidden["item"].all() and hidden["move"].all() and hidden["tera"].all()
    assert hidden["move"].shape == (2, 6, 4)


# ------------------------------------------------------------------------------------ agent wiring

def make_agent_batch(b: int = 2, t: int = 7):
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

    torch.manual_seed(2)
    tokens = torch.full((b, 6, 9), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, 320, (b, 6))
    legality = torch.zeros(b, 14, dtype=torch.bool)
    legality[:, 0] = True
    legality[:, 8] = True
    my_tokens = torch.zeros(b, 6, 9, dtype=torch.long)
    group = dict(
        entities=torch.randn(b, 12, ENTITY_RAW_DIM),
        entity_ids=torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long),
        field=torch.randn(b, DIMS.field_dim),
        history=torch.randn(b, t, DIMS.turn_dim),
        active_idx=torch.tensor([[0, 6]] * b),
    )
    batch = dict(
        **group,
        legality_mask=legality,
        opp_revealed={"tokens": tokens, "masks": None},
        opp_view={**group, "my_view": {"tokens": my_tokens.clone().fill_(-1), "masks": None,
                                       "labels": labels_from_tokens(my_tokens)}},
    )
    return batch


def test_agent_forward_shapes_and_budget():
    torch.manual_seed(3)
    agent = FogAgent()
    agent.assert_within_budget()
    batch = make_agent_batch()
    out = agent(batch)
    assert out.log_probs.shape == (2, 14)
    probs = out.log_probs.exp()
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)
    assert (probs[:, 1:8] == 0).all()  # illegal actions have zero probability
    assert out.value.shape == (2,)
    assert (out.value.abs() <= 1).all()
    assert out.belief is not None and out.self_belief is not None
    assert out.belief.b_vec.shape == (2, 128)


@pytest.mark.parametrize("aux_into_trunk", [True, False])
def test_aux_into_trunk_switch(aux_into_trunk):
    torch.manual_seed(4)
    agent = FogAgent({"model": {"aux_into_trunk": aux_into_trunk, "use_belief_vec": False}})
    batch = make_agent_batch()
    out = agent(batch)
    # backprop ONLY a belief loss; check whether backbone got gradients
    targets = {
        "species": torch.randint(0, 320, (2, 6)),
        "item": torch.randint(0, 64, (2, 6)),
        "ability": torch.randint(0, 160, (2, 6)),
        "move": torch.randint(0, 720, (2, 6, 4)),
        "spread": torch.randint(0, 17, (2, 6)),
        "tera": torch.randint(0, 19, (2, 6)),
    }
    valid = {k: torch.ones_like(v, dtype=torch.bool) for k, v in targets.items()}
    loss = belief_loss(out.belief.logits, targets, valid)
    loss.backward()
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in agent.backbone.parameters()
    )
    belief_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in agent.belief.parameters()
    )
    assert belief_has_grad
    assert backbone_has_grad == aux_into_trunk
