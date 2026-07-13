"""Mechanics tests for the reveal-set-conditioned any-order belief chain (BELIEF_PLAN §9, T1-T8).

These pin the properties the redesign exists to guarantee and that the old fixed-order chain could
not: every hidden conditional sees every reveal (T8), the chain is causal with no self-leak (T2),
the revealed prefix is order-free but slot-identity-aware (T3/T3b), revealed attrs are exact delta
rows (T4), and move distinctness follows the conditioning set, not the subslot index (T5).
"""
from __future__ import annotations

import math

import torch

from fog.models.belief import (
    N_ATTRS,
    N_SLOTS,
    SLOT_ATTRS,
    BeliefModel,
)

SIZES = dict(n_species=10, n_items=6, n_abilities=5, n_moves=12, n_spreads=4, n_tera=19)
STATE_DIM = 32
D = 16
B = 3

_ATTR_COLS = {"species": 0, "item": 1, "ability": 2, "spread": 7, "tera": 8}
_FAMILY_OF_COL = ("species", "item", "ability", "move", "move", "move", "move", "spread", "tera")


def make_model() -> BeliefModel:
    torch.manual_seed(0)
    return BeliefModel(state_dim=STATE_DIM, d=D, **SIZES).eval()


def logit_at(out, p: int) -> torch.Tensor:
    """The [B, n] logit vector for canonical attribute index p (0..53)."""
    slot, col = divmod(p, len(SLOT_ATTRS))
    fam = _FAMILY_OF_COL[col]
    if fam == "move":
        return out.logits["move"][:, slot, col - 3]
    return out.logits[fam][:, slot]


def full_truth_grid() -> torch.Tensor:
    """A [B, 6, 9] grid of valid, in-range values, with distinct moves per slot (aligned already)."""
    torch.manual_seed(4)
    g = torch.zeros(B, N_SLOTS, len(SLOT_ATTRS), dtype=torch.long)
    highs = [SIZES["n_species"], SIZES["n_items"], SIZES["n_abilities"], 0, 0, 0, 0,
             SIZES["n_spreads"], SIZES["n_tera"]]
    for col, hi in enumerate(highs):
        if col in (3, 4, 5, 6):
            continue
        g[:, :, col] = torch.randint(0, hi, (B, N_SLOTS))
    for bi in range(B):
        for j in range(N_SLOTS):
            g[bi, j, 3:7] = torch.randperm(SIZES["n_moves"])[:4]
    return g


def hidden_input(reveal_cols=()):
    """Observation tokens with only `reveal_cols` (list of (slot, col)) revealed off the truth grid."""
    truth = full_truth_grid()
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    for slot, col in reveal_cols:
        tokens[:, slot, col] = truth[:, slot, col]
    s = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(1))
    return s, {"tokens": tokens, "masks": None}, truth


# ---------------------------------------------------------------------------- T1: index-map reference
def test_t1_read_scatter_matches_reference_loop():
    """The vectorized gather/mask/scatter equals a slow per-example Python construction of the same
    physical layout, decode, and shifted read."""
    model = make_model()
    torch.manual_seed(2)
    # a mixed observation: species revealed everywhere, a couple of extra reveals
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, SIZES["n_species"], (B, N_SLOTS))
    tokens[:, 1, 8] = 3   # slot1 tera revealed
    tokens[:, 2, 1] = 2   # slot2 item revealed
    s = torch.randn(B, STATE_DIM)
    opp = {"tokens": tokens, "masks": None}

    out = model(s, opp)  # pass A, canonical

    # reference: build each example's sequence explicitly and decode one row at a time
    revealed = model._revealed_mask(tokens)          # [B,54]
    fed = model._fed_values(tokens, None)
    for bi in range(B):
        rev_idx = [p for p in range(N_ATTRS) if bool(revealed[bi, p])]
        hid_idx = [p for p in range(N_ATTRS) if not bool(revealed[bi, p])]
        content_order = rev_idx + [N_ATTRS] + hid_idx   # prefix | START | tail (canonical)
        # content embeddings for this row
        content = model._content_embeddings(fed[bi : bi + 1])[0]         # [55, d]
        phys = torch.stack([content[c] for c in content_order], dim=0).unsqueeze(0)  # [1,55,d]
        rc = torch.tensor([len(rev_idx)])
        mask = model._attn_mask(rc, s.device).repeat_interleave(model.nhead, dim=0)
        memory = model.s_proj(s[bi : bi + 1]).unsqueeze(1)
        hidden = model.decoder(phys, memory, tgt_mask=mask)[0]           # [55, d]
        # read each hidden attr at the position before its own token
        phys_of = {c: i for i, c in enumerate(content_order)}
        for p in hid_idx:
            read = hidden[phys_of[p] - 1]
            fam = _FAMILY_OF_COL[p % len(SLOT_ATTRS)]
            ref_logit = model.heads[fam](read)
            got = logit_at(out, p)[bi]
            finite = torch.isfinite(got)
            assert torch.allclose(got[finite], ref_logit[finite], atol=1e-5)


# ------------------------------------------------------------- T2: causality + no self-leak (pass B)
def test_t2_causal_no_self_leak():
    """All hidden, canonical teacher-forced tail. Perturbing the fed value of tail rank q changes only
    strictly-later ranks; rank q itself and all earlier ranks are unchanged (no self-leak)."""
    model = make_model()
    s, opp, truth = hidden_input()                    # nothing revealed -> all 54 are tail
    q = 28                                            # slot3 item (a non-move attr: no distinctness)
    base = model(s, opp, teacher_values=truth, tail_order="canonical")

    pert = truth.clone()
    pert[:, 3, 1] = (truth[:, 3, 1] + 1) % SIZES["n_items"]
    out = model(s, opp, teacher_values=pert, tail_order="canonical")

    for p in (0, 10, 27, q):                          # rank q and everything earlier: identical
        assert torch.allclose(logit_at(base, p), logit_at(out, p), atol=1e-6), p
    assert not torch.allclose(logit_at(base, 53), logit_at(out, 53))  # a later rank moves


def test_t2_prefix_perturb_reaches_tail():
    """Perturbing a REVEALED (prefix) value may change any tail conditional -- the core requirement
    that hidden predictions condition on all reveals."""
    model = make_model()
    truth = full_truth_grid()
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    tokens[:, :, 0] = truth[:, :, 0]                  # species revealed (prefix)
    s = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(1))
    base = model(s, {"tokens": tokens, "masks": None})

    tokens2 = tokens.clone()
    tokens2[:, 0, 0] = (tokens[:, 0, 0] + 1) % SIZES["n_species"]
    out = model(s, {"tokens": tokens2, "masks": None})
    # slot0 item (a hidden tail attr) sees the changed slot0 species in the prefix
    assert not torch.allclose(logit_at(base, 1), logit_at(out, 1))


# ------------------------------------------------------ T3 / T3b: prefix set order-free, slot-aware
def test_t3_prefix_order_invariance():
    """Physically permuting the revealed prefix leaves every tail conditional unchanged (bidirectional
    prefix attention + identity-not-position embeddings)."""
    model = make_model()
    truth = full_truth_grid()
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    for slot in range(N_SLOTS):
        tokens[:, slot, 0] = truth[:, slot, 0]        # species (6 revealed prefix tokens)
    tokens[:, 0, 8] = truth[:, 0, 8]                  # + a revealed tera
    s = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(1))
    opp = {"tokens": tokens, "masks": None}
    base = model(s, opp)

    orig_order = model._order

    def shuffled_order(revealed, tail_order, generator):
        order = orig_order(revealed, tail_order, generator)              # [B,55]
        for bi in range(B):
            rc = int(revealed[bi].sum())
            perm = torch.randperm(rc)                                    # shuffle prefix block only
            order[bi, :rc] = order[bi, :rc][perm]
        return order

    model._order = shuffled_order
    try:
        out = model(s, opp)
    finally:
        model._order = orig_order

    for p in range(N_ATTRS):
        a, b = logit_at(base, p), logit_at(out, p)
        fin = torch.isfinite(a) & torch.isfinite(b)
        assert torch.allclose(a[fin], b[fin], atol=1e-5), p


def test_t3b_slot_identity_is_live():
    """Swapping two same-family revealed values ACROSS slots (identical multiset, different slot
    assignment) changes tail conditionals -- reveal tokens carry slot identity, not just value."""
    model = make_model()
    truth = full_truth_grid()
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    tokens[:, :, 0] = truth[:, :, 0]                  # species everywhere
    tokens[:, 0, 8] = 3                               # slot0 tera = 3
    tokens[:, 1, 8] = 7                               # slot1 tera = 7
    s = torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(1))
    base = model(s, {"tokens": tokens, "masks": None})

    swapped = tokens.clone()
    swapped[:, 0, 8] = 7                              # same {3,7} set, swapped between slots
    swapped[:, 1, 8] = 3
    out = model(s, {"tokens": swapped, "masks": None})
    # slot0 item conditional must react to slot0's tera identity changing
    assert not torch.allclose(logit_at(base, 1), logit_at(out, 1))
    assert model.attr_pos.num_embeddings == N_ATTRS  # identity table stays 54-way


# ---------------------------------------------------------------------------------- T4: delta-fill
def test_t4_revealed_attrs_are_delta_rows():
    model = make_model()
    truth = full_truth_grid()
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    tokens[:, :, 0] = truth[:, :, 0]                  # species revealed
    tokens[:, 2, 1] = truth[:, 2, 1]                  # slot2 item revealed
    out = model(torch.randn(B, STATE_DIM, generator=torch.Generator().manual_seed(9)),
                {"tokens": tokens, "masks": None})
    sp = out.logits["species"]                        # [B,6,n]
    for bi in range(B):
        for slot in range(N_SLOTS):
            v = int(tokens[bi, slot, 0])
            row = sp[bi, slot]
            assert row[v] == 0.0
            assert (row[torch.arange(SIZES["n_species"]) != v] == -math.inf).all()
    it = out.logits["item"][:, 2]                     # slot2 item is a delta row
    for bi in range(B):
        v = int(tokens[bi, 2, 1])
        assert it[bi, v] == 0.0 and torch.isinf(it[bi]).sum() == SIZES["n_items"] - 1


# --------------------------------------------------------------------------------- T5: distinctness
def test_t5_distinctness_follows_conditioning_set():
    """A revealed move (always in the prefix) is excluded from every hidden move subslot of its slot;
    aligned teacher-forced targets never land on the -inf distinctness mask."""
    model = make_model()
    truth = full_truth_grid()                         # distinct moves per slot (the real invariant)
    tokens = torch.full((B, N_SLOTS, len(SLOT_ATTRS)), -1, dtype=torch.long)
    tokens[:, :, 0] = truth[:, :, 0]
    tokens[:, 0, 3] = truth[:, 0, 3]                  # slot0 move0 revealed (prefix), per-row truth
    s = torch.randn(B, STATE_DIM)
    opp = {"tokens": tokens, "masks": None}

    out = model(s, opp)                               # pass A marginals
    for bi in range(B):
        rev = int(truth[bi, 0, 3])
        for sub in (1, 2, 3):                          # revealed move excluded from later subslots
            assert out.logits["move"][bi, 0, sub, rev] == -math.inf
        assert torch.isfinite(out.logits["move"][bi, 1, 1, rev])  # slot1 has no such reveal

    # pass B: the aligned true target for every hidden move subslot is finite (never on -inf)
    chain = model(s, opp, teacher_values=truth, tail_order="random",
                  generator=torch.Generator().manual_seed(5))
    for bi in range(B):
        for slot in range(N_SLOTS):
            for sub in range(4):
                col = 3 + sub
                if int(tokens[bi, slot, col]) >= 0:
                    continue                          # revealed -> delta row, not a chain target
                tgt = int(truth[bi, slot, col])
                assert torch.isfinite(chain.logits["move"][bi, slot, sub, tgt]), (bi, slot, sub)


# ---------------------------------------------------------------- T8: a late reveal sharpens an early
# ---------------------------------------------------- T10: two passes leave the policy path unchanged
def test_t10_pass_b_does_not_touch_policy():
    """With use_belief_vec, the policy/value read pass A's b_vec; running the random-order pass-B
    chain (training + targets) must not change log_probs or value -- so PPO's epoch-0 ratio stays 1."""
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
    from fog.models.agent import FogAgent

    torch.manual_seed(0)
    agent = FogAgent({"model": {"use_belief_vec": True}})
    for mod in agent.modules():                       # deterministic train-mode forward
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.dropout = 0.0                          # functional dropout (not a Dropout module)
    agent.train()

    b = 2
    tokens = torch.full((b, 6, 9), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, 320, (b, 6))   # species revealed
    legality = torch.zeros(b, 14, dtype=torch.bool)
    legality[:, 0] = True
    legality[:, 8] = True
    base = dict(
        entities=torch.randn(b, 12, ENTITY_RAW_DIM),
        entity_ids=torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long),
        field=torch.randn(b, DIMS.field_dim),
        history=torch.randn(b, 5, DIMS.turn_dim),
        active_idx=torch.tensor([[0, 6]] * b),
        legality_mask=legality,
        opp_revealed={"tokens": tokens, "masks": None},
    )
    targets = {
        "species": tokens[:, :, 0].clone(),
        "item": torch.randint(0, 64, (b, 6)),
        "ability": torch.randint(0, 160, (b, 6)),
        "move": torch.stack([torch.stack([torch.randperm(720)[:4] for _ in range(6)])
                             for _ in range(b)]),
        "spread": torch.randint(0, 17, (b, 6)),
        "tera": torch.randint(0, 19, (b, 6)),
    }

    out_plain = agent(dict(base))                     # no targets -> no pass B
    out_chain = agent(dict(base, belief_targets=targets))  # pass B runs
    assert out_plain.belief_chain is None and out_chain.belief_chain is not None
    assert torch.allclose(out_plain.log_probs, out_chain.log_probs, atol=1e-6)
    assert torch.allclose(out_plain.value, out_chain.value, atol=1e-6)


def test_t8_late_reveal_conditions_early_attr():
    """Revealing tera (canonical LAST attr in a slot) shifts the item marginal (canonical 2nd) -- the
    fixed-order chain could not do this; the prefix-LM does it structurally."""
    model = make_model()
    s, opp_no_tera, truth = hidden_input(reveal_cols=[(j, 0) for j in range(N_SLOTS)])  # species only
    tokens_tera = opp_no_tera["tokens"].clone()
    tokens_tera[:, 0, 8] = truth[:, 0, 8]             # additionally reveal slot0 tera
    a = model(s, opp_no_tera)
    b = model(s, {"tokens": tokens_tera, "masks": None})
    assert not torch.allclose(logit_at(a, 1), logit_at(b, 1))  # slot0 item marginal moved
