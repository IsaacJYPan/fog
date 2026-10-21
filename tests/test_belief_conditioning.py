"""Mechanics tests for the belief-conditioning harness (fog/eval/belief_conditioning.py).

These pin the EXPERIMENT's correctness — the designed distribution, the reveal conventions, the
exact reference CEs, and that the eval runs end-to-end on an untrained model — NOT the training
outcome (the gate itself is a runnable experiment, deliberately not a unit test)."""
from __future__ import annotations

import math

import torch

from fog.eval.belief_conditioning import (
    DEP_COL,
    DEP_FLAT,
    DEP_SLOT,
    SRC_COL,
    SRC_FLAT,
    SRC_SLOT,
    _CaptureOrder,
    _drift_vs_best,
    _family_mean_sum,
    eval_rung2,
    make_rung1_teams,
    make_spec,
    reference_ce,
    rung1_floor,
    sample_truth,
    sample_view,
    targets_and_valid,
    verdict_rung2,
)
from fog.models.belief import N_ATTRS, N_SLOTS, SLOT_ATTRS, BeliefModel
from fog.training.losses import belief_loss


def _spec():
    return make_spec(seed=0, p_reveal_max=0.6)


def test_spec_rule_is_injective_and_species_fixed():
    spec = _spec()
    assert len(set(spec.rule_map.tolist())) == spec.sizes["n_items"]  # injective
    g = torch.Generator().manual_seed(1)
    t = sample_truth(spec, 64, g)
    assert (t[:, :, 0] == spec.species_team).all()                    # shared species line-up


def test_truth_rule_holds_and_moves_distinct():
    spec = _spec()
    g = torch.Generator().manual_seed(2)
    t = sample_truth(spec, 128, g)
    assert (t[:, DEP_SLOT, DEP_COL] == spec.rule_map[t[:, SRC_SLOT, SRC_COL]]).all()
    for bi in range(0, 128, 17):
        for s in range(N_SLOTS):
            assert len(set(t[bi, s, 3:7].tolist())) == 4


def test_view_conventions():
    spec = _spec()
    g = torch.Generator().manual_seed(3)
    t = sample_truth(spec, 64, g)
    v0 = sample_view(spec, t, g, rate=0.0)
    assert (v0[:, :, 0] == t[:, :, 0]).all()                          # species always revealed
    assert (v0[:, :, 1:] == -1).all()                                 # nothing else at rate 0
    v1 = sample_view(spec, t, g, rate=1.0)
    assert (v1 == t).all()                                            # everything at rate 1
    vm = sample_view(spec, t, g)                                      # mixture: first-k move clamp
    revealed_moves = (vm[:, :, 3:7] >= 0).to(torch.int8)
    assert (revealed_moves.diff(dim=-1) <= 0).all()                   # packed into earliest subslots
    hidden = vm < 0
    assert bool(hidden[:, :, 1:].any())                               # and something IS hidden


def test_targets_valid_are_hidden_only():
    spec = _spec()
    g = torch.Generator().manual_seed(4)
    t = sample_truth(spec, 32, g)
    v = sample_view(spec, t, g)
    targets, valid = targets_and_valid(t, v)
    assert not valid["species"].any()                                 # species never a chain target
    assert (valid["item"] == (v[:, :, 1] < 0)).all()
    assert (valid["move"] == (v[:, :, 3:7] < 0)).all()
    assert (targets["tera"] == t[:, :, 14]).all()


def test_reference_ce_exact_values():
    spec = _spec()
    g = torch.Generator().manual_seed(5)
    t = sample_truth(spec, 16, g)
    v = sample_view(spec, t, g)
    model = BeliefModel(state_dim=16, d=16, **spec.sizes).eval()
    revealed = model._revealed_mask(v)
    order = model._order(revealed, "random", torch.Generator().manual_seed(6))
    seq_pos = order.argsort(dim=1)
    marg, cond = reference_ce(spec, seq_pos)
    assert abs(marg[0, DEP_FLAT] - math.log(spec.sizes["n_items"])) < 1e-6  # dep marg = ln n_items
    assert (cond <= marg + 1e-6).all()                                # conditioning never hurts
    dep_cond = cond[:, DEP_FLAT]
    src_first = seq_pos[:, SRC_FLAT] < seq_pos[:, DEP_FLAT]
    assert (dep_cond[src_first] == 0).all()
    assert (dep_cond[~src_first] - math.log(spec.sizes["n_items"])).abs().max() < 1e-6
    # a move with all 3 slot-mates earlier: conditional = ln(n_moves - 3)
    fl = [s * len(SLOT_ATTRS) + c for s in range(N_SLOTS) for c in (3, 4, 5, 6)]
    assert cond[:, fl].min() >= math.log(spec.sizes["n_moves"] - 3) - 1e-6


def test_family_mean_sum_matches_belief_loss_convention():
    """On uniform logits, model CE == the analytic per-entry CE aggregated by _family_mean_sum."""
    spec = _spec()
    g = torch.Generator().manual_seed(7)
    t = sample_truth(spec, 8, g)
    v = sample_view(spec, t, g)
    targets, valid = targets_and_valid(t, v)
    logits = {
        f: torch.zeros(8, N_SLOTS, spec.sizes[k])
        for f, k in (("species", "n_species"), ("item", "n_items"), ("ability", "n_abilities"),
                     ("nature", "n_natures"), ("tera", "n_tera"))
    }
    logits["move"] = torch.zeros(8, N_SLOTS, 4, spec.sizes["n_moves"])
    logits["ev"] = torch.zeros(8, N_SLOTS, 6, spec.sizes["n_ev_units"])
    got = float(belief_loss(logits, targets, valid))
    per_entry = torch.zeros(8, N_ATTRS)
    for f, cols in (("item", (1,)), ("ability", (2,)), ("move", (3, 4, 5, 6)),
                    ("nature", (7,)), ("ev", (8, 9, 10, 11, 12, 13)), ("tera", (14,))):
        n = spec.sizes[{"item": "n_items", "ability": "n_abilities", "move": "n_moves",
                        "nature": "n_natures", "ev": "n_ev_units",
                        "tera": "n_tera"}[f]]
        for s in range(N_SLOTS):
            for c in cols:
                per_entry[:, s * len(SLOT_ATTRS) + c] = math.log(n)
    want = _family_mean_sum(per_entry, v.reshape(8, N_ATTRS) < 0)
    assert abs(got - want) < 1e-5


def test_capture_order_restores_and_records():
    spec = _spec()
    model = BeliefModel(state_dim=16, d=16, **spec.sizes).eval()
    g = torch.Generator().manual_seed(8)
    t = sample_truth(spec, 4, g)
    v = sample_view(spec, t, g)
    orig = model._order
    with _CaptureOrder(model) as cap:
        model(torch.zeros(4, 16), {"tokens": v, "masks": None}, teacher_values=t,
              tail_order="random", generator=torch.Generator().manual_seed(9))
    assert model._order == orig
    assert len(cap.orders) == 1 and cap.orders[0].shape == (4, N_ATTRS + 1)


def test_eval_rung2_smoke_on_untrained_model():
    """The full fair eval runs end-to-end and an untrained model shows NO conditioning: dep CE far
    from 0 everywhere, references sane."""
    spec = _spec()
    torch.manual_seed(10)
    model = BeliefModel(state_dim=16, d=16, **spec.sizes)
    g = torch.Generator().manual_seed(11)
    t = sample_truth(spec, 64, g)
    v = sample_view(spec, t, g)
    m = eval_rung2(model, spec, t, v, "cpu", n_orders=2, eval_seed=12, sample_b=8, sample_k=2,
                   probe_n_per_v=4)
    for key in ("total_ce", "dep_ce_src_revealed", "dep_ce_src_hidden_after",
                "dep_ce_src_hidden_before", "probe_acc_strict", "probe_acc_matched",
                "fallback_ce", "fallback_image_mass", "sample_fwd_consistency",
                "sample_rev_consistency"):
        assert math.isfinite(m[key]), key
    assert m["dep_n_src_revealed"] > 0 and m["dep_n_src_hidden_before"] > 0
    assert m["total_ce_conditional_ref"] <= m["total_ce_marginal_ref"]
    assert m["dep_ce_src_revealed"] > 0.5                             # untrained: no conditioning


def test_rung1_floor_bounds():
    spec = _spec()
    teams = make_rung1_teams(spec, seed=13)
    assert len({tuple(tm[:, 1].tolist()) for tm in teams}) == len(teams)
    g = torch.Generator().manual_seed(14)
    idx = torch.randint(0, len(teams), (32,), generator=g)
    truth = teams[idx]
    model = BeliefModel(state_dim=16, d=16, **spec.sizes).eval()
    teams_flat = teams.reshape(len(teams), N_ATTRS)
    truth_flat = truth.reshape(32, N_ATTRS)

    # everything revealed -> no hidden entries -> floor 0
    view_all = truth.clone()
    revealed = model._revealed_mask(view_all)
    order = model._order(revealed, "random", torch.Generator().manual_seed(15))
    floor = rung1_floor(teams_flat, truth_flat, view_all.reshape(32, N_ATTRS) < 0,
                        order.argsort(dim=1))
    assert floor == 0.0

    # species only -> positive floor, bounded by the mixture entropy spread over family means
    view_sp = sample_view(spec, truth, g, rate=0.0)
    revealed = model._revealed_mask(view_sp)
    order = model._order(revealed, "random", torch.Generator().manual_seed(16))
    floor = rung1_floor(teams_flat, truth_flat, view_sp.reshape(32, N_ATTRS) < 0,
                        order.argsort(dim=1))
    assert 0.0 < floor < math.log(len(teams)) * len(_family_nonempty(view_sp))


def _series(dep_ce_seq, probe_seq):
    """Minimal rung-2 series entries carrying the keys verdict_rung2 reads. Non-drift metrics are
    pinned to clean PASS values so only the regression logic under test decides no_drift_back."""
    out = []
    for dc, pa in zip(dep_ce_seq, probe_seq):
        out.append({
            "total_ce": 8.0, "total_ce_marginal_ref": 9.0, "dep_ce_src_revealed": dc,
            "dep_ce_src_hidden_after": dc, "dep_ce_src_hidden_before": 1.386,
            "probe_acc_strict": 1.0, "probe_acc_matched": pa, "fallback_ce": 1.386,
            "fallback_image_mass": 1.0, "fallback_max_prob": 0.2, "sample_fwd_consistency": 0.99,
        })
    return out


def test_no_drift_back_robust_to_oscillation():
    """A noisy-but-stable conditional (bounces ±0.1 around a flat mean, stable probe) must NOT be
    flagged as regression — the exact false positive a single-eval final-vs-best comparison gave and
    the reason the window de-noising exists."""
    refs = {"dep_marginal_ce": 1.386}
    osc = [0.30, 0.42, 0.28, 0.40, 0.26, 0.44, 0.29, 0.41, 0.27, 0.43]
    v = verdict_rung2(_series(osc, [1.0] * len(osc)), refs)
    assert v["checks"]["no_drift_back"]
    assert v["drift"]["probe_acc_matched_window"] == 0.0


def test_no_drift_back_catches_real_regression():
    """A genuinely regressing run — conditional CE climbing back toward the marginal and the probe
    decaying — MUST fail no_drift_back (the previous session's 'probe drifted back to a constant')."""
    refs = {"dep_marginal_ce": 1.386}
    rising = [0.20, 0.18, 0.22, 0.25, 0.40, 0.70, 1.10, 1.30, 1.38, 1.386]
    probe_decay = [1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.05]
    v = verdict_rung2(_series(rising, probe_decay), refs)
    assert not v["checks"]["no_drift_back"]


def test_drift_vs_best_window_direction():
    """_drift_vs_best: worst='max' (a low-is-good metric) reports how far the tail window rose above
    its best; worst='min' (an accuracy) reports how far it fell below its best."""
    lo = _series([0.2] * 4 + [0.9] * 3, [1.0] * 7)          # ends high -> positive drift up
    assert _drift_vs_best(lo, "dep_ce_src_revealed", worst="max") > 0.5
    flat = _series([0.3] * 7, [1.0] * 7)
    assert abs(_drift_vs_best(flat, "dep_ce_src_revealed", worst="max")) < 1e-9
    acc = _series([0.3] * 7, [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5])
    assert _drift_vs_best(acc, "probe_acc_matched", worst="min") < -0.2


def _family_nonempty(view: torch.Tensor) -> list[str]:
    hidden = view.reshape(view.shape[0], N_ATTRS) < 0
    fams = {
        "item": (1,), "ability": (2,), "move": (3, 4, 5, 6),
        "nature": (7,), "ev": (8, 9, 10, 11, 12, 13), "tera": (14,),
    }
    out = []
    for f, cols in fams.items():
        flats = [s * len(SLOT_ATTRS) + c for s in range(N_SLOTS) for c in cols]
        if hidden[:, flats].any():
            out.append(f)
    return out
