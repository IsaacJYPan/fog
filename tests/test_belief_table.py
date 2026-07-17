"""JointBeliefTable (fog/search/belief_table.py, PROPOSAL_IMPL_PLAN Phase D): hand-computed
Bayes chains (the proposal's §5.1 Specs/Scarf example made exact), log-space equals direct
multiplication, floored behavior likelihoods never eliminate, label-lacking candidates ARE
eliminated, mechanical zeroing, the never-empty refusal, and effective-sample-size readouts."""
from __future__ import annotations

import numpy as np
import pytest

from fog.search.belief_table import JointBeliefTable


def _uniform(k: int, j: int) -> JointBeliefTable:
    return JointBeliefTable(np.zeros((k, j)))


def test_specs_scarf_protect_chain_hand_computed():
    """Proposal §5.1, strategy Y: the Specs version Protects 90%, the Scarf version 10%. Two
    observed Protects (floor disabled for the exact check) must give the exact posterior
    [0.81, 0.01] normalized — the behavioral update the four summary features could never carry."""
    t = _uniform(2, 1)                                   # K = {Specs, Scarf}, J trivial
    sigma1 = np.array([[0.9, 0.1], [0.1, 0.9]])          # columns: [protect, attack]
    avail1 = np.ones((2, 2), dtype=bool)
    for _ in range(2):
        assert t.behavior_step(sigma1, avail1, 0, None, None, None, floor=0.0)
    p_own, _ = t.margins()
    np.testing.assert_allclose(p_own, [0.81 / 0.82, 0.01 / 0.82])
    assert t.behavior_steps == 2


def test_two_sided_step_multiplies_both_likelihoods():
    t = _uniform(2, 2)
    s1 = np.array([[0.8, 0.2], [0.4, 0.6]])
    s2 = np.array([[0.5, 0.5], [0.9, 0.1]])
    avail = np.ones((2, 2), dtype=bool)
    assert t.behavior_step(s1, avail, 0, s2, avail, 1, floor=0.0)
    want = np.outer([0.8, 0.4], [0.5, 0.1])
    np.testing.assert_allclose(t.normalized(), want / want.sum())


def test_log_space_matches_direct_product_over_many_steps():
    rng = np.random.default_rng(3)
    t = _uniform(3, 4)
    direct = np.ones((3, 4))
    avail1 = np.ones((3, 2), dtype=bool)
    avail2 = np.ones((4, 2), dtype=bool)
    for _ in range(50):
        s1 = rng.dirichlet([1.0, 1.0], size=3)
        s2 = rng.dirichlet([1.0, 1.0], size=4)
        col1, col2 = int(rng.integers(2)), int(rng.integers(2))
        assert t.behavior_step(s1, avail1, col1, s2, avail2, col2, floor=0.0)
        direct = direct * np.outer(s1[:, col1], s2[:, col2])
    np.testing.assert_allclose(t.normalized(), direct / direct.sum(), rtol=1e-9)


def test_floor_keeps_wrong_model_candidates_alive_but_missing_labels_eliminate():
    """§7.2: 'the model says never' shrinks; 'the world lacks the action' eliminates."""
    t = _uniform(2, 1)
    sigma1 = np.array([[0.5, 0.5], [0.0, 1.0]])          # model: candidate 1 never plays col 0
    avail_all = np.ones((2, 2), dtype=bool)
    assert t.behavior_step(sigma1, avail_all, 0, None, None, None, floor=0.02)
    rho = t.normalized()
    assert rho[1, 0] > 0.0 and rho[1, 0] < rho[0, 0]

    t2 = _uniform(2, 1)
    avail_missing = np.array([[True, True], [False, True]])   # candidate 1 LACKS the label
    assert t2.behavior_step(sigma1, avail_missing, 0, None, None, None, floor=0.02)
    rho2 = t2.normalized()
    assert rho2[1, 0] == 0.0
    assert rho2[0, 0] == pytest.approx(1.0)


def test_forced_single_choice_carries_no_behavior_but_mechanics_still_apply():
    """§14: a forced action has no behavioral likelihood (side passed as None), yet the revealed
    switch identity can still mechanically eliminate candidates."""
    t = _uniform(2, 3)
    before = t.normalized().copy()
    assert t.behavior_step(None, None, None, None, None, None)
    np.testing.assert_allclose(t.normalized(), before)
    assert t.mechanical_step(keep_opp=np.array([True, False, True]))
    _, p_opp = t.margins()
    np.testing.assert_allclose(p_opp, [0.5, 0.0, 0.5])
    assert not t.needs_rejuvenation()


def test_never_empty_refusal_and_collapse_flag():
    """Invariant 19.18: a step that would kill every cell is refused loudly — the table keeps
    its previous state and reports collapse for the caller's rejuvenation."""
    t = _uniform(2, 2)
    before = t.normalized().copy()
    assert not t.mechanical_step(keep_own=np.array([False, False]))
    np.testing.assert_allclose(t.normalized(), before)
    assert t.collapsed and t.refused_steps == 1
    assert t.needs_rejuvenation()
    # the table remains usable after a refusal
    assert t.mechanical_step(keep_own=np.array([True, False]))


def test_effective_sample_sizes_and_open_space_threshold():
    t = _uniform(4, 4)
    assert t.effective_sample_sizes() == (pytest.approx(4.0), pytest.approx(4.0))
    assert not t.needs_rejuvenation(min_ess=1.5)
    assert t.mechanical_step(keep_opp=np.array([True, False, False, False]))
    _, ess_opp = t.effective_sample_sizes()
    assert ess_opp == pytest.approx(1.0)
    assert t.needs_rejuvenation(min_ess=1.5), "open-space policy: diversity died"
    assert not t.needs_rejuvenation(), "exact-pool default: concentration is success"


def test_prior_construction():
    t = JointBeliefTable.from_margins(np.log([0.75, 0.25]), np.log([0.5, 0.5]))
    np.testing.assert_allclose(t.normalized(),
                               np.outer([0.75, 0.25], [0.5, 0.5]))
    with pytest.raises(ValueError, match="support"):
        JointBeliefTable(np.full((2, 2), -np.inf))
