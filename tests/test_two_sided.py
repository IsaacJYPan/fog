"""Two-sided public-belief regret solver (fog/search/two_sided.py) — exact-game gates.

Every test here is a game small enough to know the answer exactly (PROJECT_DESIGN_PROPOSAL.md
§15.1): hand-computed posteriors and action values, matrix games with unique known mixed
equilibria, an exploitability gate computed by exact per-type best response, the bluff game whose
analytic equilibrium the one-sided loop already pins (ported as a degenerate two-sided case), and
the elimination/floor/never-empty rules of §7.2-7.4.
"""
from __future__ import annotations

import numpy as np
import pytest

from fog.search.two_sided import (
    TwoSidedSolution,
    behavior_update,
    exploitability,
    floored_strategy,
    pair_posteriors,
    two_sided_regret_solve,
    type_action_values,
)

RHO_2x2 = np.array([[0.4, 0.1], [0.2, 0.3]])


# ------------------------------------------------------------------------------ posterior algebra
def test_pair_posteriors_hand_computed():
    s1 = np.array([[0.8, 0.2], [0.4, 0.6]])
    s2 = np.array([[0.5, 0.5], [1.0, 0.0]])
    post = pair_posteriors(RHO_2x2, s1, s2)
    assert post.shape == (2, 2, 2, 2)
    # cell (a=0, b=0) by hand: rho[k,j] * s1[k,0] * s2[j,0]
    unnorm = np.array([[0.4 * 0.8 * 0.5, 0.1 * 0.8 * 1.0],
                       [0.2 * 0.4 * 0.5, 0.3 * 0.4 * 1.0]])
    np.testing.assert_allclose(post[0, 0], unnorm / unnorm.sum())
    # every played pair normalizes to one
    np.testing.assert_allclose(post[0, 0].sum(), 1.0)
    # a pair nobody plays (b=1 has zero prob under opponent type 1; type 0 plays it) is fine,
    # but a pair NOBODY plays is all-zero: kill action a for both our types
    s1_dead = np.array([[1.0, 0.0], [1.0, 0.0]])
    post = pair_posteriors(RHO_2x2, s1_dead, s2)
    np.testing.assert_allclose(post[1], 0.0)   # a=1 never played -> zero, not NaN


def test_pair_posteriors_eliminates_type_lacking_the_label():
    """§7.3 load-bearing property: a candidate type that cannot play an observed public action is
    eliminated by observing it."""
    avail1 = np.array([[True, True], [True, False]])   # type 1 lacks label a=1
    s1 = floored_strategy(np.array([[0.5, 0.5], [1.0, 0.0]]), avail1, floor=0.1)
    s2 = np.ones((1, 1))
    post = pair_posteriors(np.array([[0.5], [0.5]]), s1, s2)
    assert post[1, 0, 0, 0] == pytest.approx(1.0)   # observing a=1 -> surely type 0
    assert post[1, 0, 1, 0] == 0.0


def test_type_action_values_hand_computed():
    """Pins the conditional-belief weighting: q1 uses P(j|k) (rows of rho), q2 uses P(k|j)
    (columns), and q2 is the NEGATED payoff (zero-sum)."""
    s1 = np.array([[1.0, 0.0], [0.5, 0.5]])
    s2 = np.array([[0.5, 0.5], [1.0, 0.0]])
    k_, j_, a_, b_ = np.meshgrid(np.arange(2), np.arange(2), np.arange(2), np.arange(2),
                                 indexing="ij")
    u = (k_ + 2 * j_ + 3 * a_ + 4 * b_).astype(np.float64)
    q1, q2 = type_action_values(u, RHO_2x2, s1, s2)
    np.testing.assert_allclose(q1[0], [2.0, 5.0])          # 3a + 2.0
    np.testing.assert_allclose(q1[1], [3.0, 6.0])          # 3a + 3.0
    np.testing.assert_allclose(q2[0], [-5.0 / 6.0, -4 - 5.0 / 6.0])   # -(4b + 5/6)
    np.testing.assert_allclose(q2[1], [-3.875, -7.875])               # -(4b + 3.875)


def test_zero_mass_type_is_safe():
    """A publicly impossible type (zero rho row) must not produce NaN and must not influence the
    opponent's values."""
    rho = np.array([[0.5, 0.5], [0.0, 0.0]])
    avail = np.ones((2, 2), dtype=bool)
    u = np.zeros((2, 2, 2, 2))
    u[0] = 1.0    # only type 0 rows matter
    sol = two_sided_regret_solve(lambda a, b: u, rho, avail, avail, iters=64)
    assert np.isfinite(sol.sigma1).all() and np.isfinite(sol.sigma2).all()
    np.testing.assert_allclose(sol.sigma1.sum(axis=1), 1.0)


# ------------------------------------------------------------------- known matrix-game equilibria
def _matrix_game(m: np.ndarray):
    """K=J=1 game with fixed payoff matrix m[a, b]."""
    u = np.asarray(m, dtype=np.float64)[None, None, :, :]
    rho = np.ones((1, 1))
    avail1 = np.ones((1, u.shape[2]), dtype=bool)
    avail2 = np.ones((1, u.shape[3]), dtype=bool)
    return u, rho, avail1, avail2


def test_matching_pennies_reaches_the_mixed_equilibrium():
    u, rho, a1, a2 = _matrix_game([[1.0, -1.0], [-1.0, 1.0]])
    sol = two_sided_regret_solve(lambda x, y: u, rho, a1, a2, iters=4000)
    assert isinstance(sol, TwoSidedSolution)
    np.testing.assert_allclose(sol.sigma1[0], [0.5, 0.5], atol=0.02)
    np.testing.assert_allclose(sol.sigma2[0], [0.5, 0.5], atol=0.02)
    assert abs(sol.root_value) < 0.02
    assert exploitability(u, rho, sol.sigma1, sol.sigma2, a1, a2) < 0.02


def test_asymmetric_pennies_unique_equilibrium():
    """u = [[2,-1],[-1,1]]: unique mixed equilibrium p = q = (2/5, 3/5), game value 1/5."""
    u, rho, a1, a2 = _matrix_game([[2.0, -1.0], [-1.0, 1.0]])
    sol = two_sided_regret_solve(lambda x, y: u, rho, a1, a2, iters=6000)
    np.testing.assert_allclose(sol.sigma1[0], [0.4, 0.6], atol=0.02)
    np.testing.assert_allclose(sol.sigma2[0], [0.4, 0.6], atol=0.02)
    assert sol.root_value == pytest.approx(0.2, abs=0.02)
    assert exploitability(u, rho, sol.sigma1, sol.sigma2, a1, a2) < 0.02


def test_rock_paper_scissors_uniform():
    u, rho, a1, a2 = _matrix_game([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    sol = two_sided_regret_solve(lambda x, y: u, rho, a1, a2, iters=6000)
    np.testing.assert_allclose(sol.sigma1[0], [1 / 3] * 3, atol=0.05)
    np.testing.assert_allclose(sol.sigma2[0], [1 / 3] * 3, atol=0.05)
    assert exploitability(u, rho, sol.sigma1, sol.sigma2, a1, a2) < 0.03


# --------------------------------------------------------------- Bayesian games (private types)
def test_bayesian_game_exploitability_gate():
    """K=2 x J=2 fixed-payoff Bayesian game with a CORRELATED joint belief: exploitability of the
    averaged strategies passes a tight gate, and more iterations make it smaller (§15.1)."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(2, 2, 3, 3))
    rho = RHO_2x2
    avail1 = np.ones((2, 3), dtype=bool)
    avail2 = np.ones((2, 3), dtype=bool)
    short = two_sided_regret_solve(lambda a, b: u, rho, avail1, avail2, iters=50)
    long = two_sided_regret_solve(lambda a, b: u, rho, avail1, avail2, iters=8000)
    e_short = exploitability(u, rho, short.sigma1, short.sigma2, avail1, avail2)
    e_long = exploitability(u, rho, long.sigma1, long.sigma2, avail1, avail2)
    assert e_long < 0.02
    assert e_long < e_short


def test_information_restriction_strategies_cannot_see_the_other_type():
    """Sanity pin for §19.3: our solved strategy has shape [K, A] — one row per OUR type. If two
    of our types are identical in payoff and belief, their rows must coincide (nothing else could
    distinguish them), even though the OPPONENT's types differ wildly."""
    u = np.zeros((2, 2, 2, 2))
    u[:, 0] = [[1.0, -1.0], [-1.0, 1.0]]    # vs opponent type 0: pennies
    u[:, 1] = [[-1.0, 1.0], [1.0, -1.0]]    # vs opponent type 1: reversed pennies
    rho = np.full((2, 2), 0.25)              # both our types see the same opponent mix
    avail = np.ones((2, 2), dtype=bool)
    sol = two_sided_regret_solve(lambda a, b: u, rho, avail, avail, iters=2000)
    np.testing.assert_allclose(sol.sigma1[0], sol.sigma1[1], atol=1e-9)


# ------------------------------------------------------- posterior-dependent values (signaling)
def _bluff_values(rho):
    """The pinned bluff game (tests/test_joint_search.py) as a degenerate two-sided game: J=1,
    B=1, our K=2 types Strong/Weak, actions (attack, wait). Attack's value depends on the
    posterior the CURRENT strategies induce after observing 'attack'. Analytic equilibrium:
    Strong always attacks, Weak attacks 1/3, posterior on Strong after attack = 3/4."""
    def values_fn(s1, s2):
        post = pair_posteriors(rho, s1, s2)          # [A, B, K, J]
        p_s = post[0, 0, 0, 0]                       # P(Strong | attack observed)
        p_w = post[0, 0, 1, 0]
        u = np.zeros((2, 1, 2, 1))
        u[0, 0, 0, 0] = 0.8 - 1.0 * p_s              # Strong attack: worse as it identifies S
        u[1, 0, 0, 0] = 0.1 - 0.4 * p_w              # Weak bluff: gone once expected
        return u
    return values_fn


def test_bluff_equilibrium_reproduced_two_sided_degenerate():
    rho = np.array([[0.5], [0.5]])
    avail1 = np.ones((2, 2), dtype=bool)
    avail2 = np.ones((1, 1), dtype=bool)
    sol = two_sided_regret_solve(_bluff_values(rho), rho, avail1, avail2, iters=4000)
    assert sol.sigma1[0, 0] > 0.9, "Strong keeps attacking (the bluff protects it)"
    assert 0.28 < sol.sigma1[1, 0] < 0.40, "Weak bluffs ~1/3"
    post = pair_posteriors(rho, sol.sigma1, sol.sigma2)
    assert 0.70 < post[0, 0, 0, 0] < 0.80, "posterior on Strong after attack ~3/4"


def test_frozen_posterior_never_bluffs():
    """The uncoupled contrast (what K independent searches compute): with the posterior frozen at
    the uniform-strategy prior, bluffing is strictly bad and must vanish from the average."""
    rho = np.array([[0.5], [0.5]])
    frozen = pair_posteriors(rho, np.full((2, 2), 0.5), np.ones((1, 1)))

    def values_fn(s1, s2):
        u = np.zeros((2, 1, 2, 1))
        u[0, 0, 0, 0] = 0.8 - 1.0 * frozen[0, 0, 0, 0]
        u[1, 0, 0, 0] = 0.1 - 0.4 * frozen[0, 0, 1, 0]
        return u

    sol = two_sided_regret_solve(values_fn, rho, np.ones((2, 2), dtype=bool),
                                 np.ones((1, 1), dtype=bool), iters=2000)
    assert sol.sigma1[0, 0] > 0.98
    assert sol.sigma1[1, 0] < 0.02


# ----------------------------------------------------------------- floors, elimination, updates
def test_floor_never_survives_into_the_answer():
    """Settled decision 24: action a=1 is strictly dominated for every type; the averaged output
    must carry (essentially) no probability on it even though the in-solve play was floored."""
    u = np.zeros((1, 1, 2, 2))
    u[..., 0, :] = 1.0
    sol = two_sided_regret_solve(lambda a, b: u, np.ones((1, 1)),
                                 np.ones((1, 2), dtype=bool), np.ones((1, 2), dtype=bool),
                                 iters=500, floor=0.05)
    assert sol.sigma1[0, 1] < 0.01
    assert sol.converged, "a dominant-action game must trip the stability stop before the cap"
    assert sol.iterations < 500


def test_floored_strategy_rows_and_zeros():
    avail = np.array([[True, True, False], [True, True, True]])
    pi = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = floored_strategy(pi, avail, floor=0.05)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert (out[avail] >= 0.05 - 1e-12).all()
    assert (out[~avail] == 0).all()
    # a row with no legal action stays all-zero, never NaN
    none = floored_strategy(np.zeros((1, 3)), np.zeros((1, 3), dtype=bool), floor=0.05)
    np.testing.assert_array_equal(none, 0.0)


def test_behavior_update_floor_keeps_wrong_model_candidates_alive():
    """§7.2 safety rule 1: the model said 'type 1 never plays a=0'; type 1 must survive observing
    a=0 (shrunk, not destroyed), because behavior likelihoods are modeled, not mechanical."""
    rho = np.array([[0.5, 0.0], [0.5, 0.0]])
    avail1 = np.ones((2, 2), dtype=bool)
    sigma1 = np.array([[0.5, 0.5], [0.0, 1.0]])        # model: type 1 never plays a=0
    like1 = floored_strategy(sigma1, avail1, floor=0.02)[:, 0]
    rho_next, alive = behavior_update(rho, like1, np.ones(2))
    assert alive
    assert rho_next[1, 0] > 0.0, "floored likelihood must not eliminate"
    assert rho_next[1, 0] < rho_next[0, 0], "but it must shrink the weight"


def test_behavior_update_eliminates_on_missing_label_and_reports_death():
    """Elimination stays exact when the label is UNAVAILABLE (mechanical), and a fully
    contradicted table reports alive=False with the input unchanged (§7.4 never-empty guard)."""
    rho = np.array([[0.5], [0.5]])
    avail1 = np.array([[True, True], [True, False]])   # type 1 cannot play a=1
    sigma1 = floored_strategy(np.array([[0.5, 0.5], [1.0, 0.0]]), avail1, floor=0.02)
    rho_next, alive = behavior_update(rho, sigma1[:, 1], np.ones(1))
    assert alive
    assert rho_next[1, 0] == 0.0
    assert rho_next[0, 0] == pytest.approx(1.0)
    dead, alive = behavior_update(rho, np.zeros(2), np.ones(1))
    assert not alive
    np.testing.assert_array_equal(dead, rho)           # unchanged, caller must rejuvenate


def test_solver_is_deterministic():
    rho = np.array([[0.5], [0.5]])
    a1 = np.ones((2, 2), dtype=bool)
    a2 = np.ones((1, 1), dtype=bool)
    s_a = two_sided_regret_solve(_bluff_values(rho), rho, a1, a2, iters=300)
    s_b = two_sided_regret_solve(_bluff_values(rho), rho, a1, a2, iters=300)
    np.testing.assert_array_equal(s_a.sigma1, s_b.sigma1)
    np.testing.assert_array_equal(s_a.sigma2, s_b.sigma2)


def test_policy_prior_initialization_washes_out():
    """§10.2: initialization is not anchoring. A wildly wrong prior (always the dominated action)
    must not survive linear-weighted averaging."""
    u = np.zeros((1, 1, 2, 2))
    u[..., 0, :] = 1.0
    bad_init = np.array([[0.01, 0.99]])
    sol = two_sided_regret_solve(lambda a, b: u, np.ones((1, 1)),
                                 np.ones((1, 2), dtype=bool), np.ones((1, 2), dtype=bool),
                                 init1=bad_init, iters=500)
    assert sol.sigma1[0, 0] > 0.99
