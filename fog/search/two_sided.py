"""Two-sided public-belief regret solver.

`rho[K, J]` is the joint belief over our and the opponent's candidate teams.
`sigma1[K, A]` and `sigma2[J, B]` are their type-conditioned strategies;
`avail1` and `avail2` mask illegal public-action labels for each type.

`values_fn(sigma1, sigma2)` returns our continuation payoffs [K, J, A, B].
It receives current strategies because leaf values may depend on their induced
posterior. The opponent's payoff is the negation. Unavailable actions are ignored.
Each strategy depends only on its own private type, averaging the other player's
types using the conditional distribution derived from rho.

Regret matching optionally clamps accumulated regret at zero (`plus=True`).
The returned strategy averages iterations with weight t + 1. An annealed
exploration floor stabilizes the strategies used during solving, but is excluded
from the accumulated average. Convergence uses total-variation changes between
average-strategy checkpoints, subject to an iteration cap.

This module contains no engine or network calls. Grid construction, leaf
valuation, and persistent belief tracking live in the surrounding search modules.
See tests/test_two_sided.py for checks on exact games."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


# ------------------------------------------------------------------------------ posterior algebra
def pair_posteriors(rho: Array, sigma1: Array, sigma2: Array) -> Array:
    """[K,J] belief x per-type strategies -> post[A, B, K, J]: the updated joint belief after the
    public simultaneous action pair (a, b) is observed, before any mechanical evidence.

    post[a, b, k, j] ∝ rho[k, j] * sigma1[k, a] * sigma2[j, b]   (proposal §7.2)

    Each (a, b) slice is normalized to sum to one; a pair no type combination plays is all-zero,
    never NaN.
    """
    rho = np.asarray(rho, dtype=np.float64)
    s1 = np.asarray(sigma1, dtype=np.float64)
    s2 = np.asarray(sigma2, dtype=np.float64)
    # unnorm[a, b, k, j] = rho[k, j] * s1[k, a] * s2[j, b]
    unnorm = rho[None, None, :, :] * s1.T[:, None, :, None] * s2.T[None, :, None, :]
    z = unnorm.sum(axis=(2, 3), keepdims=True)
    return np.divide(unnorm, z, out=np.zeros_like(unnorm), where=z > 0)


def floored_strategy(sigma: Array, avail: Array, floor: float) -> Array:
    """Mix a row-stochastic strategy with `floor` probability per available action (capped so the
    row stays a distribution). Unavailable columns stay exactly zero. Rows with no available
    action stay all-zero. Used both as the in-solve exploration floor and as the behavior-
    likelihood floor of proposal §7.2 ("the model says never" must become "the model says rarely",
    because zeroing a candidate from a modeled action choice would destroy it permanently)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    avail = np.asarray(avail, dtype=bool)
    n_avail = avail.sum(axis=1, keepdims=True).astype(np.float64)
    f = np.minimum(float(floor), np.divide(0.5, n_avail, out=np.zeros_like(n_avail),
                                           where=n_avail > 0))
    out = (1.0 - f * n_avail) * np.where(avail, sigma, 0.0) + f * avail
    return np.where(n_avail > 0, out, 0.0)


def behavior_update(rho: Array, like1: Array, like2: Array) -> tuple[Array, bool]:
    """One persistent-belief step after a real observed action pair (proposal §7.2, §14 step 11).

    `like1[K]` — modeled probability that OUR type k would play the action WE were observed to
    play; `like2[J]` — same for the opponent's observed action. Callers must build these from the
    distribution that actually acted (solver strategy during solver play, policy during
    policy-only play, floored per `floored_strategy`); a type whose label set lacks the observed
    action gets exact zero (elimination is mechanical, not behavioral).

    Returns (rho_next, alive). `alive=False` means every cell hit zero — the caller must trigger
    evidence-conditioned particle rejuvenation (proposal §7.4); the returned table is then the
    UNCHANGED input so no caller ever propagates an all-zero belief.
    """
    rho = np.asarray(rho, dtype=np.float64)
    unnorm = rho * np.outer(np.asarray(like1, dtype=np.float64),
                            np.asarray(like2, dtype=np.float64))
    z = unnorm.sum()
    if z <= 0.0:
        return rho.copy(), False
    return unnorm / z, True


# ------------------------------------------------------------------------------- best responses
def type_action_values(u: Array, rho: Array, sigma1: Array, sigma2: Array) -> tuple[Array, Array]:
    """Expected action values for each private type against the other side's current mixture.

    q1[k, a] = sum_j P(j | k) sum_b sigma2[j, b] * u[k, j, a, b]      (our payoff)
    q2[j, b] = sum_k P(k | j) sum_a sigma1[k, a] * (-u[k, j, a, b])   (opponent payoff, zero-sum)

    P(j | k) is row-normalized `rho` (what we believe about the opponent given our type k);
    P(k | j) is column-normalized (what the opponent believes about us given its type j). Types
    with zero public probability get zero value rows: their strategy is irrelevant and carries no
    weight in the other player's update.
    """
    rho = np.asarray(rho, dtype=np.float64)
    row = rho.sum(axis=1, keepdims=True)
    col = rho.sum(axis=0, keepdims=True)
    p_j_given_k = np.divide(rho, row, out=np.zeros_like(rho), where=row > 0)   # [K, J]
    p_k_given_j = np.divide(rho, col, out=np.zeros_like(rho), where=col > 0)   # [K, J]
    # ub[k, j, a] = sum_b sigma2[j, b] u[k, j, a, b];  q1[k, a] = sum_j p_j_given_k ub
    ub = np.einsum("kjab,jb->kja", u, sigma2)
    q1 = np.einsum("kja,kj->ka", ub, p_j_given_k)
    ua = np.einsum("kjab,ka->kjb", u, sigma1)
    q2 = -np.einsum("kjb,kj->jb", ua, p_k_given_j)
    return q1, q2


def exploitability(u: Array, rho: Array, sigma1: Array, sigma2: Array,
                   avail1: Array, avail2: Array) -> float:
    """How much either player could gain by deviating to its exact best response, summed over
    both players (zero at an exact equilibrium). Valid as an exact measure only when `u` is a
    FIXED payoff tensor (posterior-independent values); with posterior-dependent values it is a
    diagnostic at the supplied strategies. Best responses decompose per private type because each
    type is a separate information state: for fixed opposing strategy, type k independently picks
    its best action."""
    u = np.asarray(u, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    rho = rho / rho.sum()
    q1, q2 = type_action_values(u, rho, sigma1, sigma2)
    p_k = rho.sum(axis=1)
    p_j = rho.sum(axis=0)
    v = float(np.einsum("ka,k->", q1 * np.asarray(sigma1, dtype=np.float64), p_k))
    br1 = float(np.einsum("k,k->", p_k, np.where(avail1, q1, -np.inf).max(axis=1)))
    br2 = float(np.einsum("j,j->", p_j, np.where(avail2, q2, -np.inf).max(axis=1)))
    # our value is v; opponent's value is -v; gains are (br1 - v) and (br2 - (-v))
    return (br1 - v) + (br2 + v)


# ---------------------------------------------------------------------------------- the solver
@dataclass
class TwoSidedSolution:
    sigma1: Array            # [K, A] weighted-average strategy, floor-free (execute THIS)
    sigma2: Array            # [J, B] weighted-average opponent strategy (belief updates use this)
    iterations: int          # iterations actually run
    converged: bool          # True if the stability rule fired before the cap
    root_value: float        # diagnostic: our expected payoff under the averaged strategies and
    #                          the last computed payoff tensor (exact only for fixed-u games)
    # Random-iteration snapshot for value-query coverage: the
    # CURRENT floored strategy pair at one uniformly drawn iteration — exactly what `values_fn`
    # received there, so posteriors under it are beliefs the leaf values were actually queried at,
    # including the early near-uniform ones the final averages never revisit. None / -1 unless a
    # `sample_seed` was passed. `sample_iteration` is 1-indexed (1 = the first iteration).
    sample_sigma1: Array | None = None    # [K, A]
    sample_sigma2: Array | None = None    # [J, B]
    sample_iteration: int = -1


def _regret_matched(regret: Array, avail: Array) -> Array:
    """Row-wise regret matching: play proportional to positive accumulated regret over available
    actions; uniform over available when no regret is positive. No floor here — flooring is the
    caller's separately-accounted step."""
    pos = np.where(avail, np.maximum(regret, 0.0), 0.0)
    z = pos.sum(axis=1, keepdims=True)
    n_avail = avail.sum(axis=1, keepdims=True).astype(np.float64)
    uniform = np.divide(avail.astype(np.float64), n_avail, out=np.zeros_like(pos),
                        where=n_avail > 0)
    return np.where(z > 0, np.divide(pos, z, out=np.zeros_like(pos), where=z > 0), uniform)


def _normalized_rows(acc: Array, avail: Array) -> Array:
    z = acc.sum(axis=1, keepdims=True)
    out = np.divide(acc, z, out=np.zeros_like(acc), where=z > 0)
    return np.where(avail, out, 0.0)


def two_sided_regret_solve(
    values_fn: Callable[[Array, Array], Array],
    rho: Array,
    avail1: Array,
    avail2: Array,
    *,
    init1: Array | None = None,
    init2: Array | None = None,
    iters: int = 1000,
    floor: float = 0.02,
    plus: bool = True,
    check_every: int = 32,
    min_iters: int = 64,
    tol: float = 1e-3,
    sample_seed: int | None = None,
) -> TwoSidedSolution:
    """Solve one simultaneous public-belief decision for both players (proposal §10.1).

    Per iteration: derive each side's play from its accumulated regrets (the initialization on
    iteration 0), apply the annealed exploration floor, ask `values_fn` for the payoff tensor
    under the current strategies, compute each type's expected action values against the other
    side's mixture weighted by its conditional belief from `rho`, accumulate regrets for BOTH
    players, and accumulate the un-floored strategies into linearly weighted running averages.

    `init1/init2` are strategy initializations (e.g. the policy networks' distributions,
    proposal §10.2 — initialization, not anchoring: with linear weighting its influence on the
    average vanishes as iterations grow). Default: uniform over available actions.

    Stops early when, at a `check_every` checkpoint past `min_iters`, the largest per-row
    total-variation change of BOTH running averages since the previous checkpoint is below `tol`.

    The floor anneals as floor / (1 + t) — it exists to keep in-solve posterior updates away
    from exact zeros, and it is never part of the returned averages (settled decision 24).

    `sample_seed` (Phase G1, proposal §9.3 mitigation 2): when given, one iteration is drawn
    uniformly at random (reservoir sampling, so early stopping keeps the draw uniform over the
    iterations actually run) and the CURRENT floored strategy pair of that iteration — the exact
    arguments `values_fn` received — is returned in `sample_sigma1/sample_sigma2` with its
    1-indexed `sample_iteration`. Value training computes posteriors under this pair so the net
    sees the mid-solve belief queries, not only the final averages.
    """
    rho = np.asarray(rho, dtype=np.float64)
    rho = rho / rho.sum()
    avail1 = np.asarray(avail1, dtype=bool)
    avail2 = np.asarray(avail2, dtype=bool)
    k, a = avail1.shape
    j, b = avail2.shape

    def _uniform(avail: Array) -> Array:
        n = avail.sum(axis=1, keepdims=True).astype(np.float64)
        return np.divide(avail.astype(np.float64), n, out=np.zeros((*avail.shape,)), where=n > 0)

    base1 = _uniform(avail1) if init1 is None else np.where(avail1, np.asarray(init1, np.float64), 0.0)
    base2 = _uniform(avail2) if init2 is None else np.where(avail2, np.asarray(init2, np.float64), 0.0)
    base1 = _normalized_rows(base1, avail1)
    base2 = _normalized_rows(base2, avail2)

    r1 = np.zeros((k, a))
    r2 = np.zeros((j, b))
    acc1 = np.zeros((k, a))
    acc2 = np.zeros((j, b))
    prev1 = prev2 = None
    converged = False
    u = None
    t = 0
    sample_rng = None if sample_seed is None else np.random.default_rng(int(sample_seed))
    samp1: Array | None = None
    samp2: Array | None = None
    samp_iter = -1
    for t in range(max(1, int(iters))):
        floor_t = floor / (1.0 + t)
        play1 = floored_strategy(base1, avail1, floor_t)
        play2 = floored_strategy(base2, avail2, floor_t)
        if sample_rng is not None and sample_rng.random() < 1.0 / (t + 1):
            samp1, samp2, samp_iter = play1, play2, t + 1

        u = np.asarray(values_fn(play1, play2), dtype=np.float64)
        q1, q2 = type_action_values(u, rho, play1, play2)
        v1 = (play1 * np.where(avail1, q1, 0.0)).sum(axis=1, keepdims=True)
        v2 = (play2 * np.where(avail2, q2, 0.0)).sum(axis=1, keepdims=True)
        r1 += np.where(avail1, q1 - v1, 0.0)
        r2 += np.where(avail2, q2 - v2, 0.0)
        if plus:
            np.maximum(r1, 0.0, out=r1)
            np.maximum(r2, 0.0, out=r2)

        w = float(t + 1)                       # linear weighting: later rounds count more
        acc1 += w * base1
        acc2 += w * base2

        base1 = _regret_matched(r1, avail1)
        base2 = _regret_matched(r2, avail2)

        if (t + 1) >= min_iters and (t + 1) % check_every == 0:
            avg1 = _normalized_rows(acc1, avail1)
            avg2 = _normalized_rows(acc2, avail2)
            if prev1 is not None:
                tv1 = 0.5 * np.abs(avg1 - prev1).sum(axis=1).max()
                tv2 = 0.5 * np.abs(avg2 - prev2).sum(axis=1).max()
                if max(tv1, tv2) < tol:
                    converged = True
                    prev1, prev2 = avg1, avg2
                    break
            prev1, prev2 = avg1, avg2

    avg1 = _normalized_rows(acc1, avail1)
    avg2 = _normalized_rows(acc2, avail2)
    q1, _ = type_action_values(u, rho, avg1, avg2)
    p_k = rho.sum(axis=1)
    root_value = float(np.einsum("ka,k->", q1 * avg1, p_k))
    return TwoSidedSolution(sigma1=avg1, sigma2=avg2, iterations=t + 1,
                            converged=converged, root_value=root_value,
                            sample_sigma1=samp1, sample_sigma2=samp2,
                            sample_iteration=samp_iter)
