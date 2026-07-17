"""Two-sided public-belief regret solver (PROJECT_DESIGN_PROPOSAL.md §7, §10) — pure math core.

This module is the claim-bearing successor of the one-sided loop in fog/search/joint.py. The
one-sided loop optimizes only OUR per-type strategies against a fixed scripted reply; this module
optimizes BOTH players' private-type-conditioned strategies together, which is what the proposal's
Bayesian-equilibrium target requires (§21.2). It is deliberately engine-free and network-free:
everything battle-specific arrives through the `values_fn` callback, so every property here is
testable on exact hand-built games (tests/test_two_sided.py).

Objects and their meanings (proposal §7, §10.1):

  * `rho[K, J]`    — the joint public belief: probability that OUR private team is candidate `k`
                     AND the opponent's private team is candidate `j`, given everything public.
                     Both players can compute it; neither player's strategy may depend on the
                     dimension it cannot observe.
  * `sigma1[K, A]` — our strategy: probability of public action `a` when our private state is `k`.
  * `sigma2[J, B]` — the opponent's strategy over ITS public actions `b`, per opponent type `j`.
  * `avail1/avail2` — boolean masks: which public-action labels each candidate type can legally
                     play. A type that lacks a label plays it with probability zero, so observing
                     that label eliminates the type (proposal §7.3 — load-bearing, tested).
  * `values_fn(sigma1, sigma2) -> u[K, J, A, B]` — OUR expected continuation payoff if the true
                     types are `(k, j)` and the committed simultaneous actions are `(a, b)`.
                     Zero-sum: the opponent's payoff is `-u`. The callback receives the CURRENT
                     iteration's strategies because leaf continuation values may depend on the
                     action-conditioned posterior those strategies induce (§5.1, §10). For a
                     posterior-independent game it may ignore its arguments and return a constant
                     tensor. Entries at unavailable `(k, a)` or `(j, b)` combinations are ignored.

Information restrictions enforced by construction (proposal §4.3, §19.1-3):
  * `sigma1` varies with `k` only — a row's update uses the opponent-type distribution `P(j | k)`
    derived from `rho`, never a specific `j`.
  * `sigma2` varies with `j` only — symmetric.
  * Neither strategy conditions on the other player's CURRENT action; actions meet only inside
    the payoff expectation.

Solver mechanics (proposal §10.1 + settled decision 24):
  * Regret matching with optional nonnegative accumulated regrets (`plus=True`, the standard
    "regret matching plus" variant: clamping accumulated regret at zero speeds convergence).
  * The reported answer is the WEIGHTED AVERAGE strategy, weighting iteration `t` by `t + 1`
    (later rounds count more; with equal weights and few iterations the "average" is mostly the
    initialization).
  * A small exploration floor keeps every legal action at nonzero probability DURING the solve
    (posterior updates stay numerically stable) but is annealed toward zero and EXCLUDED from the
    accumulated average: the floored strategy is what plays each iteration; the un-floored
    regret-matched strategy is what accumulates. The floor must not survive into the answer.
  * Stopping: by measured stability of the running averages (total-variation change between
    checkpoints), with a hard iteration cap. Iterations used are reported.

What this module does NOT do (later phases, PROPOSAL_IMPL_PLAN.md): build the K x J x A x B cell
grid from engine simulations, evaluate leaves with the public-belief value network, or maintain
the persistent cross-turn belief table (only the single-step update primitive lives here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    for t in range(max(1, int(iters))):
        floor_t = floor / (1.0 + t)
        play1 = floored_strategy(base1, avail1, floor_t)
        play2 = floored_strategy(base2, avail2, floor_t)

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
                            converged=converged, root_value=root_value)
