"""Probability-guided depth-two expansion for public-belief search.

Full-width depth multiplies the already large K x J x A x B grid at every leaf.  The appropriate
analogue of selective Monte Carlo tree search is therefore to solve depth one first, compute each
public leaf's actual reach probability under that solution, and expand only the most likely leaves.

An expanded child is still a two-sided public-belief solve.  Its root belief is the parent's Bayes
update after the visible action pair and truthful branch evidence.  The child returns a continuation
value for every surviving private-state pair; the parent substitutes those pair values at that leaf
and is solved again.  Unexpanded leaves retain the learned value estimate.

This is an explicit depth-two scaffold, not the final unbounded algorithm.  The long-run target is
outcome-sampling public-belief regret search, where repeated traversals revisit high-reach public
states without materializing a full tree.  The scaffold is useful now because it validates belief
propagation, pair-specific value survival, simulator forking beyond one ply, and the compute knobs
before self-play training is committed to a deeper expert.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fog.search.pb_grid import build_pb_grid
from fog.search.pb_value_fn import make_pb_value_grid_values
from fog.search.pb_values import make_grid_values
from fog.search.two_sided import pair_posteriors, two_sided_regret_solve


@dataclass
class SelectiveDepthStats:
    considered: int = 0
    expanded: int = 0
    expanded_reach_mass: float = 0.0
    child_sims: int = 0
    failures: list[str] = field(default_factory=list)


def leaf_reach_probabilities(grid: Any, rho: np.ndarray, sigma1: np.ndarray,
                             sigma2: np.ndarray) -> np.ndarray:
    """Probability of reaching each concrete chance leaf under the solved root strategy."""
    reach = np.zeros(len(grid.leaf_obs), dtype=np.float64)
    for k, j, a, b, branch in np.argwhere(grid.leaf_index >= 0):
        leaf = int(grid.leaf_index[k, j, a, b, branch])
        reach[leaf] += (
            rho[k, j] * sigma1[k, a] * sigma2[j, b]
            * grid.branch_prob[k, j, a, b, branch]
        )
    return reach


def _leaf_action_pair(grid: Any) -> np.ndarray:
    positions = np.argwhere(grid.leaf_index >= 0)
    result = np.zeros((len(grid.leaf_obs), 2), dtype=np.int64)
    if len(positions):
        result[grid.leaf_index[tuple(positions.T)]] = positions[:, [2, 3]]
    return result


def build_depth2_overrides(
    agent: Any,
    root_grid: Any,
    rho: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    role: str,
    own_tokens: np.ndarray,
    opp_tokens: np.ndarray,
    *,
    rng: random.Random,
    value_net: Any | None = None,
    value_recompute_every: int = 8,
    child_iters: int = 64,
    floor: float = 0.02,
    min_prob: float = 0.0,
    max_branches: int | None = None,
    max_nodes: int = 2,
    min_reach: float = 0.01,
    fallback_value: float = 0.0,
) -> tuple[np.ndarray, SelectiveDepthStats]:
    """Return [L,K,J] pair-value overrides, NaN at leaves left to the value network."""
    n_leaves = len(root_grid.leaf_obs)
    n_k, n_j = rho.shape
    overrides = np.full((n_leaves, n_k, n_j), np.nan, dtype=np.float64)
    stats = SelectiveDepthStats()
    if max_nodes <= 0 or n_leaves == 0:
        return overrides, stats
    if len(getattr(root_grid, "leaf_battles", [])) != n_leaves:
        raise ValueError("depth-two expansion requires one retained battle fork per public leaf")

    reach = leaf_reach_probabilities(root_grid, rho, sigma1, sigma2)
    leaf_ab = _leaf_action_pair(root_grid)
    root_post = pair_posteriors(rho, sigma1, sigma2)
    ranked = np.argsort(reach)[::-1]

    for leaf in ranked:
        if stats.expanded >= int(max_nodes) or reach[leaf] < float(min_reach):
            break
        child_battle = root_grid.leaf_battles[int(leaf)]
        if role not in child_battle.pending_sides():
            continue
        if len(np.flatnonzero(child_battle.action_mask(role))) <= 1:
            continue
        stats.considered += 1

        a, b = leaf_ab[leaf]
        child_rho_full = root_post[a, b] * root_grid.leaf_evidence[leaf]
        total = child_rho_full.sum()
        if total <= 0:
            stats.failures.append(f"leaf {leaf}: evidence removed all candidate pairs")
            continue
        child_rho_full = child_rho_full / total
        alive_k = np.flatnonzero(child_rho_full.sum(axis=1) > 0)
        alive_j = np.flatnonzero(child_rho_full.sum(axis=0) > 0)
        child_rho = child_rho_full[np.ix_(alive_k, alive_j)]
        child_rho = child_rho / child_rho.sum()

        try:
            child_grid = build_pb_grid(
                child_battle, role, own_tokens[alive_k], opp_tokens[alive_j],
                seed=rng.randrange(2**31), min_prob=min_prob,
                max_branches=max_branches,
            )
            if value_net is None:
                values_fn, _ = make_grid_values(
                    agent, child_grid, child_rho, fallback_value=fallback_value
                )
            else:
                values_fn, _ = make_pb_value_grid_values(
                    agent, value_net, child_grid, child_rho,
                    own_tokens[alive_k], opp_tokens[alive_j],
                    recompute_every=value_recompute_every,
                    fallback_value=fallback_value,
                )
            solution = two_sided_regret_solve(
                values_fn, child_rho, child_grid.avail1, child_grid.avail2,
                iters=child_iters, floor=floor, sample_seed=rng.randrange(2**31),
            )
            payoff = values_fn(solution.sigma1, solution.sigma2)
            pair_value = np.einsum(
                "ka,jb,kjab->kj", solution.sigma1, solution.sigma2, payoff
            )
        except Exception as exc:
            stats.failures.append(f"leaf {leaf}: {type(exc).__name__}: {exc}"[:240])
            continue

        overrides[int(leaf)][np.ix_(alive_k, alive_j)] = pair_value
        stats.expanded += 1
        stats.expanded_reach_mass += float(reach[leaf])
        stats.child_sims += int(child_grid.sims_run)

    return overrides, stats
