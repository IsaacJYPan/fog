"""Interim scalar leaf values for the two-sided search.

Bridges a simulated PBGrid to `two_sided_regret_solve`'s `values_fn` contract using the EXISTING
value head. It is retained so the two-sided machinery can run end-to-end and expose
information-flow defects; the real public-belief value network consumes candidate identities and
the full joint weights.

What is interim here, named precisely:

  * The leaf evaluator is the small policy/value head on `(private_state, public_context, four
    posterior features)`. Four concentration numbers cannot say which candidate teams retain
    probability; they only make leaf values react to the solver's changing posterior at all.
  * The features summarize the OWN-side margin of the action-pair posterior (what the opponent
    would now believe about us), evaluated at the leaf's own candidate index — the same semantics
    the frozen one-sided baseline fed its head, so its §4.2 value fine-tune transfers.
  * §10.6's discrete branch-evidence conditioning is not applied — the interim head has no input
    for it. The grid's leaf observations already carry the reveal facts for the replacement path.

Cost model (invariant 19.13): the expensive parts — engine simulation (already inside the grid)
and the backbone trunk over all leaves — happen ONCE here at construction. Each solver iteration
re-runs only the small head, and when the agent has no posterior-features input at all the u
tensor is computed once and reused (leaf values cannot react to the posterior, so iterating the
head would be waste — that configuration is the §18 degenerate mode, pinned by the baseline's
tests, useful only as a contrast arm).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fog.search.two_sided import pair_posteriors

Array = np.ndarray


def _posterior_features(post_own: Array, k_index: int) -> Array:
    """Four concentration features of an own-side posterior column: [normalized entropy, mass on
    this candidate, top-1 mass, effective-candidate fraction]. Deliberate duplicate of the frozen
    baseline's private helper (fog/search/joint.py `posterior_features_np`) — new code never
    imports from the frozen module; this interim remains a controlled comparison only."""
    p = np.asarray(post_own, dtype=np.float64)
    k = len(p)
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    h_norm = h / np.log(k) if k > 1 else 0.0
    return np.array([h_norm, float(p[k_index]), float(p.max()) if k else 0.0,
                     float(np.exp(h)) / k if k else 0.0], dtype=np.float32)


@dataclass
class GridValueStats:
    """Reported per decision (proposal §15 asks for honest budget accounting)."""
    leaves: int                  # live leaf observations behind the u tensor
    failed_cells: int            # simulation failures — their u entries are `fallback_value`
    trunk_forwards: int          # backbone batches run at construction (once)
    head_passes: int = 0         # full head sweeps over the leaves (one per values_fn call
    #                              unless the static fast path holds)


def make_grid_values(
    agent: Any,
    grid: Any,
    rho: Array,
    *,
    fallback_value: float = 0.0,
    max_batch_forward: int = 256,
    leaf_pair_values: Array | None = None,
) -> tuple[Callable[[Array, Array], Array], GridValueStats]:
    """Build `values_fn(sigma1, sigma2) -> u[K, J, A, B]` over a simulated PBGrid.

    u[k, j, a, b] = sum over the cell's chance branches of
                    probability x (terminal payoff, or head value at the censored leaf).

    `rho` is the decision's root joint belief (`JointBeliefTable.normalized()`): the per-leaf
    posterior features are the own-side margin of `pair_posteriors(rho, sigma1, sigma2)` at the
    leaf's action pair — recomputed from the CURRENT strategies every call, which is exactly the
    coupling channel (§5.1: same cached leaf, different iteration, different belief).

    `fallback_value` fills cells whose simulation failed (the caller passes the root value, the
    baseline's convention). Unavailable cells come out 0 and are masked by the solver.
    """
    import torch

    from fog.env.engine_runner import _device, _stack_obs

    agent.eval()
    device = _device(agent)
    posterior_dim = int(getattr(agent.policy_value, "posterior_dim", 0))
    if posterior_dim:
        # A zero-pad-expanded checkpoint (load_posterior_agent on a blind head) has exactly-zero
        # weights on the value MLP's posterior input columns, so the features cannot move the
        # output — iterating the head would recompute an identical u every call. Take the static
        # fast path until the first fine-tune/distill makes those columns nonzero. Exact-zero
        # check on purpose: any trained column, however small, restores the aware path.
        w = agent.policy_value.value[0].weight
        if float(w[:, -posterior_dim:].abs().sum().item()) == 0.0:
            posterior_dim = 0

    # ---- one-time trunk cache over every live leaf (the §19.13 expensive half) ----------------
    head_cache: list[tuple] = []
    for lo in range(0, len(grid.leaf_obs), max_batch_forward):
        chunk = _stack_obs(grid.leaf_obs[lo: lo + max_batch_forward], device)
        with torch.no_grad():
            private_state, public_context = agent.head_inputs(chunk)
        head_cache.append((private_state, public_context, chunk["legality_mask"]))
    stats = GridValueStats(leaves=len(grid.leaf_obs), failed_cells=int(grid.failed.sum()),
                           trunk_forwards=len(head_cache))

    # ---- leaf -> (k, a, b) for feature lookup --------------------------------------------------
    positions = np.argwhere(grid.leaf_index >= 0)                      # rows (k, j, a, b, m)
    leaf_kab = np.zeros((len(grid.leaf_obs), 3), dtype=np.int64)
    leaf_kab[grid.leaf_index[tuple(positions.T)]] = positions[:, [0, 2, 3]]

    rho = np.asarray(rho, dtype=np.float64)
    terminal_fill = np.where(np.isnan(grid.terminal_value), 0.0, grid.terminal_value)
    pair_override = None
    if leaf_pair_values is not None:
        pair_override = np.asarray(leaf_pair_values, dtype=np.float64)
        want = (len(grid.leaf_obs), *rho.shape)
        if pair_override.shape != want:
            raise ValueError(f"leaf_pair_values must be {want}, got {pair_override.shape}")
    static_u: Array | None = None

    def _leaf_values(sigma1: Array, sigma2: Array) -> Array:
        feats = None
        if posterior_dim:
            post_own = pair_posteriors(rho, sigma1, sigma2).sum(axis=3)     # [A, B, K]
            feats = (np.stack([_posterior_features(post_own[a, b], int(k))
                               for k, a, b in leaf_kab])
                     if len(leaf_kab) else np.zeros((0, posterior_dim), dtype=np.float32))
        values = np.zeros(len(grid.leaf_obs))
        pos = 0
        for private_state, public_context, mask in head_cache:
            n = private_state.shape[0]
            pf = (torch.as_tensor(feats[pos: pos + n], device=device)
                  if feats is not None else None)
            with torch.no_grad():
                _, v = agent.policy_value(
                    private_state, mask, public_context, posterior_features=pf
                )
            values[pos: pos + n] = v.cpu().numpy()
            pos += n
        stats.head_passes += 1
        return values

    def values_fn(sigma1: Array, sigma2: Array) -> Array:
        nonlocal static_u
        if posterior_dim == 0 and static_u is not None:
            return static_u                     # values cannot react to the posterior: compute once
        v = _leaf_values(sigma1, sigma2)
        # A grid can be all-terminal (every branch ends the game, e.g. a KO near game end) -> no
        # leaves and `v` is empty; np.where still evaluates the (unused) leaf lookup, so guard the
        # index. All-terminal means per_branch is just the terminal payoffs.
        if v.size:
            safe_leaf = np.maximum(grid.leaf_index, 0)
            if pair_override is None:
                branch_value = v[safe_leaf]
            else:
                leaf_value = np.where(
                    np.isfinite(pair_override), pair_override, v[:, None, None]
                )
                k_axis = np.arange(rho.shape[0])[:, None, None, None, None]
                j_axis = np.arange(rho.shape[1])[None, :, None, None, None]
                branch_value = leaf_value[safe_leaf, k_axis, j_axis]
            per_branch = np.where(grid.leaf_index >= 0, branch_value, terminal_fill)
        else:
            per_branch = terminal_fill
        u = (grid.branch_prob * per_branch).sum(axis=-1)
        u[grid.failed] = float(fallback_value)
        if posterior_dim == 0:
            static_u = u
        return u

    return values_fn, stats
