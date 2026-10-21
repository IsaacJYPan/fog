"""Per-pair leaf values for the public-belief regret solver.

Backbone leaf encodings and candidate-team encodings are cached once per decision.
A network refresh produces V[leaf, k, j], conditioned on that leaf's public-observation
posterior under the current strategies:

    likelihood[o,k,j] = sum_{a,b} sigma1[k,a] sigma2[j,b] P(o | k,j,a,b)
    post_leaf = normalize(rho * likelihood[public_observation_of_leaf])
    u[k,j,a,b] = sum over branches of branch_prob * V[leaf,k,j]

Each simulated cell uses its own candidate pair's value. Cross-pair averaging
happens in the solver; averaging here too would apply the belief twice.

`recompute_every=1` refreshes on every call and is the reference calculation.
Larger values reuse a table as the strategies change, trading stale belief
conditioning for speed. `fog.eval.pb_value_drift` measures the resulting error.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fog.search.public_observation import PublicObservationLikelihood

Array = np.ndarray


@dataclass
class PBValueStats:
    """Budget accounting. A network refresh is expensive; cached-table uses are cheap."""
    leaves: int              # live leaf observations behind the u tensor
    failed_cells: int        # simulation failures — their u entries are `fallback_value`
    trunk_forwards: int      # backbone batches over the leaf set (once, at construction)
    representation_forwards: int = 0  # leaf + candidate encoder preparation (once per decision)
    net_refreshes: int = 0   # PBValueNet passes over the leaves (one per `recompute_every` calls)
    reweightings: int = 0    # legacy name: cached table uses (one per values_fn call)


def make_pb_value_grid_values(
    agent: Any,
    net: Any,
    grid: Any,
    rho: Array,
    own_tokens: Array,
    opp_tokens: Array,
    *,
    recompute_every: int = 8,
    fallback_value: float = 0.0,
    max_batch_forward: int = 256,
    trace: list | None = None,
    leaf_pair_values: Array | None = None,
) -> tuple[Callable[[Array, Array], Array], PBValueStats]:
    """Build `values_fn(sigma1, sigma2) -> u[K, J, A, B]` backed by the public-belief value network.

    `agent` supplies the shared trunk (`head_inputs` -> private state and public context per leaf).
    `own_tokens [K,6,15]` / `opp_tokens [J,6,15]` must match the candidate identities used to build
    the grid. `rho [K,J]` is the decision's root joint
    belief. `recompute_every = 1` gives the exact single-query reference; larger N trades measured
    drift for speed.

    If `trace` is a list, each `values_fn` call appends `{"post_leaf": [L,K,J], "v_table": [L,K,J],
    "refreshed": bool}` — the evidence-conditioned per-leaf belief queried and the possibly stale
    table used. The drift gate consumes this to measure cached-table error against full
    re-evaluation.
    """
    import torch

    from fog.env.engine_runner import _device, _stack_obs

    agent.eval()
    net.eval()
    device = _device(agent)

    own_t = torch.as_tensor(np.asarray(own_tokens, dtype=np.int64), device=device)
    opp_t = torch.as_tensor(np.asarray(opp_tokens, dtype=np.int64), device=device)
    n_k, n_j = own_t.shape[0], opp_t.shape[0]

    # Cache the backbone encodings for all live leaves.
    private_state_parts: list = []
    public_context_parts: list = []
    trunk_forwards = 0
    for lo in range(0, len(grid.leaf_obs), max_batch_forward):
        chunk = _stack_obs(grid.leaf_obs[lo: lo + max_batch_forward], device)
        with torch.no_grad():
            private_state, public_context = agent.head_inputs(chunk)
        private_state_parts.append(private_state)
        public_context_parts.append(public_context)
        trunk_forwards += 1
    n_leaves = len(grid.leaf_obs)
    # widths are irrelevant when empty: every consumer below is guarded by `n_leaves`.
    private_state_all = torch.cat(private_state_parts, dim=0) if private_state_parts else torch.zeros((0, 0), device=device)
    public_context_all = torch.cat(public_context_parts, dim=0) if public_context_parts else torch.zeros((0, 0), device=device)

    stats = PBValueStats(leaves=n_leaves, failed_cells=int(grid.failed.sum()),
                         trunk_forwards=trunk_forwards)

    # The public leaf encoder and both candidate-team encoders are independent of rho and of the
    # regret strategies. Cache them once. Only the belief-conditioned message rounds and pair head
    # are allowed to run at a refresh. A custom test net without this split keeps the reference
    # `forward` fallback.
    prepared = None
    if n_leaves and hasattr(net, "prepare") and hasattr(net, "forward_cached"):
        with torch.no_grad():
            prepared = net.prepare(
                private_state_all, public_context_all, own_t, opp_t
            )
        stats.representation_forwards = 1

    observation_model = PublicObservationLikelihood(grid)

    rho = np.asarray(rho, dtype=np.float64)
    terminal_fill = np.where(np.isnan(grid.terminal_value), 0.0, grid.terminal_value)
    pair_override = None
    if leaf_pair_values is not None:
        pair_override = np.asarray(leaf_pair_values, dtype=np.float64)
        want = (n_leaves, n_k, n_j)
        if pair_override.shape != want:
            raise ValueError(f"leaf_pair_values must be {want}, got {pair_override.shape}")

    v_table: Array | None = None          # cached [L, K, J], refreshed every `recompute_every` calls
    call_count = 0

    def _post_leaf(sigma1: Array, sigma2: Array) -> Array:
        return observation_model.posteriors(rho, sigma1, sigma2)

    def _refresh(post_leaf: Array) -> Array:
        """One actual network pass: condition V on the current per-leaf posterior."""
        stats.net_refreshes += 1
        rho_t = torch.as_tensor(post_leaf, dtype=torch.float32, device=device)
        out = np.zeros((n_leaves, n_k, n_j), dtype=np.float64)
        for lo in range(0, n_leaves, max_batch_forward):
            hi = min(lo + max_batch_forward, n_leaves)
            with torch.no_grad():
                if prepared is not None:
                    from fog.models.pb_value import PBValueCache

                    chunk = PBValueCache(
                        ctx=prepared.ctx[lo:hi], own=prepared.own, opp=prepared.opp
                    )
                    v = net.forward_cached(chunk, rho_t[lo:hi])
                else:
                    v = net(
                        private_state_all[lo:hi], public_context_all[lo:hi],
                        own_t, opp_t, rho_t[lo:hi],
                    )
            out[lo:hi] = v.detach().cpu().numpy()
        return out

    def values_fn(sigma1: Array, sigma2: Array) -> Array:
        nonlocal v_table, call_count
        post_leaf = _post_leaf(sigma1, sigma2)
        refreshed = False
        if n_leaves:  # nothing to refresh for an all-terminal grid (no network work, no count)
            refreshed = v_table is None or (call_count % max(1, int(recompute_every)) == 0)
            if refreshed:
                v_table = _refresh(post_leaf)
                if pair_override is not None:
                    v_table = np.where(
                        np.isfinite(pair_override), pair_override, v_table
                    )
        call_count += 1
        stats.reweightings += 1
        if trace is not None:
            trace.append({"post_leaf": post_leaf.copy(),
                          "v_table": v_table.copy() if v_table is not None else post_leaf.copy(),
                          "refreshed": refreshed})

        # A branch generated from private pair (k,j) receives V[leaf,k,j]. The solver later
        # averages these pair payoffs once. Never collapse V over post_leaf here.
        if n_leaves:
            safe_leaf = np.maximum(grid.leaf_index, 0)
            k_axis = np.arange(n_k)[:, None, None, None, None]
            j_axis = np.arange(n_j)[None, :, None, None, None]
            branch_leaf_value = v_table[safe_leaf, k_axis, j_axis]
            per_branch = np.where(grid.leaf_index >= 0, branch_leaf_value, terminal_fill)
        else:
            per_branch = terminal_fill
        u = (grid.branch_prob * per_branch).sum(axis=-1)
        u[grid.failed] = float(fallback_value)
        return u

    return values_fn, stats
