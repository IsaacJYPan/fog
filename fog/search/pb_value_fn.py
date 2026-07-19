"""Phase-G leaf values: the public-belief value network wired into the solver's `values_fn`
(PROJECT_DESIGN_PROPOSAL.md §9, §10 steps 6-7; PROPOSAL_IMPL_PLAN.md Phase G — OWNER DECISION
2026-07-16). This is the successor of the interim in fog/search/pb_values.py, which reused the
policy's scalar value head with four posterior features.

THE COST STRUCTURE THIS EXISTS TO GET RIGHT (§9.2). Expensive work runs ONCE per decision: the
backbone trunk over every leaf (shared with the interim path), and the encoding of the K own- and J
opponent-candidate teams. The value network then produces, per leaf, the full per-pair table

    V[leaf, k, j] = our continuation payoff from this leaf if the true teams are (k, j),
                    conditioned on the belief at recompute time.

Each solver iteration turns those tables into the scalar the cell payoff needs by a plain average
over the leaf's CURRENT action-conditioned posterior (proposal §10.6) — no network pass:

    post[a, b, k, j]   = pair_posteriors(rho, sigma1, sigma2)     # the coupling channel, §5.1
    value(leaf)        = sum_{k,j} post[a_leaf, b_leaf, k, j] * V[leaf, k, j]
    u[k, j, a, b]      = sum over branches of branch_prob * value(leaf-or-terminal)

`recompute_every = N` controls how often the tables are refreshed by an actual network pass (the
tables' belief-conditioning goes stale as the strategies move — a pair's continuation value depends
on the belief carried into the leaf, the §5.1 thesis). Between refreshes only the cheap re-weighting
runs.

    recompute_every = 1  ==  THE CORRECTNESS REFERENCE (the "single value per query" form): every
                            iteration re-runs the net at the current posterior and weights by that
                            same posterior, so there is no drift. This is the definition the N > 1
                            approximation is measured against on the exact small-pool gate (§15.3);
                            it is exact but pays a full network pass per iteration.
    recompute_every = N  ==  the deployed approximation: one network pass per N iterations, cheap
                            re-weighting in between. The drift error vs. N = 1 is the measured knob.

`values_fn` counts its own calls to decide when a refresh is due, so the solver needs no changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from fog.search.two_sided import pair_posteriors

Array = np.ndarray


@dataclass
class PBValueStats:
    """Budget accounting (proposal §15). A network refresh is the expensive event; re-weightings are
    the cheap majority."""
    leaves: int              # live leaf observations behind the u tensor
    failed_cells: int        # simulation failures — their u entries are `fallback_value`
    trunk_forwards: int      # backbone batches over the leaf set (once, at construction)
    net_refreshes: int = 0   # PBValueNet passes over the leaves (one per `recompute_every` calls)
    reweightings: int = 0    # cheap table re-averagings (one per values_fn call)


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
) -> tuple[Callable[[Array, Array], Array], PBValueStats]:
    """Build `values_fn(sigma1, sigma2) -> u[K, J, A, B]` backed by the public-belief value network.

    `agent` supplies the shared backbone trunk (`head_inputs` -> s_board, c_dyn per leaf); `net` is a
    `PBValueNet`. `own_tokens [K,6,9]` / `opp_tokens [J,6,9]` are the SAME candidate identities the
    grid was built from (proposal §10 shared particles). `rho [K,J]` is the decision's root joint
    belief. `recompute_every = 1` gives the exact single-query reference; larger N trades measured
    drift for speed.

    If `trace` is a list, each `values_fn` call appends `{"post_leaf": [L,K,J], "v_table": [L,K,J],
    "refreshed": bool}` — the per-leaf posterior queried and the (possibly stale) table it was
    weighted against. The drift gate (fog/eval/pb_value_drift.py) consumes this to measure the §9.2
    re-weighting error against full re-evaluation.
    """
    import torch

    from fog.env.engine_runner import _device, _stack_obs

    agent.eval()
    net.eval()
    device = _device(agent)

    own_t = torch.as_tensor(np.asarray(own_tokens, dtype=np.int64), device=device)
    opp_t = torch.as_tensor(np.asarray(opp_tokens, dtype=np.int64), device=device)
    n_k, n_j = own_t.shape[0], opp_t.shape[0]

    # ---- one-time trunk pass over every live leaf (the §19.13 expensive half) ------------------
    s_board_parts: list = []
    c_dyn_parts: list = []
    trunk_forwards = 0
    for lo in range(0, len(grid.leaf_obs), max_batch_forward):
        chunk = _stack_obs(grid.leaf_obs[lo: lo + max_batch_forward], device)
        with torch.no_grad():
            s_board, c_dyn = agent.head_inputs(chunk)
        s_board_parts.append(s_board)
        c_dyn_parts.append(c_dyn)
        trunk_forwards += 1
    n_leaves = len(grid.leaf_obs)
    # widths are irrelevant when empty: every consumer below is guarded by `n_leaves`.
    s_board_all = torch.cat(s_board_parts, dim=0) if s_board_parts else torch.zeros((0, 0), device=device)
    c_dyn_all = torch.cat(c_dyn_parts, dim=0) if c_dyn_parts else torch.zeros((0, 0), device=device)

    stats = PBValueStats(leaves=n_leaves, failed_cells=int(grid.failed.sum()),
                         trunk_forwards=trunk_forwards)

    # ---- leaf -> (a, b) so a leaf can read its own action-conditioned posterior slice -----------
    positions = np.argwhere(grid.leaf_index >= 0)                      # rows (k, j, a, b, m)
    leaf_ab = np.zeros((n_leaves, 2), dtype=np.int64)
    if len(positions):
        leaf_ab[grid.leaf_index[tuple(positions.T)]] = positions[:, [2, 3]]
    a_idx = leaf_ab[:, 0]
    b_idx = leaf_ab[:, 1]

    rho = np.asarray(rho, dtype=np.float64)
    terminal_fill = np.where(np.isnan(grid.terminal_value), 0.0, grid.terminal_value)

    v_table: Array | None = None          # cached [L, K, J], refreshed every `recompute_every` calls
    call_count = 0

    def _post_leaf(sigma1: Array, sigma2: Array) -> Array:
        """Each leaf's current action-conditioned posterior [L, K, J] (proposal §10.6)."""
        post = pair_posteriors(rho, sigma1, sigma2)                   # [A, B, K, J]
        if n_leaves == 0:
            return np.zeros((0, n_k, n_j))
        return post[a_idx, b_idx]                                     # [L, K, J]

    def _refresh(post_leaf: Array) -> Array:
        """One actual network pass: condition V on the current per-leaf posterior (§9.2)."""
        stats.net_refreshes += 1
        rho_t = torch.as_tensor(post_leaf, dtype=torch.float32, device=device)
        out = np.zeros((n_leaves, n_k, n_j), dtype=np.float64)
        for lo in range(0, n_leaves, max_batch_forward):
            hi = min(lo + max_batch_forward, n_leaves)
            with torch.no_grad():
                v = net(s_board_all[lo:hi], c_dyn_all[lo:hi], own_t, opp_t, rho_t[lo:hi])
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
        call_count += 1
        stats.reweightings += 1
        if trace is not None:
            trace.append({"post_leaf": post_leaf.copy(),
                          "v_table": v_table.copy() if v_table is not None else post_leaf.copy(),
                          "refreshed": refreshed})

        # cheap re-weighting: scalar(leaf) = <post_leaf, V_leaf>  (plain arithmetic, no net pass)
        if n_leaves:
            leaf_scalar = (post_leaf * v_table).sum(axis=(1, 2))      # [L]
        else:
            leaf_scalar = np.zeros((0,))

        # all-terminal grid (no leaves) -> np.where would still index empty leaf_scalar; guard it.
        per_branch = (np.where(grid.leaf_index >= 0,
                               leaf_scalar[np.maximum(grid.leaf_index, 0)], terminal_fill)
                      if n_leaves else terminal_fill)
        u = (grid.branch_prob * per_branch).sum(axis=-1)
        u[grid.failed] = float(fallback_value)
        return u

    return values_fn, stats
