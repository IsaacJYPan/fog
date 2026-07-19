"""Drift gate for the per-pair value table: does re-weighting a stale table approximate re-running
the network? (PROJECT_DESIGN_PROPOSAL.md §9.2 + §15.3 line "the error from re-weighting a stored
per-pair value table instead of re-running the network is measured against full re-evaluation on
the exact small-pool harness"; PROPOSAL_IMPL_PLAN.md Phase G drift-vs-full-reeval measurement.)

WHY THIS IS THE LOAD-BEARING MEASUREMENT. The whole Option-2 decision (per-pair table refreshed
every N iterations, cheap re-weighting in between) is only worth anything if the re-weighting stays
close to a full network re-evaluation. If it drifts badly at every usable N, the approximation
collapses back to Option 1's cost (a full encoder pass per leaf per iteration). This module measures
the drift so the choice of N is a MEASURED knob, not a hope.

Two metrics, deliberately separated:

  * `reweight_staleness_curve` — the ISOLATED approximation error. Run the exact reference solve
    (recompute_every=1, so the table is refreshed every iteration) and, along that single
    trajectory, ask: at iteration t, how wrong is the leaf value if the table were `s` iterations
    stale? i.e. `<post_t, V_{t-s}>` vs the exact `<post_t, V_t>`. This isolates the table's staleness
    from any change in the solve trajectory — a clean "how stale can the table get before the value
    is off by epsilon" curve, in the same units the cell payoff sums (per-leaf value).

  * `solution_gap` — the END-TO-END consequence. Solve fully at recompute_every=N and compare the
    FINAL averaged strategies and root value to the exact (N=1) solve. This captures the compounding
    effect the staleness curve omits (a wrong value feeds back into the next strategy update). This
    is what actually matters for decision quality; the staleness curve explains it.

COST NOTE (honest budget). The reference arm here runs a full network pass EVERY iteration — exactly
the cost Option 2 exists to avoid. That is fine: this is a measurement gate run on the small exact
pools, not the deployment path. Both solves are deterministic given the same init, so the only
difference between the N-arm and the reference is the value approximation.

CAVEAT ON UNTRAINED NETS. Drift magnitude depends on how strongly the trained V depends on the belief
it is conditioned on. On a randomly-initialized net the numbers are a MECHANISM smoke test only
(they prove the metric responds to real table drift); the gate-quality numbers must be taken with a
trained checkpoint.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fog.search.pb_value_fn import make_pb_value_grid_values
from fog.search.two_sided import two_sided_regret_solve

Array = np.ndarray


def _leaf_scalar(post_leaf: Array, v_table: Array) -> Array:
    """<post_leaf, V> per leaf: [L,K,J] . [L,K,J] -> [L] (the number the cell payoff sums)."""
    if post_leaf.size == 0:
        return np.zeros((0,))
    return (post_leaf * v_table).sum(axis=(1, 2))


def reweight_staleness_curve(
    agent: Any,
    net: Any,
    grid: Any,
    rho: Array,
    own_tokens: Array,
    opp_tokens: Array,
    avail1: Array,
    avail2: Array,
    *,
    staleness: tuple[int, ...] = (1, 2, 4, 8, 16),
    iters: int = 256,
    floor: float = 0.02,
) -> dict[int, dict[str, float]]:
    """Per-leaf value error from an `s`-iteration-stale table, along the exact reference trajectory.

    Returns {s: {"max": worst per-leaf abs error over the run, "mean": mean per-leaf abs error}}.
    `s = 1` is the error from being one iteration behind; larger `s` is a more stale table. All errors
    are in leaf-value units (the same units the solve's payoffs and root value live in).
    """
    trace: list = []
    values_fn, _ = make_pb_value_grid_values(
        agent, net, grid, rho, own_tokens, opp_tokens, recompute_every=1, trace=trace)
    # tol=0 => never early-stop, so the trace is a full `iters`-length trajectory.
    two_sided_regret_solve(values_fn, rho, avail1, avail2, iters=iters, floor=floor, tol=0.0)

    posts = [tr["post_leaf"] for tr in trace]
    tables = [tr["v_table"] for tr in trace]        # fresh every step (recompute_every=1)
    exact = [_leaf_scalar(posts[t], tables[t]) for t in range(len(trace))]

    out: dict[int, dict[str, float]] = {}
    for s in staleness:
        maxes: list[float] = []
        means: list[float] = []
        for t in range(s, len(trace)):
            stale = _leaf_scalar(posts[t], tables[t - s])
            d = np.abs(stale - exact[t])
            if d.size:
                maxes.append(float(d.max()))
                means.append(float(d.mean()))
        out[int(s)] = {"max": max(maxes) if maxes else 0.0,
                       "mean": float(np.mean(means)) if means else 0.0}
    return out


def solution_gap(
    agent: Any,
    net: Any,
    grid: Any,
    rho: Array,
    own_tokens: Array,
    opp_tokens: Array,
    avail1: Array,
    avail2: Array,
    *,
    Ns: tuple[int, ...] = (1, 2, 4, 8, 16),
    iters: int = 256,
    floor: float = 0.02,
) -> dict[int, dict[str, float]]:
    """Final-solution difference between an N-refresh solve and the exact (N=1) solve.

    Returns {N: {"sigma1_tv", "sigma2_tv": worst per-row total-variation vs the exact solve;
    "root_value_gap": |root value difference|; "net_refreshes"; "iterations"}}. N=1 is the reference,
    so its gaps are exactly zero (a built-in self-consistency check).
    """
    def _solve(n: int):
        vfn, stats = make_pb_value_grid_values(
            agent, net, grid, rho, own_tokens, opp_tokens, recompute_every=n)
        sol = two_sided_regret_solve(vfn, rho, avail1, avail2, iters=iters, floor=floor, tol=0.0)
        return sol, stats

    ref, _ = _solve(1)

    def _tv(a: Array, b: Array) -> float:
        return float(0.5 * np.abs(a - b).sum(axis=1).max()) if a.size else 0.0

    out: dict[int, dict[str, float]] = {}
    for n in Ns:
        sol, stats = _solve(n)
        out[int(n)] = {
            "sigma1_tv": _tv(sol.sigma1, ref.sigma1),
            "sigma2_tv": _tv(sol.sigma2, ref.sigma2),
            "root_value_gap": float(abs(sol.root_value - ref.root_value)),
            "net_refreshes": int(stats.net_refreshes),
            "iterations": int(sol.iterations),
        }
    return out


def format_report(staleness: dict[int, dict[str, float]], gap: dict[int, dict[str, float]]) -> str:
    """A compact human-readable summary of both metrics (for the gate log)."""
    lines = ["reweight drift vs full re-evaluation (§9.2/§15.3)",
             "  staleness  mean_err   max_err"]
    for s in sorted(staleness):
        r = staleness[s]
        lines.append(f"  s={s:<7d} {r['mean']:.5f}   {r['max']:.5f}")
    lines.append("  N        sigma1_tv sigma2_tv root_val_gap refreshes iters")
    for n in sorted(gap):
        r = gap[n]
        lines.append(f"  N={n:<6d} {r['sigma1_tv']:.5f}   {r['sigma2_tv']:.5f}   "
                     f"{r['root_value_gap']:.5f}      {r['net_refreshes']:<9d} {r['iterations']}")
    return "\n".join(lines)
