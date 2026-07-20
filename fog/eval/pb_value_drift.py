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


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    """Measured drift numbers on real pool positions (the Phase G4 gate run — mechanism tests
    live in tests/test_pb_value_drift.py; THIS produces the §15.3 NUMBERS with a trained net).

    Positions: random pool matchups advanced `--depths` decision-steps by the scripted
    max-damage policy on both sides, measured at the first decision we own. The conditioning
    belief is the runner-style mechanical posterior (uniform prior + all reveals to date);
    behavior-likelihood concentration is absent — recorded approximation: mechanical-only rho is
    LESS concentrated, which makes the drift measurement conservative (wider beliefs move the
    table more per iteration, not less).

        python -m fog.eval.pb_value_drift --pool data/teams/pool_a \
            --ckpt runs/g0_v2_champion_resume/checkpoints/ppo_final.pt \
            --net runs/pb_value_a/pb_value.pt --out runs/pb_value_drift.json
    """
    import argparse
    import json
    import random
    from pathlib import Path

    import torch

    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.env.teams import load_pool
    from fog.eval.pb_truncation_check import opening_table
    from fog.models.agent import FogAgent
    from fog.models.pb_value import PBValueNet
    from fog.search.own_states import StablePoolOwnStates
    from fog.search.pb_grid import build_pb_grid

    ap = argparse.ArgumentParser(description="PBValueNet reweight-drift gate (§9.2/§15.3).")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ckpt", required=True, help="champion agent checkpoint (frozen trunk)")
    ap.add_argument("--net", default=None,
                    help="trained PBValueNet state dict; default random = mechanism smoke ONLY")
    ap.add_argument("--positions", type=int, default=3)
    ap.add_argument("--depths", default="0,4,8",
                    help="decision-step depths positions are advanced to (cycled)")
    ap.add_argument("--staleness", default="1,2,4,8,16")
    ap.add_argument("--ns", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=128)
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--max-branches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pastes = load_pool(args.pool)
    src = StablePoolOwnStates(pastes)
    agent = FogAgent.from_tables({})
    agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    agent.eval()
    net = PBValueNet.from_tables(state_dim=int(agent.backbone.state_dim),
                                 dyn_dim=int(agent.backbone.dyn_dim))
    if args.net:
        net.load_state_dict(torch.load(args.net, map_location="cpu"))
    else:
        print("[pb_value_drift] WARNING: no --net — random net, numbers are a mechanism "
              "smoke, NOT gate-quality (module docstring caveat).", flush=True)
    net.eval()

    depths = [int(x) for x in args.depths.split(",")]
    staleness = tuple(int(x) for x in args.staleness.split(","))
    ns = tuple(int(x) for x in args.ns.split(","))
    rng = random.Random(args.seed)
    results = []
    built = attempts = 0
    while built < args.positions and attempts < args.positions * 10:
        attempts += 1
        i, j = rng.randrange(len(pastes)), rng.randrange(len(pastes))
        depth = depths[built % len(depths)]
        battle = EngineBattle(pastes[i], pastes[j], seed=rng.randrange(2**31),
                              battle_tag=f"drift-{built}")
        policy = EngineMaxDamagePolicy(tables=battle.tables, seed=rng.randrange(2**31))
        steps = 0
        # advance `depth` decision-steps, then keep stepping until WE own a decision
        while not battle.finished and (steps < depth or "p1" not in battle.pending_sides()):
            moves = {side: battle.engine_action(side, int(policy.choose(battle, side)))
                     for side in battle.pending_sides()}
            battle.step(moves)
            steps += 1
        if battle.finished:
            continue                                    # game ended first — resample matchup

        table = opening_table(battle, "p1", src.tokens, src.tokens)
        rho_full = table.normalized()
        kept_k = rho_full.sum(axis=1) > 0
        kept_j = rho_full.sum(axis=0) > 0
        rho = rho_full[kept_k][:, kept_j]
        rho = rho / rho.sum()
        own_t, opp_t = src.tokens[kept_k], src.tokens[kept_j]
        grid = build_pb_grid(battle, "p1", own_t, opp_t, seed=rng.randrange(2**31),
                             min_prob=args.min_prob, max_branches=args.max_branches)
        curve = reweight_staleness_curve(agent, net, grid, rho, own_t, opp_t,
                                         grid.avail1, grid.avail2,
                                         staleness=staleness, iters=args.iters,
                                         floor=args.floor)
        gap = solution_gap(agent, net, grid, rho, own_t, opp_t, grid.avail1, grid.avail2,
                           Ns=ns, iters=args.iters, floor=args.floor)
        print(f"--- position {built} (matchup {i}x{j}, depth {steps}, "
              f"K={int(kept_k.sum())} J={int(kept_j.sum())}, leaves={len(grid.leaf_obs)})")
        print(format_report(curve, gap), flush=True)
        results.append({"matchup": [i, j], "depth": steps,
                        "K": int(kept_k.sum()), "J": int(kept_j.sum()),
                        "leaves": len(grid.leaf_obs),
                        "staleness": {str(s): r for s, r in curve.items()},
                        "solution_gap": {str(n): r for n, r in gap.items()}})
        built += 1

    summary = {
        "per_position": results,
        "worst_sigma1_tv_by_N": {str(n): max(r["solution_gap"][str(n)]["sigma1_tv"]
                                             for r in results) for n in ns},
        "worst_root_gap_by_N": {str(n): max(r["solution_gap"][str(n)]["root_value_gap"]
                                            for r in results) for n in ns},
        "trained_net": bool(args.net),
        "config": vars(args),
    }
    print("worst sigma1 TV by N:", summary["worst_sigma1_tv_by_N"])
    print("worst root-value gap by N:", summary["worst_root_gap_by_N"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
