"""One-line drivers for the SEARCH_PLAN phases (and the M-milestone runs they depend on).

Every subcommand is a single terminal line; extra `--set key=value` overrides are forwarded to
the underlying config. Anything Showdown-backed needs the local server up
(`cd server/pokemon-showdown; node pokemon-showdown start --no-security`); engine-backed runs
need no server, EXCEPT that engine training keeps `cross_sim.every_decisions > 0`, whose gauntlet
does need it (the binding divergence control — don't disable it, start the server).

    python -m fog.scripts.pipeline m15                # Phase 0: M1.5 belief-supervised PPO (gate!)
    python -m fog.scripts.pipeline m3-control         # Phase 0: leakage beta=0 control (engine)
    python -m fog.scripts.pipeline m3-grid            # Phase 0: beta grid x seeds (engine),
                                                      #          baseline_run wired automatically
    python -m fog.scripts.pipeline smoke-joint        # Phase 1: live joint-search smoke (engine,
                                                      #          no training, prints the decision)
    python -m fog.scripts.pipeline log-posteriors     # §4.2: log search posteriors (engine)
    python -m fog.scripts.pipeline finetune-value     # §4.2: value fine-tune + regression gate
    python -m fog.scripts.pipeline probe              # engine throughput probe (SIMULATION_PLAN)

The M1.5 gate is the go/no-go for the whole search program (SEARCH_PLAN §3): if b_vec buys no
win rate, information has no payoff channel and neither the beta penalty nor belief-space search
has anything to work with.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

M3_BETAS = (0.01, 0.03, 0.1)
M3_SEEDS = (0, 1, 2)
M3_CONTROL_DIR = "runs/m3_engine_beta0"


def _run(argv: list[str]) -> None:
    print(f"\n>>> {' '.join(argv)}", flush=True)
    r = subprocess.run(argv)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _train(config: str, overrides: list[str]) -> None:
    cmd = [sys.executable, "-m", "fog.training.ppo", "--config", config]
    if overrides:
        cmd += ["--set", *overrides]
    _run(cmd)


def cmd_m15(args) -> None:
    _train("configs/m1_5_belief.yaml", args.overrides or [])


def cmd_m3_control(args) -> None:
    _train("configs/m3_leakage_engine.yaml",
           ["leakage.beta=0", f"log.dir={M3_CONTROL_DIR}", *(args.overrides or [])])


def cmd_m3_grid(args) -> None:
    if not Path(M3_CONTROL_DIR).exists():
        sys.exit(f"control run missing ({M3_CONTROL_DIR}): run `m3-control` first — the grid's "
                 "watchdog baselines come from it (HANDOFF run-order rule).")
    for beta in args.betas:
        for seed in args.seeds:
            _train("configs/m3_leakage_engine.yaml", [
                f"leakage.beta={beta}", f"seed={seed}",
                f"leakage.baseline_run={M3_CONTROL_DIR}",
                f"log.dir=runs/m3_engine_b{beta}_s{seed}",
                *(args.overrides or []),
            ])


def cmd_log_posteriors(args) -> None:
    cmd = [sys.executable, "-m", "fog.search.posterior_log",
           "--pool", str(args.pool), "--out", str(args.out),
           "--games", str(args.games), "--iters", str(args.iters), "--seed", str(args.seed)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    _run(cmd)


def cmd_finetune_value(args) -> None:
    cmd = [sys.executable, "-m", "fog.training.value_finetune",
           "--data", str(args.data), "--out", str(args.out)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    _run(cmd)


def cmd_probe(args) -> None:
    _run([sys.executable, "-m", "fog.env.engine_runner", "--probe",
          "--pool", str(args.pool), "--config", "configs/default.yaml"])


def cmd_smoke_joint(args) -> None:
    """Live end-to-end joint-search smoke: fresh pool battle, a few scripted turns, one joint
    decision with exact pool enumeration. Verifies the whole Phase-1 stack without training."""
    import numpy as np

    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent
    from fog.search.joint import joint_search_action
    from fog.search.own_states import EnumeratedPoolStates

    pool = load_pool(args.pool)
    agent = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    agent.eval()
    battle = EngineBattle(pool[0], pool[min(1, len(pool) - 1)], seed=args.seed,
                          battle_tag="joint-smoke")

    rand, maxd = EngineRandomPolicy(seed=args.seed), EngineMaxDamagePolicy(seed=args.seed + 1)
    for _ in range(args.turns):
        if battle.finished or "p1" in battle.pending_sides():
            break
        battle.step({r: battle.engine_action(r, (rand if r == "p1" else maxd).choose(battle, r))
                     for r in battle.pending_sides()})
    if battle.finished or "p1" not in battle.pending_sides():
        sys.exit("smoke battle ended before a searcher decision; retry with another --seed")

    source = EnumeratedPoolStates(pool, battle.tables)
    res = joint_search_action(agent, battle, "p1", own_source=source, iters=args.iters,
                              seed=args.seed)
    print(f"pool={len(pool)} teams | consistent own-states K={len(res.prior)} "
          f"(true index {res.true_index}) | public actions={len(res.labels)} "
          f"| sim failures={res.failures}")
    print(f"labels: {res.labels}")
    with np.printoptions(precision=3, suppress=True):
        print(f"true-state policy over labels: {res.policies[res.true_index]}")
        print(f"posterior over states per action:\n{res.posterior.T}")
        print(f"chosen action: {res.action} | 14-dim dist: {res.action_dist}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("m15", help="Phase 0: M1.5 belief-supervised PPO (Showdown; the gate run)")
    p.add_argument("--set", nargs="*", dest="overrides")
    p.set_defaults(fn=cmd_m15)

    p = sub.add_parser("m3-control", help="Phase 0: M3 beta=0 control on the engine backend")
    p.add_argument("--set", nargs="*", dest="overrides")
    p.set_defaults(fn=cmd_m3_control)

    p = sub.add_parser("m3-grid", help="Phase 0: M3 beta grid x seeds (control must exist)")
    p.add_argument("--betas", nargs="*", type=float, default=list(M3_BETAS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(M3_SEEDS))
    p.add_argument("--set", nargs="*", dest="overrides")
    p.set_defaults(fn=cmd_m3_grid)

    p = sub.add_parser("smoke-joint", help="Phase 1: live joint-search smoke (no training)")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_smoke_joint)

    p = sub.add_parser("log-posteriors", help="§4.2: log search posteriors from joint-search games")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.add_argument("--out", type=Path, default=Path("runs/posterior_log"))
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_log_posteriors)

    p = sub.add_parser("finetune-value", help="§4.2: posterior-aware value fine-tune + gate")
    p.add_argument("--data", type=Path, default=Path("runs/posterior_log"))
    p.add_argument("--out", type=Path, default=Path("runs/value_finetune"))
    p.add_argument("--ckpt", type=str, default=None)
    p.set_defaults(fn=cmd_finetune_value)

    p = sub.add_parser("probe", help="engine throughput probe")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.set_defaults(fn=cmd_probe)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
