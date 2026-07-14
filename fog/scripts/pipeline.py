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
    python -m fog.scripts.pipeline gauntlet-search    # Phase 2: arm gauntlet + h2h + §5 gate
                                                      #          (engine; §4.2 gate must be green)
    python -m fog.scripts.pipeline gate-bvec          # THE go/no-go: M1.5-vs-M1 paired h2h
                                                      #          (Showdown, both runs finished)
    python -m fog.scripts.pipeline distill            # Phase 4: distill search play -> policy
    python -m fog.scripts.pipeline refresh-beliefs    # Phase 4: belief consistency retrain (§7)
    python -m fog.scripts.pipeline agreement          # Phase 3: exact-vs-sampled agreement
    python -m fog.scripts.pipeline gate-belief        # M1.5 belief gate: pool-id top-1 by turn 4
                                                      #          (engine, no server)
    python -m fog.scripts.pipeline probe              # engine throughput probe (ARCHITECTURE.md §3.2)

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
           "--games", str(args.games), "--iters", str(args.iters), "--seed", str(args.seed),
           "--opponent", args.opponent]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    _run(cmd)


def cmd_distill(args) -> None:
    cmd = [sys.executable, "-m", "fog.training.distill",
           "--data", str(args.data), "--out", str(args.out), "--scope", args.scope]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    _run(cmd)


def cmd_finetune_value(args) -> None:
    cmd = [sys.executable, "-m", "fog.training.value_finetune",
           "--data", str(args.data), "--out", str(args.out)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    _run(cmd)


def cmd_agreement(args) -> None:
    cmd = [sys.executable, "-m", "fog.eval.source_agreement",
           "--pool", str(args.pool), "--ckpt", args.ckpt, "--games", str(args.games),
           "--ks", *[str(k) for k in args.ks], "--seed", str(args.seed)]
    if args.out:
        cmd += ["--out", str(args.out)]
    _run(cmd)


def cmd_refresh_beliefs(args) -> None:
    cmd = [sys.executable, "-m", "fog.training.belief_refresh",
           "--ckpt", args.ckpt, "--pool", str(args.pool), "--out", str(args.out),
           "--games", str(args.games), "--seed", str(args.seed)]
    _run(cmd)


def cmd_gate_belief(args) -> None:
    cmd = [sys.executable, "-m", "fog.eval.pool_id",
           "--ckpt", args.ckpt, "--config", args.config, "--pool", str(args.pool),
           "--games", str(args.games), "--seed", str(args.seed)]
    if args.out:
        cmd += ["--out", str(args.out)]
    _run(cmd)


def cmd_gate_bvec(args) -> None:
    """THE go/no-go (SEARCH_PLAN §3 / M1.5 gate): paired head-to-head M1.5 (b_vec ON) vs M1
    (b_vec OFF) at matched training budget, on the Showdown harness (server up). The two
    checkpoints have different head widths, so each side is built from its own config.
    Gate: >=3% win-rate gain, i.e. head-to-head point >= 0.53 — and quote the CI."""
    from fog.config import load_config
    from fog.env.teams import load_pool
    from fog.eval.readability import lvn_match

    cfg_l = load_config(args.config_l)
    cfg_n = load_config(args.config_n)
    pool = load_pool(cfg_l["env"]["team_pool"])
    print(f">>> b_vec gate: {args.ckpt_l} (L, b_vec) vs {args.ckpt_n} (N) | {args.games} games")
    r = lvn_match(args.ckpt_l, args.ckpt_n, args.games,
                  cfg_l.get("env", {}).get("battle_format", "gen9ou"), pool, cfg_l, cfg_n=cfg_n)
    passed = r["point"] >= 0.53
    ci_ok = r["low"] > 0.5
    print(f"b_vec gate (>=0.53 head-to-head): point {r['point']:.3f} "
          f"[{r['low']:.3f}, {r['high']:.3f}] -> {'PASS' if passed else 'FAIL'}"
          f"{' (CI-separated from 0.5)' if ci_ok else ' (CI includes 0.5 — more games)' if passed else ''}")
    if not passed:
        print("b_vec gate FAILED: conditioning on beliefs buys no win rate at this budget — "
              "the payoff channel for BOTH the beta penalty and belief-space search is in doubt "
              "(SEARCH_PLAN §3). Check whether both runs are undertrained before concluding.")


def cmd_gauntlet_search(args) -> None:
    cmd = [sys.executable, "-m", "fog.eval.search_gauntlet",
           "--pool", str(args.pool), "--ckpt", args.ckpt, "--out", str(args.out),
           "--games", str(args.games), "--h2h-games", str(args.h2h_games),
           "--iters", str(args.iters), "--seed", str(args.seed)]
    if args.det_k is not None:
        cmd += ["--det-k", str(args.det_k)]
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
    p.add_argument("--opponent", type=str, default="MaxDamage",
                   help="MaxDamage/Random, 'self' (Phase-4 self-play), or a checkpoint path")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_log_posteriors)

    p = sub.add_parser("distill", help="Phase 4: distill search play into the policy (§7)")
    p.add_argument("--data", type=Path, default=Path("runs/posterior_log"))
    p.add_argument("--out", type=Path, default=Path("runs/distill"))
    p.add_argument("--ckpt", type=str, default=None,
                   help="input checkpoint (natural: the §4.2 value_finetune output)")
    p.add_argument("--scope", choices=("heads", "full"), default="heads")
    p.set_defaults(fn=cmd_distill)

    p = sub.add_parser("finetune-value", help="§4.2: posterior-aware value fine-tune + gate")
    p.add_argument("--data", type=Path, default=Path("runs/posterior_log"))
    p.add_argument("--out", type=Path, default=Path("runs/value_finetune"))
    p.add_argument("--ckpt", type=str, default=None)
    p.set_defaults(fn=cmd_finetune_value)

    p = sub.add_parser("agreement",
                       help="Phase 3: exact-vs-sampled own-state decision agreement (engine)")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_gen"))
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--ks", type=int, nargs="*", default=[8, 16, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/agreement/report.json"))
    p.set_defaults(fn=cmd_agreement)

    p = sub.add_parser("refresh-beliefs",
                       help="Phase 4: retrain belief nets on the distilled agent's self-play (§7)")
    p.add_argument("--ckpt", type=str, required=True, help="distilled checkpoint")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.add_argument("--out", type=Path, default=Path("runs/belief_refresh"))
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_refresh_beliefs)

    p = sub.add_parser("gate-belief",
                       help="M1.5 belief gate: pool-id top-1 by turn 4 (engine, no server)")
    p.add_argument("--ckpt", type=str, default="runs/m1_5_belief/checkpoints/ppo_final.pt")
    p.add_argument("--config", type=str, default="configs/m1_5_belief.yaml")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/pool_id_m15/report.json"))
    p.set_defaults(fn=cmd_gate_belief)

    p = sub.add_parser("gate-bvec",
                       help="THE go/no-go: paired M1.5-vs-M1 head-to-head (Showdown, server up)")
    p.add_argument("--ckpt-l", type=str,
                   default="runs/m1_5_belief/checkpoints/ppo_final.pt", help="b_vec-ON checkpoint")
    p.add_argument("--ckpt-n", type=str,
                   default="runs/m1_poc/checkpoints/ppo_final.pt", help="b_vec-OFF checkpoint")
    p.add_argument("--config-l", type=str, default="configs/m1_5_belief.yaml")
    p.add_argument("--config-n", type=str, default="configs/m1_poc.yaml")
    p.add_argument("--games", type=int, default=1000)
    p.set_defaults(fn=cmd_gate_bvec)

    p = sub.add_parser("gauntlet-search",
                       help="Phase 2: policy/determinized/joint gauntlet + h2h + §5 gate")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.add_argument("--ckpt", type=str, required=True,
                   help="posterior-aware checkpoint (value_finetune output; §4.2 gate first)")
    p.add_argument("--out", type=Path, default=Path("runs/search_gauntlet"))
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--h2h-games", type=int, default=200)
    p.add_argument("--det-k", type=int, default=None)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_gauntlet_search)

    p = sub.add_parser("probe", help="engine throughput probe")
    p.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    p.set_defaults(fn=cmd_probe)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
