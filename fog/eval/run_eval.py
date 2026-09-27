"""Evaluation entry point: gauntlet, oracle-gap closure, readability, L-vs-N, ablations.

    # gauntlet + closure (default)
    python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt runs/.../ppo_final.pt

    # §7b/§7c readability report (fresh probe trained on this checkpoint's play)
    python -m fog.eval.run_eval --config ... --ckpt CKPT --readability

    # §7a L-vs-N head-to-head (CKPT is the L-agent)
    python -m fog.eval.run_eval --config ... --ckpt L.pt --lvn N.pt

    # β-grid frontier: config eval.pareto = [{label, ckpt, beta}, ...]
    python -m fog.eval.run_eval --config ... --ckpt any --pareto

    # ablation matrix: config eval.ablations = [{label, ckpt}, ...] (input-ablations of the same
    # weights — blind_reactive / set-oracle — are already covered by the closure report)
    python -m fog.eval.run_eval --config ... --ckpt any --ablations

Config keys (under sensible defaults): env.battle_format, env.team_pool, eval.games_per,
eval.closure_games, eval.closure_opponent, eval.include_reading_opponent,
eval.readability.{games, opponent, probe_epochs}, eval.lvn_games, eval.pareto, eval.ablations.
"""
from __future__ import annotations

import argparse

from fog.config import load_config_with_args


def main() -> None:
    ap = argparse.ArgumentParser(description="FOG evaluation (§9).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bc-ckpt", default=None)
    ap.add_argument("--readability", action="store_true", help="§7b/§7c fresh-probe report")
    ap.add_argument("--lvn", default=None, metavar="N_CKPT", help="§7a: play --ckpt (L) vs N_CKPT")
    ap.add_argument("--lvn-config", default=None, metavar="N_CONFIG",
                    help="config that builds the N side, when its architecture differs from "
                         "--config (e.g. the b_vec gate: M1.5 vs M1)")
    ap.add_argument("--pareto", action="store_true", help="β-grid frontier from cfg eval.pareto")
    ap.add_argument("--ablations", action="store_true", help="table from cfg eval.ablations")
    ap.add_argument("--set", nargs="*", dest="overrides", help="dotted config overrides")
    args = ap.parse_args()
    cfg = load_config_with_args(args.config, args.overrides)

    fmt = cfg.get("env", {}).get("battle_format", "gen9ou")
    ev = cfg.get("eval", {})
    team_pool = None
    if cfg.get("env", {}).get("team_pool"):
        from fog.env.teams import load_pool

        team_pool = load_pool(cfg["env"]["team_pool"])

    ran_special = False
    if args.readability:
        from fog.eval.readability import run_readability

        run_readability(args.ckpt, cfg)
        ran_special = True

    if args.lvn:
        from fog.eval.readability import lvn_match

        cfg_n = None
        if args.lvn_config:
            from fog.config import load_config

            cfg_n = load_config(args.lvn_config)
        print("== L vs N (§7a) ==")
        lvn_match(args.ckpt, args.lvn, int(ev.get("lvn_games", 3000)), fmt, team_pool, cfg,
                  cfg_n=cfg_n)
        ran_special = True

    if args.pareto:
        from fog.eval.readability import pareto_report

        entries = ev.get("pareto") or []
        if not entries:
            raise SystemExit("--pareto needs cfg eval.pareto = [{label, ckpt, beta}, ...]")
        pareto_report(entries, cfg)
        ran_special = True

    if args.ablations:
        from fog.eval.gauntlet import run_gauntlet

        entries = ev.get("ablations") or []
        if not entries:
            raise SystemExit("--ablations needs cfg eval.ablations = [{label, ckpt}, ...]")
        print("== Ablation matrix (training-knob ablations are separate runs; input ablations of ==")
        print("== the same weights are in the closure report)                                   ==")
        for e in entries:
            print(f"-- {e.get('label', e['ckpt'])}")
            run_gauntlet(e["ckpt"], games_per=int(ev.get("games_per", 2000)), fmt=fmt,
                         team_pool=team_pool, cfg=cfg)
        ran_special = True

    if ran_special:
        return

    from fog.eval.gauntlet import run_gauntlet
    from fog.eval.oracle_gap import run_closure_report

    print(f"== Gauntlet ({fmt}) ==")
    run_gauntlet(
        args.ckpt, games_per=int(ev.get("games_per", 2000)), fmt=fmt,
        team_pool=team_pool, cfg=cfg, bc_ckpt=args.bc_ckpt,
        include_reading_opponent=bool(ev.get("include_reading_opponent", team_pool is not None)),
    )

    print("== Oracle-gap closure ==")
    report = run_closure_report(
        args.ckpt, ev.get("closure_opponent", "SimpleHeuristics"),
        int(ev.get("closure_games", 2000)), fmt, team_pool, cfg,
    )
    for k, v in report.items():
        print(f"  {k:<16} {v:.3f}")


if __name__ == "__main__":
    main()
