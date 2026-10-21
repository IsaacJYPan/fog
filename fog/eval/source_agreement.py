"""Compare enumerated and sampled own-state search on identical positions.

The learner follows `EnumeratedPoolStates` in engine games. At each nontrivial
decision, `SampledOwnStates` is evaluated at every requested K using the same
position and search seed. Agreement means equal argmax actions at temperature zero.
Sampled choices do not steer the game, so this measures decisions on the exact
source's trajectory rather than each sampler's resulting win rate. For that,
use `search_gauntlet --own-source sampled`.

The measurement costs 1 + len(ks) searches per decision. Off-pool play needs sampled
candidates because there is no finite pool to enumerate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LEARNER = "p1"
OPPONENT = "p2"
TURN_BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 10**9))


def aggregate_agreement(records: list[tuple[int, int, bool, int, int]], ks: tuple[int, ...],
                        *, games: int, failed: int) -> dict:
    """records: (turn, k, agree, k_exact_states, k_sampled_states) -> report dict."""
    per_k = {}
    for k in ks:
        rows = [(t, a, ke, ks_) for t, kk, a, ke, ks_ in records if kk == k]
        if not rows:
            per_k[k] = {"decisions": 0}
            continue
        turns = np.array([t for t, *_ in rows])
        agree = np.array([a for _, a, *_ in rows], dtype=bool)
        by_bucket = {}
        for lo, hi in TURN_BUCKETS:
            m = (turns >= lo) & (turns <= hi)
            if m.any():
                label = f"t{lo}+" if hi >= 10**9 else f"t{lo}-{hi}"
                by_bucket[label] = float(agree[m].mean())
        per_k[k] = {
            "decisions": len(rows),
            "agreement": float(agree.mean()),
            "agreement_by_turn": by_bucket,
            "mean_exact_states": float(np.mean([ke for *_, ke, _ in rows])),
            "mean_sampled_states": float(np.mean([ks_ for *_, ks_ in rows])),
        }
    return {"games": games, "failed_games": failed, "ks": list(ks), "per_k": per_k}


def run_agreement(agent: Any, pool: list[str], *, ks=(8, 16, 32), games: int = 64,
                  iters: int = 16, seed: int = 0, max_turns: int = 300,
                  tables=None) -> dict:
    """Play `games` reference games; measure sampled-source agreement at each K. Returns the
    report dict (also printed)."""
    import random

    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.search.own_states import EnumeratedPoolStates, SampledOwnStates

    if tables is None:
        from fog.constants.tables import TABLES as tables
    rng = random.Random(seed)
    agent.eval()
    exact_source = EnumeratedPoolStates(pool, tables)
    sampled_sources = {k: SampledOwnStates(agent, tables, k=int(k), seed=seed + 13 * k)
                       for k in ks}

    records: list[tuple[int, int, bool, int, int]] = []
    failed = 0
    for g in range(int(games)):
        battle = EngineBattle(rng.choice(pool), rng.choice(pool), seed=rng.randrange(2**31),
                              tables=tables, battle_tag=f"agree-{g}")
        opp = EngineMaxDamagePolicy(tables=tables, seed=rng.randrange(2**31))
        try:
            _play_measured_game(battle, agent, opp, exact_source, sampled_sources, records,
                                iters=iters, rng=rng, max_turns=max_turns)
        except Exception as exc:  # noqa: BLE001 - failed game is counted, printed, and bounded
            failed += 1
            print(f"[agreement] game {g} failed and was dropped: {exc!r}", flush=True)
    if failed > max(1, int(games) // 5):
        raise RuntimeError(f"{failed}/{games} games failed — a defect, not noise. Fix it first.")

    report = aggregate_agreement(records, tuple(ks), games=int(games) - failed, failed=failed)
    for k in ks:
        r = report["per_k"][k]
        if not r["decisions"]:
            print(f"[agreement] K={k}: no decisions measured")
            continue
        buckets = "  ".join(f"{t}:{v:.2f}" for t, v in r["agreement_by_turn"].items())
        print(f"[agreement] K={k:<3} agreement {r['agreement']:.3f} over {r['decisions']} "
              f"decisions | by turn {buckets} | mean states exact {r['mean_exact_states']:.1f} "
              f"sampled {r['mean_sampled_states']:.1f}", flush=True)
    return report


def _play_measured_game(battle, agent, opp_policy, exact_source, sampled_sources, records, *,
                        iters, rng, max_turns) -> None:
    from fog.search.joint import joint_search_action

    while not battle.finished and battle.turn <= max_turns:
        moves: dict[str, str] = {}
        pending = battle.pending_sides()
        if LEARNER in pending:
            legal = np.flatnonzero(battle.action_mask(LEARNER))
            if len(legal) <= 1:
                action = int(legal[0]) if len(legal) else 13
            else:
                s = rng.randrange(2**31)
                res = joint_search_action(agent, battle, LEARNER, own_source=exact_source,
                                          iters=iters, temperature=0.0, seed=s)
                for k, source in sampled_sources.items():
                    res_k = joint_search_action(agent, battle, LEARNER, own_source=source,
                                                iters=iters, temperature=0.0, seed=s)
                    records.append((int(battle.turn), k, res_k.action == res.action,
                                    len(res.prior), len(res_k.prior)))
                action = res.action
                from fog.encoding.events import record_history_decision

                record_history_decision(
                    battle.trackers[LEARNER],
                    battle.turn,
                    battle.view(LEARNER),
                    battle.tables,
                    action,
                )
            moves[LEARNER] = battle.engine_action(LEARNER, action)
        if OPPONENT in pending:
            moves[OPPONENT] = battle.engine_action(
                OPPONENT, opp_policy.choose(battle, OPPONENT)
            )
        battle.step(moves)


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.env.teams import load_pool
    from fog.training.value_finetune import load_posterior_agent

    ap = argparse.ArgumentParser(description="Measure exact-versus-sampled own-state agreement.")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=Path, default=None, help="optional report.json path")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--ks", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = load_posterior_agent(cfg, args.ckpt, device)

    report = run_agreement(agent, load_pool(args.pool), ks=tuple(args.ks), games=args.games,
                           iters=args.iters, seed=args.seed, max_turns=args.max_turns)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[agreement] report -> {args.out}")


if __name__ == "__main__":
    main()
