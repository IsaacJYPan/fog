"""G2 truncation validation (PROPOSAL_IMPL_PLAN Phase G2; amendment A3's reconsideration gate):
how much does posterior-mass truncation move the EXECUTED action row?

For each sampled pool position, solve the same decision at several `top_j` caps and compare each
arm's executed-row distribution (the true own state's averaged strategy mapped to real action
indices) against a large-J reference by total variation. The plan's tolerance question — "is
top-J at the data-gen budget close enough?" — is answered by these numbers, not argument; if the
TV exceeds tolerance, raise J budgets or add importance-weighted cell sampling (A3).

Positions are matchup OPENINGS (the most expensive real decisions and the ones with the widest
live candidate sets — later turns only concentrate rho, making truncation strictly easier), with
the belief table built exactly the way the runner builds it: pool-uniform prior + the initial
both-sides mechanical reveal step. The own side is capped at a constant `top_k` across arms so
the comparison isolates the opponent-side cap.

Usage:
    python -m fog.eval.pb_truncation_check --pool data/teams/pool_gen --positions 3 \
        --js 8,16,32 --j-ref 64 --top-k 8 --iters 64 --out runs/pb_trunc_check.json
"""
from __future__ import annotations

import argparse
import json
import random
import time


def opening_table(battle, role: str, own_tokens, opp_tokens):
    """The runner's opening belief state: uniform pool prior, then the initial mechanical step
    (team preview / leads are public before the first decision)."""
    import numpy as np

    from fog.constants.schema import NO_ITEM
    from fog.encoding.observation import Observation
    from fog.search.belief_table import JointBeliefTable
    from fog.search.pb_log import keep_mask_from_reveals

    opp_role = "p2" if role == "p1" else "p1"
    no_item = int(battle.tables.item_index(NO_ITEM))
    table = JointBeliefTable.from_margins(
        np.zeros(len(own_tokens)) - np.log(len(own_tokens)),
        np.zeros(len(opp_tokens)) - np.log(len(opp_tokens)))
    own_view = Observation.from_battle(battle.view(opp_role), tracker=battle.trackers[opp_role])
    opp_view = Observation.from_battle(battle.view(role), tracker=battle.trackers[role])
    table.mechanical_step(
        keep_own=keep_mask_from_reveals(own_view.opp_revealed["tokens"], own_tokens,
                                        no_item_token=no_item),
        keep_opp=keep_mask_from_reveals(opp_view.opp_revealed["tokens"], opp_tokens,
                                        no_item_token=no_item))
    return table


def executed_row_tv(agent, battle, role: str, own_source, opp_pool_tokens, *,
                    js: list[int], j_ref: int, top_k: int | None, seed: int,
                    search_kwargs: dict | None = None) -> dict:
    """One position's arm comparison. Returns {'j_used': {J: kept}, 'tv': {J: TV vs j_ref},
    'dropped_opp_mass': {J: mass}, 'wall_s': {J: seconds}}."""
    import numpy as np

    from fog.search.pb_search import pb_search_action

    table = opening_table(battle, role, own_source.tokens, opp_pool_tokens)
    rows: dict[int, np.ndarray] = {}
    out = {"j_used": {}, "dropped_opp_mass": {}, "wall_s": {}}
    for arm in [*js, j_ref]:
        t0 = time.perf_counter()
        r = pb_search_action(
            agent, battle, role, own_source=own_source, opp_tokens=opp_pool_tokens,
            table=table, top_k=top_k, top_j=int(arm), rng=random.Random(seed),
            **(search_kwargs or {}))
        rows[arm] = r.action_dist
        out["j_used"][arm] = int(r.j_used)
        out["dropped_opp_mass"][arm] = float(r.dropped_opp_mass)
        out["wall_s"][arm] = round(time.perf_counter() - t0, 2)
    out["tv"] = {arm: float(0.5 * abs(rows[arm] - rows[j_ref]).sum()) for arm in js}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--positions", type=int, default=3)
    ap.add_argument("--js", default="8,16,32")
    ap.add_argument("--j-ref", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--max-branches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from fog.env.engine_battle import EngineBattle
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent
    from fog.search.own_states import StablePoolOwnStates

    pastes = load_pool(args.pool)
    src = StablePoolOwnStates(pastes)
    agent = FogAgent.from_tables({})
    js = [int(x) for x in args.js.split(",")]
    rng = random.Random(args.seed)
    kwargs = {"iters": args.iters, "min_prob": args.min_prob,
              "max_branches": args.max_branches}

    results = []
    for p in range(args.positions):
        i, j = rng.randrange(len(pastes)), rng.randrange(len(pastes))
        battle = EngineBattle(pastes[i], pastes[j], seed=rng.randrange(2**31),
                              battle_tag=f"trunc-check-{p}")
        res = executed_row_tv(agent, battle, "p1", src, src.tokens, js=js,
                              j_ref=args.j_ref, top_k=args.top_k,
                              seed=rng.randrange(2**31), search_kwargs=kwargs)
        res["matchup"] = [i, j]
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)

    summary = {"per_position": results,
               "mean_tv": {str(a): sum(r["tv"][a] for r in results) / len(results)
                           for a in js},
               "config": vars(args)}
    print("mean TV per arm:", summary["mean_tv"])
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
