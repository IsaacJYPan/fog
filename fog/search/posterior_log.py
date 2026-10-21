"""Collect one-sided joint-search decisions for value fitting and distillation.

Engine self-play logs the root observation, the executed action's posterior
features at the true team, the 14-action search distribution, and the game outcome.
Trivial decisions and counterfactual leaf observations are omitted: no search runs
for the former, and the latter have no observed game outcome.

Games run sequentially, with search batching its own leaves. Positive temperature
covers the mixed strategy's support. `load_posterior_agent` expands checkpoints
that lack posterior inputs; those columns still need training before the value
head can use them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fog.encoding.dims import DIMS

LEARNER = "p1"
OPPONENT = "p2"
MASK_PREFIX = "mask_"  # opp_revealed mask families are stored as mask_<family> arrays


# ------------------------------------------------------------------------------------ collection
def executed_posterior_features(res, battle, role: str, action: int) -> np.ndarray | None:
    """Executed-action posterior features at the true team.

    Return None if the action has no public-label column in the true-world fork;
    the caller counts this mismatch rather than filling in features.
    """
    from fog.search.joint import _public_label, posterior_features_np

    label = _public_label(battle, role, action)
    if label not in res.labels:
        return None
    return posterior_features_np(res.posterior[:, res.labels.index(label)], res.true_index)


def collect_search_decisions(
    agent: Any,
    pool: list[str],
    out_dir: str | Path,
    *,
    n_games: int = 32,
    iters: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
    own_source: Any | None = None,
    opponent: Any | None = None,
    tables: Any | None = None,
    max_turns: int = 300,
) -> dict:
    """Play `n_games` of learner-vs-scripted engine self-play with joint-search decisions and
    write `decisions.npz` + `meta.json` under `out_dir`. Returns the meta dict (stats included).

    `own_source` defaults to exact pool enumeration (`EnumeratedPoolStates`);
    pass `SampledOwnStates` to use the learned sampler. `opponent` defaults to a
    seeded `EngineMaxDamagePolicy`.
    """
    import random

    from fog.constants.tables import TABLES
    from fog.env.engine_battle import EngineBattle
    from fog.search.own_states import EnumeratedPoolStates

    if tables is None:
        tables = TABLES
    if own_source is None:
        own_source = EnumeratedPoolStates(pool, tables)
    rng = random.Random(seed)
    agent.eval()

    rows: list[dict[str, np.ndarray]] = []
    stats = {"games": 0, "wins": 0, "decisions": 0,
             "skipped_trivial": 0, "skipped_unlabeled": 0, "sim_failures": 0,
             "failed_games": 0}

    for g in range(int(n_games)):
        battle = EngineBattle(
            rng.choice(pool), rng.choice(pool),
            seed=rng.randrange(2**31), tables=tables, battle_tag=f"postlog-{g}",
        )
        opp_policy = opponent
        if opp_policy is None:
            from fog.env.engine_scripted import EngineMaxDamagePolicy

            opp_policy = EngineMaxDamagePolicy(tables=tables, seed=rng.randrange(2**31))

        game_rows: list[dict[str, np.ndarray]] = []
        try:
            _play_logged_game(battle, agent, own_source, opp_policy, game_rows, stats,
                              iters=iters, temperature=temperature, rng=rng,
                              max_turns=max_turns)
        except Exception as exc:  # noqa: BLE001 - failed game is counted, printed, and bounded
            # A data-generation harness must not lose a 200-game run to one pathological game
            # (e.g. the reveal-census bug ChangeItem/Trick used to trigger) — but silence would
            # hide real defects, so failures are counted, the game's rows are DROPPED (a partial
            # game would bias outcomes toward 0), and a high failure rate still raises below.
            stats["failed_games"] += 1
            print(f"[posterior_log] game {g} failed and was dropped: {exc!r}", flush=True)
            continue

        won = battle.winner
        outcome = 0.0 if won is None else (1.0 if won == LEARNER else -1.0)
        stats["games"] += 1
        stats["wins"] += int(won == LEARNER)
        for r in game_rows:
            r["outcome"] = np.float32(outcome)
            r["game"] = np.int64(g)
        rows.extend(game_rows)

    if stats["failed_games"] > max(1, int(n_games) // 5):
        raise RuntimeError(f"{stats['failed_games']}/{n_games} games failed — that is a defect, "
                           f"not noise; fix it before trusting this log. Stats: {stats}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_rows(rows, out_dir / "decisions.npz")
    meta = {
        "n_games": int(n_games), "iters": int(iters), "temperature": float(temperature),
        "seed": int(seed), "pool_size": len(pool),
        "own_source": type(own_source).__name__, "stats": stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _play_logged_game(battle, agent, own_source, opp_policy, game_rows, stats, *,
                      iters, temperature, rng, max_turns) -> None:
    """One engine game with joint-search learner decisions, appending logged rows in place."""
    from fog.encoding.observation import Observation
    from fog.search.joint import joint_search_action

    while not battle.finished and battle.turn <= max_turns:
        moves: dict[str, str] = {}
        pending = battle.pending_sides()
        if LEARNER in pending:
            res = joint_search_action(
                agent, battle, LEARNER, own_source=own_source, iters=iters,
                temperature=temperature, seed=rng.randrange(2**31),
            )
            stats["sim_failures"] += res.failures
            if not res.labels:
                stats["skipped_trivial"] += 1
            else:
                feats = executed_posterior_features(res, battle, LEARNER, res.action)
                if feats is None:
                    stats["skipped_unlabeled"] += 1
                else:
                    view = battle.view(LEARNER)
                    obs = Observation.from_battle(view, tracker=battle.trackers[LEARNER])
                    game_rows.append(_row(obs, feats, res, battle.turn))
                    stats["decisions"] += 1
                    from fog.encoding.events import record_history_decision

                    record_history_decision(
                        battle.trackers[LEARNER],
                        battle.turn,
                        view,
                        battle.tables,
                        res.action,
                    )
            moves[LEARNER] = battle.engine_action(LEARNER, res.action)
        if OPPONENT in pending:
            opp_action = opp_policy.choose(battle, OPPONENT)
            if getattr(opp_policy, "records_history", False):  # agent-driven opponents ('self')
                from fog.encoding.events import record_history_decision

                record_history_decision(
                    battle.trackers[OPPONENT],
                    battle.turn,
                    battle.view(OPPONENT),
                    battle.tables,
                    opp_action,
                )
            moves[OPPONENT] = battle.engine_action(OPPONENT, opp_action)
        battle.step(moves)


def _row(obs, feats: np.ndarray, res, turn: int) -> dict[str, np.ndarray]:
    from fog.encoding.history import pad_or_truncate_front

    row = {
        "entities": obs.entities.astype(np.float32),
        "entity_ids": obs.entity_ids.astype(np.int64),
        "field": obs.field.astype(np.float32),
        "history": pad_or_truncate_front(obs.history).astype(np.float32),
        "history_ids": pad_or_truncate_front(obs.history_ids).astype(np.int64),
        "history_state": pad_or_truncate_front(obs.history_state).astype(np.float32),
        "history_len": np.int64(min(obs.history_len, DIMS.history_cap)),
        "active_idx": (obs.active_idx if obs.active_idx is not None
                       else np.array([0, 6], dtype=np.int64)),
        "legality_mask": obs.legality_mask.astype(bool),
        "opp_tokens": (obs.opp_revealed or {}).get(
            "tokens", np.full((6, 15), -1, dtype=np.int64)
        ).astype(np.int64),
        "posterior_features": feats.astype(np.float32),
        "search_dist": res.action_dist.astype(np.float32),
        "action": np.int64(res.action),
        "turn": np.int64(turn),
    }
    for fam, m in ((obs.opp_revealed or {}).get("masks") or {}).items():
        row[MASK_PREFIX + fam] = m.astype(bool)
    return row


def _save_rows(rows: list[dict[str, np.ndarray]], path: Path) -> None:
    if not rows:
        raise RuntimeError("no decisions logged — every learner decision was trivial/unlabeled "
                           "(wrong pool, or games ended before any real choice).")
    keys = set(rows[0])
    for r in rows:
        if set(r) != keys:
            raise RuntimeError("inconsistent row keys across decisions (mask families changed "
                               "mid-run — should be impossible with one Tables instance).")
    np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in sorted(keys)})


# --------------------------------------------------------------------------------------- loading
def load_decisions(path: str | Path) -> dict[str, np.ndarray]:
    """decisions.npz -> {key: [N, ...] array}. Accepts the file or its directory."""
    p = Path(path)
    if p.is_dir():
        p = p / "decisions.npz"
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def batch_from_rows(data: dict[str, np.ndarray], idx: np.ndarray, device) -> dict:
    """Rebuild the agent batch dict for a row subset — same tensor contract as
    `engine_runner._stack_obs` (histories are already uniformly padded at log time)."""
    import torch

    def t(a, dtype):
        return torch.as_tensor(np.ascontiguousarray(a[idx]), dtype=dtype, device=device)

    masks = {k[len(MASK_PREFIX):]: t(data[k], torch.bool)
             for k in data if k.startswith(MASK_PREFIX)}
    return {
        "entities": t(data["entities"], torch.float32),
        "entity_ids": t(data["entity_ids"], torch.long),
        "field": t(data["field"], torch.float32),
        "history": t(data["history"], torch.float32),
        "history_ids": (
            t(data["history_ids"], torch.long)
            if "history_ids" in data
            else torch.zeros(
                len(idx), data["history"].shape[1], DIMS.hist_id_fields,
                dtype=torch.long, device=device,
            )
        ),
        "history_state": (
            t(data["history_state"], torch.float32)
            if "history_state" in data
            else torch.zeros(
                len(idx), data["history"].shape[1], DIMS.hist_state,
                dtype=torch.float32, device=device,
            )
        ),
        "history_len": (
            t(data["history_len"], torch.long)
            if "history_len" in data
            else torch.as_tensor(
                np.any(data["history"][idx] != 0, axis=-1).sum(axis=-1),
                dtype=torch.long,
                device=device,
            )
        ),
        "active_idx": t(data["active_idx"], torch.long),
        "legality_mask": t(data["legality_mask"], torch.bool),
        "opp_revealed": {"tokens": t(data["opp_tokens"], torch.long), "masks": masks},
    }


# ------------------------------------------------------------------------------------------ CLI
def _build_opponent(spec: str, agent, cfg: dict, device, temperature: float, seed: int):
    """None (scripted MaxDamage default inside the collector), a policy-arm of the learner
    ('self'), a frozen checkpoint's policy, or another scripted name."""
    if spec == "MaxDamage":
        return None  # the collector's default (fresh-seeded per game)
    from fog.eval.search_gauntlet import ArmPlayer

    if spec == "self":
        return ArmPlayer("policy", agent, temperature=temperature, seed=seed)
    from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES

    if spec in SCRIPTED_ENGINE_POLICIES:
        return SCRIPTED_ENGINE_POLICIES[spec](seed=seed)
    from fog.training.value_finetune import load_posterior_agent

    opp_agent = load_posterior_agent(cfg, spec, device)  # a checkpoint path
    opp_agent.eval()
    return ArmPlayer("policy", opp_agent, temperature=temperature, seed=seed)


def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent

    ap = argparse.ArgumentParser(description="Log legacy one-sided search posteriors.")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    ap.add_argument("--out", type=Path, default=Path("runs/posterior_log"))
    ap.add_argument("--ckpt", type=str, default=None,
                    help="agent checkpoint (posterior-blind ones are expanded); default: fresh")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--opponent", type=str, default="MaxDamage",
                    help="'MaxDamage'/'Random' (scripted), 'self' (the same agent's raw policy — "
                         "self-play data generation), or a checkpoint path (frozen policy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.ckpt:
        from fog.training.value_finetune import load_posterior_agent

        agent = load_posterior_agent(cfg, args.ckpt, device)
    else:
        cfg.setdefault("model", {})["use_posterior_features"] = True
        agent = FogAgent.from_tables(cfg).to(device)
        print("[posterior_log] WARNING: no --ckpt — logging with a FRESH agent (smoke use only; "
              "posterior training pairs should come from a trained checkpoint).", flush=True)

    meta = collect_search_decisions(
        agent, load_pool(args.pool), args.out, n_games=args.games, iters=args.iters,
        temperature=args.temperature, seed=args.seed,
        opponent=_build_opponent(args.opponent, agent, cfg, device, args.temperature,
                                 args.seed + 1),
    )
    print(f"[posterior_log] {meta['stats']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
