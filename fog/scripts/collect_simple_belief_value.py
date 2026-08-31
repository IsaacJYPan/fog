"""Collect leak-free SimpleHeuristics-vs-SimpleHeuristics belief/value supervision.

FogPlayer records the acting side's ordinary player-visible observation, while an action override
and the opposing controller both execute poke-env's SimpleHeuristics policy.  Submitted teams are
known to the simulator, so the acting player's own private set is an exact input and the opposing
submitted set is an exact belief label.  Neither private truth is exposed through the opponent's
public observation fields.  One compressed replay-compatible shard is written per battle so train
and holdout splitting remains battle- and team-level.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np


def _game_arrays(game) -> dict[str, np.ndarray]:
    from fog.constants.schema import SET_N_ATTRS
    from fog.encoding.dims import DIMS
    from fog.training.rollout import _stack_frags

    if game.outcome is None or game.belief_targets is None:
        raise RuntimeError("completed simulator game lacks outcome or exact opponent truth")
    n = len(game.frags)
    if not n:
        raise RuntimeError("completed simulator game has no decisions")

    stacked = _stack_frags(game.frags)

    def array(key: str, *, nested: str | None = None) -> np.ndarray:
        value = stacked[key][nested] if nested is not None else stacked[key]
        return value.detach().cpu().numpy()

    def fixed_history(key: str, dtype) -> np.ndarray:
        values = array(key).astype(dtype)
        result = np.zeros(
            (values.shape[0], DIMS.history_cap, values.shape[-1]), dtype=dtype
        )
        kept = min(values.shape[1], DIMS.history_cap)
        if kept:
            result[:, -kept:] = values[:, -kept:]
        return result

    truth = np.asarray(game.belief_targets, dtype=np.int64)
    if truth.shape != (6, SET_N_ATTRS):
        raise ValueError(f"opponent truth has shape {truth.shape}, expected (6,{SET_N_ATTRS})")
    return {
        "entities": array("entities").astype(np.float32),
        "entity_ids": array("entity_ids").astype(np.int64),
        "field": array("field").astype(np.float32),
        "history": fixed_history("history", np.float32),
        "history_ids": fixed_history("history_ids", np.int64),
        "history_state": fixed_history("history_state", np.float32),
        "history_len": array("history_len").astype(np.int64),
        "active_idx": array("active_idx").astype(np.int64),
        "legality_mask": array("legality_mask").astype(bool),
        "opp_tokens": array("opp_revealed", nested="tokens").astype(np.int64),
        "action": np.asarray(game.actions, dtype=np.int64),
        "belief_targets": np.repeat(truth[None], n, axis=0),
        # Mirror observations are intentionally absent in this stage; never manufacture a target.
        "self_belief_targets": np.full((n, 6, SET_N_ATTRS), -1, dtype=np.int64),
        "outcome": np.full(n, float(game.outcome), dtype=np.float32),
        "action_features": array("action_features").astype(np.float32),
        # Actions are stored only to satisfy the common shard schema. This stage has action_coef=0.
        "policy_valid": np.zeros(n, dtype=bool),
        "own_set_provenance": np.full((n, 6, SET_N_ATTRS), 2, dtype=np.uint8),
        "opponent_set_provenance": np.zeros((n, 6, SET_N_ATTRS), dtype=np.uint8),
        "legal_provenance": np.full((n, 14), 2, dtype=np.uint8),
        "action_feature_provenance": np.full((n, 14), 2, dtype=np.uint8),
        "imputation_id": np.full(n, -1, dtype=np.int16),
    }


def collect(args: argparse.Namespace) -> dict:
    import polars as pl
    import torch

    from fog.config import load_config
    from fog.constants.tables import TABLES
    from fog.env.actions import map_order_to_action
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import (
        _scripted_order,
        bind_fog_player,
        play_local_battle,
    )
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import load_pool, to_packed
    from fog.models.agent import FogAgent
    from fog.training.direct_parallel import WorkerGame, partition_games, run_workers
    from fog.training.rollout import RolloutBuffer

    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    shard_dir = args.out / "shards"
    shard_dir.mkdir()
    cfg = load_config(args.config)
    pool = load_pool(cfg["env"]["team_pool"])
    packed = [to_packed(paste) for paste in pool]
    truths = [truth_tokens_from_paste(paste, TABLES) for paste in pool]

    split_rng = random.Random(args.seed ^ 0xA11CE)
    team_order = list(range(len(pool)))
    split_rng.shuffle(team_order)
    holdout_count = max(1, round(len(pool) * args.holdout_team_fraction))
    holdout_teams = set(team_order[:holdout_count])
    train_teams = [index for index in range(len(pool)) if index not in holdout_teams]
    held_teams = sorted(holdout_teams)
    if len(train_teams) < 2 or len(held_teams) < 2:
        raise ValueError("team split requires at least two train and two holdout teams")

    schedule_rng = random.Random(args.seed)
    jobs = []
    if args.matchup_repeats:
        if args.games % args.matchup_repeats:
            raise ValueError("games must be divisible by matchup-repeats")
        repeat_teams = train_teams if args.matchup_repeat_pool == "train" else held_teams
        ordered_pairs = [
            (p1, p2) for p1 in repeat_teams for p2 in repeat_teams if p1 != p2
        ]
        schedule_rng.shuffle(ordered_pairs)
        pair_count = args.games // args.matchup_repeats
        if pair_count > len(ordered_pairs):
            raise ValueError("requested more repeated matchup pairs than the holdout pool provides")
        game_number = 0
        for p1_index, p2_index in ordered_pairs[:pair_count]:
            for _ in range(args.matchup_repeats):
                jobs.append(
                    WorkerGame(
                        worker_id=-1,
                        game_number=game_number,
                        seed=schedule_rng.getrandbits(256),
                        p1_index=p1_index,
                        p2_index=p2_index,
                    )
                )
                game_number += 1
    else:
        for game_number in range(args.games):
            is_holdout = game_number % args.holdout_every == 0
            candidates = held_teams if is_holdout else train_teams
            jobs.append(
                WorkerGame(
                    worker_id=-1,
                    game_number=game_number,
                    seed=schedule_rng.getrandbits(256),
                    p1_index=candidates[game_number % len(candidates)],
                    p2_index=candidates[schedule_rng.randrange(len(candidates))],
                )
            )
    assignments = partition_games(jobs, args.workers)
    showdown_root = cfg.get("env", {}).get("showdown_root")

    def worker(worker_id: int, assigned: list[WorkerGame]) -> list[dict]:
        torch.set_num_threads(max(1, args.actor_threads))
        agent = FogAgent.from_tables(cfg).cpu()
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
        agent.eval()
        actor = bind_fog_player(FogPlayer(agent, temperature=0.0))
        actor.record_mirror = False
        buffer = RolloutBuffer()
        actor.attach_buffer(buffer)
        current = {"mine": None, "opponent": None}
        actor.truth_provider = lambda _battle: (
            None if current["opponent"] is None else truths[current["opponent"]].copy()
        )
        actor.private_state_truth_provider = lambda _battle: (
            None if current["mine"] is None else truths[current["mine"]].copy()
        )
        action_rng = random.Random(args.seed ^ ((worker_id + 1) * 0x9E3779B1))

        def simple_override(battle, _observation, _output) -> int:
            return map_order_to_action(
                battle, _scripted_order("simpleheuristics", battle, action_rng)
            )

        actor.action_override = simple_override
        client_kwargs = {"showdown_root": showdown_root} if showdown_root else {}
        rows = []
        with ShowdownDirectClient(**client_kwargs) as client:
            for completed, job in enumerate(assigned, start=1):
                current["mine"], current["opponent"] = job.p1_index, job.p2_index
                result = play_local_battle(
                    client,
                    {
                        "formatid": cfg["env"].get("battle_format", "gen9ou"),
                        "seed": f"sodium,{job.seed:064x}",
                        "strictChoices": True,
                        "p1": {"name": f"SimpleA{worker_id}", "team": packed[job.p1_index]},
                        "p2": {"name": "SimpleB", "team": packed[job.p2_index]},
                    },
                    actor,
                    "simpleheuristics",
                    seed=job.seed & (2**31 - 1),
                    battle_tag=job.battle_tag,
                    max_turns=args.max_turns,
                )
                if result.truncated:
                    raise RuntimeError(f"Simple-vs-Simple game {job.game_number} truncated")
                if len(buffer._games) != 1:
                    raise RuntimeError("expected exactly one finalized simulator game")
                finalized = next(iter(buffer._games.values()))
                arrays = _game_arrays(finalized)
                shard = shard_dir / f"game_{job.game_number:06d}.npz"
                np.savez_compressed(shard, **arrays)
                rows.append(
                    {
                        "id": f"simple-{job.game_number:06d}",
                        "shard": str(shard.resolve()),
                        "n_rows": len(arrays["outcome"]),
                        "split": "holdout" if job.p1_index in holdout_teams else "train",
                        "p1_team": int(job.p1_index),
                        "p2_team": int(job.p2_index),
                        "outcome": float(finalized.outcome),
                        "turns": int(result.turns),
                    }
                )
                buffer.clear()
                if completed <= 2 or completed % 25 == 0:
                    print(
                        f"[simple-bv-collect] worker={worker_id} games={completed}/"
                        f"{len(assigned)} rows={sum(row['n_rows'] for row in rows)}",
                        flush=True,
                    )
        return rows

    started = time.monotonic()
    nested = run_workers(assignments, worker)
    rows = sorted(
        [row for worker_rows in nested for row in worker_rows], key=lambda row: row["id"]
    )
    manifest = pl.DataFrame(rows)
    manifest.write_parquet(args.out / "manifest.parquet")
    manifest.filter(pl.col("split") == "train").write_parquet(args.out / "train_manifest.parquet")
    manifest.filter(pl.col("split") == "holdout").write_parquet(
        args.out / "holdout_manifest.parquet"
    )
    report = {
        "status": "complete",
        "games": len(rows),
        "rows": int(sum(row["n_rows"] for row in rows)),
        "train_games": int(sum(row["split"] == "train" for row in rows)),
        "holdout_games": int(sum(row["split"] == "holdout" for row in rows)),
        "train_teams": len(train_teams),
        "holdout_teams": len(held_teams),
        "holdout_team_indices": held_teams,
        "both_players": "SimpleHeuristics",
        "belief_truth": "exact_submitted_opponent_set",
        "own_private_input": "exact_submitted_player_set",
        "future_information_in_inputs": False,
        "validation_teams_used": False,
        "policy_supervision_enabled": False,
        "matchup_repeats": int(args.matchup_repeats),
        "matchup_repeat_pool": args.matchup_repeat_pool,
        "repeated_matchup_pairs": (
            int(args.games // args.matchup_repeats) if args.matchup_repeats else 0
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261155)
    parser.add_argument("--holdout-team-fraction", type=float, default=0.10)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=160)
    parser.add_argument(
        "--matchup-repeats", type=int, default=0,
        help="If positive, collect this many RNG repetitions of each selected heldout team pair",
    )
    parser.add_argument(
        "--matchup-repeat-pool", choices=("holdout", "train"), default="holdout",
        help="Team split used for repeated matchup pairs; holdout preserves audit behavior",
    )
    args = parser.parse_args()
    if not 0.0 < args.holdout_team_fraction < 0.5:
        raise ValueError("holdout team fraction must be in (0,0.5)")
    if args.holdout_every < 2:
        raise ValueError("holdout-every must be at least2")
    if args.matchup_repeats < 0:
        raise ValueError("matchup-repeats must be nonnegative")
    collect(args)


if __name__ == "__main__":
    main()
