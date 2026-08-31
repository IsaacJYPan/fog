"""Collect authoritative live observations for frozen-encoder reconstruction probes.

The acting model chooses every live action.  The collector retains only the ordinary observation
tensors that were available at that decision; heuristic labels, policy outputs, future facts, and
validation teams are not written. Complete pool cycles create disjoint tests: whole held-out teams
and later held-out battles for teams that occur in training. Multiple early cycles may be assigned
to training so representation learning is not limited to one battle per team.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from fog.config import load_config
from fog.training.dagger_heuristic_distill import collect_rows
from fog.training.outcome_heuristic_distill import team_holdout
from fog.training.rollout import _stack_frags

OBSERVATION_KEYS = (
    "entities",
    "entity_ids",
    "field",
    "history",
    "history_ids",
    "history_state",
    "history_len",
    "active_idx",
    "legality_mask",
)


def split_name(
    row: dict,
    team_count: int,
    heldout_teams: set[int],
    training_cycles: int = 1,
) -> str:
    """Assign an entire team or battle to exactly one split."""
    if int(row["team"]) in heldout_teams:
        return "team_holdout"
    return (
        "battle_holdout"
        if int(row["game"]) >= training_cycles * team_count
        else "train"
    )


def _compact_batch(rows: list[dict]) -> dict[str, torch.Tensor]:
    fragments = [
        {key: row["frag"][key] for key in OBSERVATION_KEYS}
        for row in rows
    ]
    if not fragments:
        raise ValueError("cannot save an empty live observation split")
    return _stack_frags(fragments)


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    pool_path = Path(cfg["env"]["team_pool"])
    if pool_path.name.lower().startswith("validation"):
        raise ValueError("validation teams may not be used for encoder probe collection")
    args.out.mkdir(parents=True, exist_ok=False)
    rows, team_count = collect_rows(
        cfg,
        str(args.checkpoint),
        games=args.games or team_count_default(cfg),
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
        retain_public_action_features=False,
    )
    required_games = (args.training_cycles + args.battle_holdout_cycles) * team_count
    if (args.games or team_count_default(cfg)) < required_games:
        raise ValueError(
            "live probe collection requires enough complete pool cycles for training and "
            f"battle holdout ({required_games} games)"
        )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    heldout_teams = team_holdout(team_count, args.team_holdout_fraction, split_seed)
    split_rows: dict[str, list[dict]] = {name: [] for name in ("train", "team_holdout", "battle_holdout")}
    for row in rows:
        split_rows[
            split_name(row, team_count, heldout_teams, args.training_cycles)
        ].append(row)
    for name, selected in split_rows.items():
        torch.save(_compact_batch(selected), args.out / f"{name}.pt")
    counts = Counter(
        split_name(row, team_count, heldout_teams, args.training_cycles)
        for row in rows
    )
    games_by_split = {
        name: len({int(row["game"]) for row in selected})
        for name, selected in split_rows.items()
    }
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "team_pool": str(pool_path),
        "validation_teams_used": False,
        "future_information_saved": False,
        "heuristic_answer_saved": False,
        "model_executed_actions": True,
        "team_count": team_count,
        "games": args.games or 2 * team_count,
        "rows": len(rows),
        "heldout_teams": sorted(heldout_teams),
        "rollout_seed": args.seed,
        "split_seed": split_seed,
        "training_cycles": args.training_cycles,
        "battle_holdout_cycles": args.battle_holdout_cycles,
        "rows_by_split": dict(counts),
        "games_by_split": games_by_split,
        "split_contract": {
            "train": (
                f"first {args.training_cycles} battles for each non-held-out team"
            ),
            "battle_holdout": (
                f"next {args.battle_holdout_cycles} battles for each non-held-out team"
            ),
            "team_holdout": "all battles for teams absent from training",
        },
        "observation_keys": list(OBSERVATION_KEYS),
    }
    (args.out / "collection_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def team_count_default(cfg: dict) -> int:
    from fog.env.teams import load_pool

    return 2 * len(load_pool(cfg["env"]["team_pool"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--team-holdout-fraction", type=float, default=0.1)
    parser.add_argument("--training-cycles", type=int, default=1)
    parser.add_argument("--battle-holdout-cycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261050)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    if (
        args.games < 0
        or args.workers <= 0
        or args.actor_threads <= 0
        or args.training_cycles <= 0
        or args.battle_holdout_cycles <= 0
    ):
        parser.error("games must be nonnegative and worker/cycle counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
