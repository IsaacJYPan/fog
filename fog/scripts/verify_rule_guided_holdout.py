"""Verify rule-guided action parity on an evaluation-only team pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    import torch

    from fog.config import load_config
    from fog.models.agent import FogAgent
    from fog.training.heuristic_distill import collect_rows
    from fog.training.rollout import _stack_frags

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--team-pool", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=21)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    cfg["env"]["team_pool"] = args.team_pool
    rows = collect_rows(
        cfg,
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=2,
        seed=args.seed,
    )

    model = FogAgent.from_tables(cfg).cpu()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    model.eval()
    predictions: list[int] = []
    finite = True
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            batch = _stack_frags([row["frag"] for row in rows[start : start + 512]], device="cpu")
            log_probs = model(batch).log_probs
            finite = finite and bool(torch.isfinite(log_probs[torch.isfinite(log_probs)]).all())
            predictions.extend(log_probs.argmax(-1).tolist())

    targets = [int(row["action"]) for row in rows]
    matches = sum(prediction == target for prediction, target in zip(predictions, targets, strict=True))
    legal = sum(
        bool(np.asarray(row["frag"]["legality_mask"], dtype=np.bool_)[prediction])
        for row, prediction in zip(rows, predictions, strict=True)
    )
    games_seen = sorted({int(row["game"]) for row in rows})
    passed = bool(rows) and matches == len(rows) and legal == len(rows) and finite
    report = {
        "status": "pass" if passed else "fail",
        "checkpoint": args.checkpoint,
        "team_pool": args.team_pool,
        "evaluation_only": True,
        "trained": False,
        "network_server_used": False,
        "direct_in_memory_showdown": True,
        "games": args.games,
        "games_with_rows": len(games_seen),
        "rows": len(rows),
        "rule_action_matches": matches,
        "rule_action_match_rate": matches / len(rows) if rows else None,
        "all_predictions_legal": legal == len(rows),
        "finite_outputs": finite,
        "seed": args.seed,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
