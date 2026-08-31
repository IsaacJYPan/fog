"""Measure independent SimpleHeuristics imitation on unseen-team teacher-played states."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    import torch

    from fog.config import load_config
    from fog.models.agent import FogAgent
    from fog.training.heuristic_distill import collect_rows
    from fog.training.rollout import _stack_frags
    from fog.training.simple_imitation import assert_teacher_answer_is_not_an_input

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260841)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = collect_rows(
        cfg,
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
    )
    assert_teacher_answer_is_not_an_input(cfg, rows)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = FogAgent.from_tables(cfg).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)
    )
    model.eval()
    correct = legal = 0
    nll_sum = 0.0
    by_opponent: dict[str, Counter] = {}
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            part = rows[start : start + args.batch_size]
            batch = _stack_frags([row["frag"] for row in part], device=device)
            labels = torch.as_tensor(
                [row["action"] for row in part], dtype=torch.long, device=device
            )
            output = model(batch)
            predictions = output.log_probs.argmax(-1)
            nll = -output.log_probs.gather(1, labels[:, None]).squeeze(1)
            if not torch.isfinite(nll).all():
                raise FloatingPointError("non-finite unseen-team imitation score")
            nll_sum += float(nll.sum())
            matches = predictions == labels
            correct += int(matches.sum())
            legal += int(batch["legality_mask"].gather(1, predictions[:, None]).sum())
            for row, matched in zip(part, matches.cpu().tolist(), strict=True):
                stats = by_opponent.setdefault(str(row["opponent"]), Counter())
                stats["rows"] += 1
                stats["correct"] += int(matched)
    report = {
        "status": "complete",
        "checkpoint": args.checkpoint,
        "games": args.games,
        "rows": len(rows),
        "unseen_validation_teams": True,
        "validation_used_for_training": False,
        "teacher_answer_available_to_model": False,
        "action_features_present": False,
        "exact_action_match_rate": correct / len(rows),
        "nll": nll_sum / len(rows),
        "all_predictions_legal": legal == len(rows),
        "by_opponent": {
            name: {
                "rows": stats["rows"],
                "correct": stats["correct"],
                "match_rate": stats["correct"] / stats["rows"],
            }
            for name, stats in sorted(by_opponent.items())
        },
        "seed": args.seed,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
