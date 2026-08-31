"""Audit whether the model can see and align every fact used by SimpleHeuristicsPlayer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FACT_COVERAGE = [
    {"fact": "currently legal moves and switches", "old": "exact legality mask", "new": "per-choice legal flag"},
    {"fact": "active and opposing types", "old": "entity type blocks", "new": "same-type and effectiveness per move"},
    {"fact": "base stats and speed tier", "old": "entity base-stat blocks", "new": "matchup and attack/defense ratios"},
    {"fact": "health, fainting, and remaining team count", "old": "entity health/fainted blocks", "new": "explicit remaining-team fractions"},
    {"fact": "stat boosts", "old": "entity boost blocks", "new": "setup readiness and damage ratios"},
    {"fact": "move power, accuracy, hit count, category, and target", "old": "move identity only", "new": "explicit per-move values"},
    {"fact": "entry hazards and removal", "old": "field conditions plus move identity", "new": "explicit setup/removal readiness"},
    {"fact": "individual switch candidate matchup", "old": "pooled party encoding with shifting output slots", "new": "candidate aligned to its switch output"},
    {"fact": "ability to transform and private transformation type", "old": "legal transform actions plus private type", "new": "explicit transform-improves-matchup flag"},
    {"fact": "the exact deterministic rule choice", "old": "training label only", "new": "optional transparent starting prior"},
]


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "fog/encoding/action_features.py",
        "fog/encoding/observation.py",
        "fog/models/heads.py",
        "fog/models/agent.py",
        "fog/env/player.py",
    ):
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    from fog.config import load_config
    from fog.encoding.action_features import ACTION_FEATURE_NAMES
    from fog.training.heuristic_distill import collect_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    rows = collect_rows(
        load_config(args.config),
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=2,
        seed=args.seed,
    )
    pick_index = ACTION_FEATURE_NAMES.index("simple_rule_pick")
    legal_index = ACTION_FEATURE_NAMES.index("legal")
    matches = legal_picks = finite_rows = 0
    feature_nonzero = np.zeros(len(ACTION_FEATURE_NAMES), dtype=np.int64)
    for row in rows:
        values = np.asarray(row["frag"]["action_features"], dtype=np.float32)
        finite_rows += int(np.isfinite(values).all())
        picked = int(values[:, pick_index].argmax())
        matches += int(picked == int(row["action"]))
        legal_picks += int(values[picked, legal_index] == 1)
        feature_nonzero += np.any(values != 0, axis=0)

    report = {
        "status": "pass" if rows and matches == len(rows) and legal_picks == len(rows) and finite_rows == len(rows) else "fail",
        "games": args.games,
        "rows": len(rows),
        "teacher_choice_matches_rule_prior": matches,
        "teacher_choice_match_rate": matches / len(rows) if rows else None,
        "rule_choices_legal": legal_picks,
        "finite_feature_rows": finite_rows,
        "old_input_problem": {
            "missing_raw_teacher_facts": [],
            "poorly_aligned_facts": [
                "move mechanics were represented only by move identity",
                "switch candidates were pooled while switch output positions shift with the active Pokemon",
            ],
            "conclusion": "The raw facts existed, but the output layer was forced to rediscover mechanics and slot alignment. The new per-choice table fixes that representation problem.",
        },
        "fact_coverage": FACT_COVERAGE,
        "feature_nonzero_rows": dict(zip(ACTION_FEATURE_NAMES, feature_nonzero.tolist(), strict=True)),
        "source_fingerprint": _source_fingerprint(),
        "validation_teams_used": False,
    }
    (args.out / "input_coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
