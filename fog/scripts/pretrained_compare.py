"""Check matched optimization budgets and compare ordinary/joint development play."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fog.eval.stats import paired_block_interval
from fog.pretrained.evaluation import matched_update_budget
from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256


def matched_reports(
    ordinary_experiment, joint_experiment, ordinary_result, joint_result, *, update=None
):
    experiments = [json.loads(path.read_text()) for path in (ordinary_experiment, joint_experiment)]
    if [row.get("condition") for row in experiments] != ["ordinary", "joint"]:
        raise ValueError("Require the ordinary and joint frozen training packages")
    for key in (
        "phase",
        "matched_recipe_sha256",
        "matched_inputs_sha256",
        "timeout_seconds",
        "updates",
    ):
        if experiments[0][key] != experiments[1][key]:
            raise ValueError("Unmatched training source/data/optimization budget: " + key)
    results = [
        json.loads((root / "training/RESULTS.json").read_text())
        for root in (ordinary_result, joint_result)
    ]
    for root, experiment in zip((ordinary_result, joint_result), experiments):
        if json.loads((root / "experiment.json").read_text()) != experiment:
            raise ValueError(
                "Returned training outputs do not belong to the declared frozen package"
            )
    completed = [row["completed_updates"] for row in results]
    if update is None:
        if len(set(completed)) != 1:
            raise ValueError("Unequal completed budgets; choose a retained common update anchor")
        update = completed[0]
    if update < 1 or any(value < update for value in completed):
        raise ValueError("Both arms must have completed the declared common update count")
    paths, throughput = [], []
    schedules = []
    for root in (ordinary_result, joint_result):
        path = root / f"training/native_{update:06d}.pt"
        if not path.exists():
            raise ValueError("The selected common native export was not retained")
        rows = [
            json.loads(line) for line in (root / "training/metrics.jsonl").read_text().splitlines()
        ]
        rows = [row for row in rows if row["update"] <= update]
        if [row["update"] for row in rows] != list(range(1, update + 1)):
            raise ValueError("Missing or repeated optimization metrics")
        schedules.append([(row["schedule_sha256"], row["counts"]) for row in rows])
        throughput.append(sum(row["seconds"] for row in rows) / update)
        paths.append(path)
    if schedules[0] != schedules[1]:
        raise ValueError("Ordinary and joint runs used different data or supervised positions")
    return {
        "updates": update,
        "models": [sha256(path) for path in paths],
        "mean_update_seconds": throughput,
        "matched_recipe_sha256": experiments[0]["matched_recipe_sha256"],
        "matched_inputs_sha256": experiments[0]["matched_inputs_sha256"],
    }


def compare_evaluations(pair, ordinary_path, joint_path):
    reports = [json.loads(path.read_text()) for path in (ordinary_path, joint_path)]
    for report, digest in zip(reports, pair["models"]):
        protocol = report["protocol"]
        if (
            protocol["stage"] != "development"
            or protocol["foulplay"]
            or protocol["original_sha256"] != KAKUNA_34_SHA256
            or protocol["candidate_sha256"] != digest
        ):
            raise ValueError(
                "Use matched development games against unchanged Kakuna for the selected update"
            )
        if report["failures"] or report["unresolved"]:
            raise ValueError("Failed or incomplete development comparisons require diagnosis")
    for key in (
        "pool_sha256",
        "interface",
        "sources_sha256",
        "seed",
        "temperature",
        "deadline_seconds",
    ):
        if reports[0]["protocol"][key] != reports[1]["protocol"][key]:
            raise ValueError("Unmatched development inference protocol: " + key)
    ordinary, joint = [row["rows"] for row in reports]
    if len(ordinary) < 256 or len(ordinary) % 4 or len(ordinary) != len(joint):
        raise ValueError("Require at least 256 matched crossed development games per arm")
    if any(
        a["index"] != b["index"] or a["schedule"] != b["schedule"] for a, b in zip(ordinary, joint)
    ):
        raise ValueError("Development game schedules differ")
    delta = paired_block_interval(
        [b["score"] - a["score"] for a, b in zip(ordinary, joint)],
        bounds=(-1.0, 1.0),
        seed=20261009,
    )
    return {
        "joint_minus_ordinary": asdict(delta),
        "auxiliary_combination_supported_on_development": delta.low > 0,
        "attribution": "The matched contrast tests belief plus outcome additions together. Belief-specific attribution requires an outcome-only ablation. Development intervals are exploratory and do not establish final playing strength.",
        "automatic_promotion": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-experiment", type=Path, required=True)
    parser.add_argument("--joint-experiment", type=Path, required=True)
    parser.add_argument("--ordinary-result", type=Path, required=True)
    parser.add_argument("--joint-result", type=Path, required=True)
    parser.add_argument("--ordinary-evaluation", type=Path)
    parser.add_argument("--joint-evaluation", type=Path)
    parser.add_argument("--update", type=int)
    parser.add_argument("--hours-per-arm", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new comparison identity")
    pair = matched_reports(
        args.ordinary_experiment,
        args.joint_experiment,
        args.ordinary_result,
        args.joint_result,
        update=args.update,
    )
    pair["proposed_equal_updates"] = matched_update_budget(
        pair["mean_update_seconds"], hours_per_arm=args.hours_per_arm
    )
    pair["automatic_promotion"] = False
    if bool(args.ordinary_evaluation) != bool(args.joint_evaluation):
        raise ValueError("Supply both development evaluation reports")
    if args.ordinary_evaluation:
        pair["development"] = compare_evaluations(
            pair, args.ordinary_evaluation, args.joint_evaluation
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pair, indent=2) + "\n")
    print(json.dumps(pair, indent=2))


if __name__ == "__main__":
    main()
