"""Run a fixed CPU development comparison and optional held-out belief benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import threading
import time

from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256
from fog.pretrained.runtime import native_environment
from fog.pretrained.evaluation import SCRIPTED_OPPONENTS
from fog.scripts.kaggle_results import write_json
from fog.scripts.pretrained_compare import compare_evaluations, compare_scripted_evaluations
from fog.scripts.pretrained_kaggle_worker import execute


def validate_recipe(recipe):
    if recipe.get("version") != 1 or recipe.get("stage") != "development":
        raise ValueError("Only the separately frozen development evaluation is supported")
    if set(recipe.get("conditions", {})) != {"ordinary", "joint"}:
        raise ValueError("Require both matched conditions in the same CPU job")
    for key, minimum in (("games", 256), ("smoke_games", 8)):
        if type(recipe.get(key)) is not int or recipe[key] < minimum or recipe[key] % 4:
            raise ValueError(f"{key} requires complete crossed blocks and at least {minimum} games")
    if type(recipe.get("seed")) is not int:
        raise ValueError("Freeze the evaluation seed")
    if "pilot_only" in recipe and type(recipe["pilot_only"]) is not bool:
        raise ValueError("Declare pilot-only execution as a boolean")
    if "diagnostic_only" in recipe and type(recipe["diagnostic_only"]) is not bool:
        raise ValueError("Declare diagnostic-only execution as a boolean")
    if recipe.get("diagnostic_only") and not (recipe.get("heuristics") or recipe.get("belief_diagnostic")):
        raise ValueError("Diagnostic-only execution requires a declared diagnostic")
    if recipe.get("diagnostic_only") and recipe.get("pilot_only"):
        raise ValueError("A full diagnostic job cannot also be a smoke-only queue pilot")
    heuristics = recipe.get("heuristics")
    if heuristics:
        opponents = heuristics.get("opponents", [])
        if not isinstance(opponents, list) or not opponents or any(type(value) is not str for value in opponents) or len(set(opponents)) != len(opponents) or not set(opponents).issubset(SCRIPTED_OPPONENTS):
            raise ValueError("Declare unique supported scripted opponents")
        for key, minimum in (("games", 256), ("smoke_games", 8)):
            if type(heuristics.get(key)) is not int or heuristics[key] < minimum or heuristics[key] % 4:
                raise ValueError("Scripted evaluation requires complete crossed game budgets")
    diagnostic = recipe.get("belief_diagnostic")
    if diagnostic:
        from fog.scripts.pretrained_belief_diagnose import validate_diagnostic

        validate_diagnostic(diagnostic)
    models = recipe.get("matched", {}).get("models")
    if models != [recipe["conditions"][arm].get("native_sha256") for arm in ("ordinary", "joint")]:
        raise ValueError("Candidates must match the audited training pair")
    for row in recipe["conditions"].values():
        if not row.get("native") or not row.get("reference"):
            raise ValueError("Candidate and its CPU probability reference are required")
    belief = recipe.get("belief")
    if belief:
        if (
            type(belief.get("games")) is not int
            or belief["games"] < 1
            or type(belief.get("draws")) is not int
            or belief["draws"] < 2
        ):
            raise ValueError("Belief diagnostics require a fixed cohort and at least two draws")
        if not belief.get("fresh") or any(
            not row.get("auxiliary") for row in recipe["conditions"].values()
        ):
            raise ValueError("Belief diagnostics require the cohort and both auxiliary exports")


def run_job(recipe, native_python, out, max_seconds):
    validate_recipe(recipe)
    root = Path.cwd()
    original = root / "inputs/kakuna34.pt"
    if sha256(original) != KAKUNA_34_SHA256:
        raise ValueError("Evaluation baseline must remain unchanged Kakuna34")
    for row in recipe["conditions"].values():
        if sha256(root / row["native"]) != row["native_sha256"]:
            raise ValueError("Candidate hash differs from the audited pair")
    out.mkdir(parents=True, exist_ok=True)
    descriptor = out / "INPUTS.json"
    if descriptor.exists() and json.loads(descriptor.read_text()) != recipe:
        raise ValueError("Cannot change a running evaluation recipe")
    write_json(descriptor, recipe)
    deadline = time.monotonic() + max_seconds
    state = {
        "status": "running",
        "pid": os.getpid(),
        "phase": "contracts",
        "started_at": datetime.now(UTC).isoformat(),
        "operations": {},
        "gpu_requested": False,
        "automatic_promotion": False,
    }
    lock = threading.Lock()

    def update(**changes):
        with lock:
            state.update(changes, updated_at=datetime.now(UTC).isoformat())
            write_json(out / "STATE.json", state)

    def child(label, command, *, native=False, games_dir=None):
        def progress():
            row = {
                "operation": label,
                "saved_games": len(list(games_dir.glob("game-*.json.gz"))) if games_dir else 0,
            }
            with lock:
                state["operations"][label] = row
                state["updated_at"] = datetime.now(UTC).isoformat()
                write_json(out / "STATE.json", state)
            return row

        progress()
        environment = native_environment(root) if native else os.environ.copy()
        execute(
            command,
            dict(environment, PYTHONUNBUFFERED="1", PYTHONUTF8="1"),
            out / (label.replace("/", "-") + ".log"),
            deadline - time.monotonic(),
            isolated=False,
            progress=progress,
        )
        progress()

    def contract(name, checkpoint, reference, *, trained):
        destination = out / "contracts" / name
        command = [
            str(native_python),
            "-m",
            "fog.scripts.pretrained_contract",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "inputs/contract/inputs.pt",
            "--reference",
            str(reference),
            "--device",
            "cpu",
            "--out",
            str(destination),
        ]
        if trained:
            command.append("--trained")
        child("contract/" + name, command, native=True)
        report = json.loads((destination / "contract.json").read_text())
        if (
            not report["passed"]
            or not report["weights_unchanged"]
            or report["checkpoint_sha256"] != sha256(checkpoint)
        ):
            raise ValueError("Remote CPU probability contract failed: " + name)

    def evaluate(condition, phase, games, *, scripted=None):
        destination = out / phase / condition
        candidate = str(original) if condition == "original" else recipe["conditions"][condition]["native"]
        command = [
            sys.executable, "-m", "fog.scripts.pretrained_evaluate",
            "--candidate", candidate, "--original", str(original),
            "--native-python", str(native_python), "--pools", "inputs/pools",
            "--out", str(destination), "--games", str(games),
            "--seed", str(recipe["seed"]), "--stage", "development",
        ]
        if scripted:
            command += ["--scripted-opponent", scripted]
        child(
            phase + "/" + condition,
            command,
            games_dir=destination,
        )
        report_path = destination / "RESULTS.json"
        report = json.loads(report_path.read_text())
        if report["failures"] or report["unresolved"] or len(report["rows"]) != games:
            raise ValueError("Evaluation failures must be diagnosed: " + condition + "/" + phase)
        return report_path

    def benchmark(condition):
        row, belief = recipe["conditions"][condition], recipe["belief"]
        destination = out / "belief" / condition
        child(
            "belief/" + condition,
            [
                str(native_python),
                "-m",
                "fog.scripts.pretrained_belief_benchmark",
                "--native",
                row["native"],
                "--auxiliary",
                row["auxiliary"],
                "--fresh",
                belief["fresh"],
                "--pools",
                "inputs/pools",
                "--out",
                str(destination),
                "--games",
                str(belief["games"]),
                "--draws",
                str(belief["draws"]),
            ],
            native=True,
        )
        return json.loads((destination / "RESULTS.json").read_text())

    update()
    try:
        contract("original", original, root / "inputs/contract/cpu/probabilities.pt", trained=False)
        for condition, row in recipe["conditions"].items():
            contract(condition, root / row["native"], root / row["reference"], trained=True)
        phases = (("smoke", recipe["smoke_games"]),)
        short_comparison = recipe.get("pilot_only", False) or recipe.get("diagnostic_only", False)
        if not short_comparison:
            phases += (("full", recipe["games"]),)
        for phase, games in phases:
            update(phase=phase)
            with ThreadPoolExecutor(max_workers=2) as executor:
                paths = list(
                    executor.map(lambda arm: evaluate(arm, phase, games), ("ordinary", "joint"))
                )
        comparison = (
            None
            if short_comparison
            else compare_evaluations(recipe["matched"], paths[0], paths[1])
        )
        result = {
            "complete": True,
            "development": comparison,
            "belief": None,
            "gpu_requested": False,
            "automatic_promotion": False,
            "pilot_only": recipe.get("pilot_only", False),
            "diagnostic_only": recipe.get("diagnostic_only", False),
            "heuristics": None,
            "belief_diagnostic": None,
        }
        if recipe.get("belief_diagnostic"):
            update(phase="belief_diagnostic")
            diagnostic_path = out / "belief-diagnostic-inputs.json"
            write_json(diagnostic_path, recipe["belief_diagnostic"])
            child("belief_diagnostic", [
                str(native_python), "-m", "fog.scripts.pretrained_belief_diagnose",
                "--config", str(diagnostic_path), "--pools", "inputs/pools",
                "--out", str(out / "belief-diagnostic"),
            ], native=True)
            result["belief_diagnostic"] = json.loads((out / "belief-diagnostic/RESULTS.json").read_text())
        if recipe.get("heuristics"):
            result["heuristics"] = {}
            for opponent in recipe["heuristics"]["opponents"]:
                for phase, games in (("smoke", recipe["heuristics"]["smoke_games"]), ("full", recipe["heuristics"]["games"])):
                    label = "heuristics/" + opponent + "/" + phase
                    update(phase=label)
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        arms = ("original", "ordinary", "joint")
                        paths = list(executor.map(lambda arm: evaluate(arm, label, games, scripted=opponent), arms))
                result["heuristics"][opponent] = compare_scripted_evaluations(
                    recipe["matched"], dict(zip(arms, paths)), opponent
                )
                write_json(out / "HEURISTICS.json", result["heuristics"])
        if recipe.get("belief"):
            update(phase="belief")
            with ThreadPoolExecutor(max_workers=2) as executor:
                ordinary, joint = list(executor.map(benchmark, ("ordinary", "joint")))
            for key in (
                "input_games",
                "development_pool_sha256",
                "training_bank_sha256",
                "games",
                "contexts",
                "draws_per_context",
                "seed",
            ):
                if ordinary[key] != joint[key]:
                    raise ValueError("Belief benchmark inputs differ: " + key)
            if not all(row["sampling_trace_finite"] for row in (ordinary, joint)):
                raise ValueError("Belief sampling returned nonfinite traces")
            result["belief"] = {"ordinary": ordinary, "joint": joint, "search_connected": False}
        if sha256(original) != KAKUNA_34_SHA256 or any(
            sha256(root / row["native"]) != row["native_sha256"]
            for row in recipe["conditions"].values()
        ):
            raise ValueError("Evaluation changed a protected checkpoint")
        write_json(out / "RESULTS.json", result)
        update(status="complete", completed_at=datetime.now(UTC).isoformat())
        return result
    except Exception as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args()
    run_job(json.loads(args.config.read_text()), args.native_python, args.out, args.max_seconds)


if __name__ == "__main__":
    main()
