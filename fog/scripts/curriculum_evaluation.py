"""Verify fixed online-pilot exports before separate full-game CPU evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from fog.scripts.teacher_pilot_queue import verified_inventory


def resolve(source, input_root: Path, destination: Path):
    expected = source["experiment"]
    if (expected.get("format") != "fog.curriculum-pilot" or expected.get("version") != 1
            or expected.get("device") != "cuda" or expected.get("stage") != "train"):
        raise ValueError("CPU screen requires a fixed online GPU pilot")
    candidates = []
    for path in input_root.rglob("fog-experiment-output.zip"):
        with zipfile.ZipFile(path) as archive:
            if "experiment.json" in archive.namelist() and json.loads(archive.read("experiment.json")).get("id") == expected["id"]:
                candidates.append(path)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Ambiguous online-pilot export")
    path = candidates[0]
    with zipfile.ZipFile(path) as archive:
        inventory = verified_inventory(archive)
        if json.loads(archive.read("experiment.json")) != expected:
            raise ValueError("GPU source descriptor differs from the frozen evaluation input")
        pipeline = json.loads(archive.read("pipeline_state.json"))
        if pipeline.get("status") != "complete" or pipeline.get("error") is not None or pipeline.get("returncode") != 0:
            raise ValueError("Online GPU pilot did not complete successfully")
        report = json.loads(archive.read("training/RESULTS.json"))
        if (not report["complete"] or report["device"] != "cuda" or report["batches"] != expected["max_batches"]
                or report["optimized_positions"] != expected["optimized_positions"]
                or report["optimizer_steps"] != expected["optimizer_steps"] or report["condition"] != expected["condition"]
                or report["identity"] != source["expected_training_identity"]):
            raise ValueError("GPU exposure, condition or training identity differs")
        for name in ("starting_contract", "final_contract"):
            contract = report[name]
            if (not contract["passed"] or not contract["cpu_cuda_verified"] or not 0 < contract["tolerance"] <= 1e-3
                    or not 0 <= contract["cpu_cuda_max_error"] <= 1e-3):
                raise ValueError("Online GPU probability contract failed or is missing")
        if report["checkpoint_sha256"] != inventory["training/latest.pt"] or report["full_state_sha256"] != inventory["training/latest.state.pt"]:
            raise ValueError("Online GPU checkpoint/state hashes differ")
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("training/latest.pt", "training/RESULTS.json", "experiment.json"):
            archive.extract(name, destination)
        with path.open("rb") as stream:
            export_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {"source_export_sha256": export_sha256, "checkpoint_sha256": report["checkpoint_sha256"],
                   "full_state_sha256": report["full_state_sha256"], "batches": report["batches"],
                   "verified_files": len(inventory), "condition": report["condition"],
                   "initial_sha256": expected["initial_sha256"], "automatic_promotion": False}
        (destination / "SOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return destination / "training/latest.pt"
