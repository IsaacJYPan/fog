"""Verify fixed online-pilot exports before separate full-game CPU evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

from fog.scripts.teacher_pilot_queue import verified_inventory


def fixed_checkpoints(descriptor, work: Path):
    """Resolve hash-bound parent/control models without allowing output path labels."""
    entries = descriptor.get("fixed_checkpoints", {
        "starting": {"path": "inputs/bc1000.pt", "sha256": descriptor["initial_sha256"]},
    })
    if not isinstance(entries, dict) or not 1 <= len(entries) <= 2:
        raise ValueError("Require one or two fixed control checkpoints")
    result = {}
    for label, entry in entries.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", label):
            raise ValueError("Invalid fixed checkpoint output label")
        name = entry["path"]
        path = (work/name).resolve()
        if not path.is_relative_to(work.resolve()) or descriptor["files"].get(name) != entry["sha256"]:
            raise ValueError("Fixed checkpoint escaped its verified input inventory")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("Fixed checkpoint hash differs")
        result[label] = path
    return result


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
        if expected.get("parent_state_sha256"):
            preflight = json.loads(archive.read("ACTOR_PREFLIGHT.json"))
            if (report["lineage"]["parent_state_sha256"] != expected["parent_state_sha256"]
                    or report["total_optimizer_steps"] != expected["total_optimizer_steps"]
                    or report["total_optimized_positions"] != expected["total_optimized_positions"]
                    or report["anchor_sha256"] != expected["original_anchor_sha256"]
                    or report["actor_device"] != expected["actor_device"]):
                raise ValueError("Continuation lineage, cumulative exposure or frozen anchor differs")
            if (not preflight["passed"] or not preflight["actor_real_collection_passed"]
                    or preflight["actor_device"] != expected["actor_device"]
                    or not preflight["cuda_full_state_resume_verified"]
                    or not all(preflight["cuda_resume_checks"].values())
                    or not 0 <= preflight["cpu_cuda_max_error"] <= 1e-3
                    or not 0 < preflight["selected_best_mean_time_ratio"] <= 1.5):
                raise ValueError("Continuation CUDA preflight was not verified")
        destination.mkdir(parents=True, exist_ok=False)
        needed = ["training/latest.pt", "training/RESULTS.json", "experiment.json"]
        if expected.get("parent_state_sha256"):
            needed.append("ACTOR_PREFLIGHT.json")
        for name in needed:
            archive.extract(name, destination)
        with path.open("rb") as stream:
            export_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {"source_export_sha256": export_sha256, "checkpoint_sha256": report["checkpoint_sha256"],
                   "full_state_sha256": report["full_state_sha256"], "batches": report["batches"],
                   "verified_files": len(inventory), "condition": report["condition"],
                   "initial_sha256": expected["initial_sha256"], "automatic_promotion": False}
        if expected.get("parent_state_sha256"):
            receipt.update(parent_state_sha256=report["lineage"]["parent_state_sha256"],
                           anchor_sha256=report["anchor_sha256"], total_optimizer_steps=report["total_optimizer_steps"],
                           total_optimized_positions=report["total_optimized_positions"], cuda_resume_verified=True)
        (destination / "SOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return destination / "training/latest.pt"
