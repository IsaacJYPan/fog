"""Verify a fixed GPU pilot export before a separately scheduled CPU screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from fog.scripts.kaggle_results import validate_members


def resolve_training(source, input_root: Path, destination: Path):
    """Return None while absent; reject failed, mismatched or unverified outputs."""
    expected = source["experiment"]
    if expected.get("format") != "fog.teacher-pilot-experiment" or expected.get("version") != 1 or expected.get("stage") != "train" or expected.get("device") != "cuda":
        raise ValueError("Queued source must be a fixed CUDA teacher pilot")
    candidates = []
    for path in input_root.rglob("fog-experiment-output.zip"):
        with zipfile.ZipFile(path) as archive:
            if "experiment.json" in archive.namelist():
                candidate = json.loads(archive.read("experiment.json"))
                if candidate.get("id") == expected["id"]:
                    candidates.append(path)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Ambiguous GPU pilot output")
    path = candidates[0]
    with zipfile.ZipFile(path) as archive:
        validate_members(archive)
        if json.loads(archive.read("experiment.json")) != expected:
            raise ValueError("GPU pilot descriptor differs from the frozen CPU source")
        inventory = json.loads(archive.read("output_hashes.json"))
        if set(archive.namelist()) != set(inventory) | {"output_hashes.json"}:
            raise ValueError("GPU export has unverified or missing members")
        for name, digest in inventory.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("GPU pilot exported hash mismatch: " + name)
        pipeline = json.loads(archive.read("pipeline_state.json"))
        if pipeline["status"] != "complete" or pipeline["error"] is not None or pipeline["returncode"] != 0:
            raise ValueError("GPU pilot did not complete successfully")
        report = json.loads(archive.read("training/RESULTS.json"))
        if not report["complete"] or report["device"] != "cuda" or report["updates"] < source["minimum_updates"]:
            raise ValueError("GPU pilot is incomplete or below its frozen minimum exposure")
        if report["data_sha256"] != expected["data_sha256"] or report["initial_sha256"] != expected["initial_sha256"]:
            raise ValueError("GPU pilot training data or starting weights differ")
        for name in ("starting_contract", "final_contract"):
            contract = report[name]
            if not contract["passed"] or not contract["cpu_cuda_verified"] or not 0 < contract["tolerance"] <= 1e-3:
                raise ValueError("GPU pilot probability contract is missing")
            if not 0 <= contract["cpu_cuda_max_error"] <= 1e-3:
                raise ValueError("GPU pilot CPU/CUDA probability disagreement")
        if report["checkpoint_sha256"] != inventory["training/latest.pt"] or report["full_state_sha256"] != inventory["training/latest.state.pt"]:
            raise ValueError("GPU pilot final weights/state are not bound to its report")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    return {"directory": str(destination), "updates": report["updates"],
            "checkpoint_sha256": report["checkpoint_sha256"], "verified_exported_files": len(inventory),
            "gpu_contracts_passed": True, "automatic_promotion": False}
