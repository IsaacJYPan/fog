"""Sealed paired native belief pilot; CPU extraction is a separate saved source."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile


def run():
    begun = time.monotonic()
    work, output = Path("/kaggle/temp/fog"), Path("/kaggle/working/retention")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require exactly one sealed training package")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        names = archive.namelist()
        if (descriptor["format"] != "fog.kakuna-belief-retention" or descriptor["device"] != "cuda"
                or len(names) != len(set(names)) or set(names) != set(descriptor["files"]) | {"experiment.json"}):
            raise ValueError("Wrong or incomplete GPU package")
        if any(not (work / name).resolve().is_relative_to(work) for name in names):
            raise ValueError("Unsafe GPU package member")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen GPU input changed: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.pretrained.runtime import native_environment
    from fog.pretrained.latent_training import HEAD_SCHEMA_FILES
    from fog.scripts.kaggle_results import validate_members
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output
    from fog.scripts.teacher_pilot_queue import verified_inventory

    candidates = []
    if any(descriptor["cache_source"]["experiment"]["files"].get(name) != descriptor["files"].get(name) for name in HEAD_SCHEMA_FILES):
        raise ValueError("Cached labels and training decoder use different token schemas")
    for path in Path("/kaggle/input").rglob("fog-experiment-output.zip"):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == descriptor["cache_source"]["archive_sha256"]:
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Missing or ambiguous audited CPU cache output")
    with zipfile.ZipFile(candidates[0]) as archive:
        validate_members(archive)
        verified_inventory(archive)
        if json.loads(archive.read("experiment.json")) != descriptor["cache_source"]["experiment"]:
            raise ValueError("CPU source differs from frozen training recipe")
        state = json.loads(archive.read("pipeline_state.json"))
        if state["status"] != "complete" or state["error"] is not None:
            raise ValueError("CPU feature extraction did not complete")
        if hashlib.sha256(archive.read("cache/manifest.json")).hexdigest() != descriptor["cache_sha256"]:
            raise ValueError("CPU cache manifest differs from audited export")
        for entry in archive.infolist():
            if entry.filename.startswith("cache/") and not entry.is_dir():
                target = work / "inputs" / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    environment = native_environment(work)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       TORCHDYNAMO_DISABLE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    status, error, phase = "failed", None, "native GPU setup"

    def remaining():
        return descriptor["timeout_seconds"] - (time.monotonic() - begun) - 120

    def child(arguments, log):
        execute(arguments, environment, output / log, remaining())

    native = str(work / ".runtime/native/bin/python")
    try:
        child([sys.executable, "-m", "fog.scripts.pretrained_setup", "--stage", "train", "--device", "cuda",
               "--root", str(work), "--out", str(output / "environment")], "setup.log")
        phase = "original CPU/CUDA and dense/cached probabilities"
        child([native, "-m", "fog.scripts.pretrained_contract", "--checkpoint", "inputs/kakuna34.pt",
               "--samples", "inputs/contract-inputs.pt", "--reference", "inputs/cpu-probabilities.pt",
               "--device", "cuda", "--out", str(output / "contract-initial")], "initial-contract.log")
        if descriptor['training'].get('retention_replay'):
            phase = "discarded CUDA regression of original-reference repair"
            child([native, "-m", "fog.scripts.kakuna_retention_repair_check", "--config", "inputs/training.json",
                   "--fixture", "inputs/retention-regression.pt", "--device", "cuda",
                   "--out", str(output / "repair-check")], "repair-check.log")
        phase = "discarded CUDA full-state resume check"
        child([native, "-m", "fog.scripts.kakuna_latent_resume_check", "--config", "inputs/training.json",
               "--device", "cuda", "--out", str(output / "resume-check")], "resume-check.log")
        phase = "matched frozen/adapted shared-representation training"
        child([native, "-m", "fog.scripts.kakuna_latent_train", "--config", "inputs/training.json",
               "--device", "cuda", "--out", str(output / "training")], "training.log")
        report = json.loads((output / "training/RESULTS.json").read_text())
        if not report["complete"] or not report["frozen_native_unchanged"]:
            raise ValueError("Paired training did not complete its fixed exposure or frozen-native checks")
        phase = "trained CPU/CUDA and dense/cached probabilities"
        for device in ("cpu", "cuda"):
            command = [native, "-m", "fog.scripts.pretrained_contract", "--checkpoint", str(output / "training/native_final.pt"),
                       "--samples", "inputs/contract-inputs.pt", "--trained", "--device", device,
                       "--out", str(output / "contract-final" / device)]
            if device == "cuda":
                command += ["--reference", str(output / "contract-final/cpu/probabilities.pt")]
            child(command, "final-contract-" + device + ".log")
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "returncode": 0 if status == "complete" else 1,
                 "phase": phase, "elapsed_seconds": time.monotonic() - begun,
                 "gpu_requested": True, "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
