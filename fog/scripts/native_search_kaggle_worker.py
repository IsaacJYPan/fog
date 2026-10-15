"""Thin private CPU worker for a sealed known-team native engineering canary."""

from pathlib import Path
import hashlib
import json
import os
import sys
import time
import zipfile


def run():
    begun = time.monotonic()
    work = Path("/kaggle/temp/fog")
    output = Path("/kaggle/working/canary")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require exactly one sealed native canary bundle")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        if (descriptor["format"] != "fog.native-search-canary" or descriptor["device"] != "cpu"
                or not descriptor["privileged_known_teams"]):
            raise ValueError("Only the declared known-team CPU engineering recipe is supported")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(descriptor["files"]) | {"experiment.json"}:
            raise ValueError("Native canary inventory has missing, extra or duplicate files")
        for entry in archive.infolist():
            if not (work / entry.filename).resolve().is_relative_to(work):
                raise ValueError("Unsafe native canary input path")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen native canary input changed: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.pretrained.runtime import native_environment
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    environment = native_environment(work)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1",
                       PYTHONIOENCODING="utf-8", TORCHDYNAMO_DISABLE="1")
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    status, error, returncode, phase = "failed", None, 1, "setup"

    def remaining():
        return descriptor["timeout_seconds"] - (time.monotonic() - begun) - 90

    try:
        execute([sys.executable, "-m", "fog.scripts.pretrained_setup", "--stage", "evaluate", "--device", "cpu",
                 "--root", str(work), "--out", str(output / "environment")],
                environment, output / "setup.log", remaining())
        phase = "known-team canary"
        execute([str(work / ".runtime/native/bin/python"), "-m", "fog.scripts.native_search_canary",
                 "--checkpoint", "inputs/kakuna34.pt", "--pool", "inputs/development.json",
                 "--reference-source", "inputs/native_control_v4.py", "--out", str(output / "checks"),
                 "--games", str(descriptor["games"]), "--max-rounds", "2000", "--seed", str(descriptor["seed"]),
                 "--max-actions", str(descriptor["max_actions"]), "--samples", str(descriptor["chance_samples"]),
                 "--seconds", str(descriptor["decision_seconds"]),
                 "--max-fallback-rate", str(descriptor["max_budget_fallback_rate"]),
                 "--known-teams-diagnostic"], environment, output / "worker.log", remaining())
        status, returncode = "complete", 0
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "returncode": returncode, "phase": phase,
                 "elapsed_seconds": time.monotonic() - begun, "gpu_requested": False,
                 "privileged_known_teams": True, "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
