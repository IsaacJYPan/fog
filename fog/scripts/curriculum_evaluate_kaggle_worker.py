"""Separate CPU full-game comparison of both online candidates and their start."""

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
    work, output = Path("/kaggle/temp/fog"), Path("/kaggle/working/evaluation")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require one sealed curriculum-evaluation bundle")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        if descriptor["format"] != "fog.curriculum-evaluation" or descriptor["device"] != "cpu":
            raise ValueError("Only the separate CPU evaluation recipe is supported")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(descriptor["files"]) | {"experiment.json"}:
            raise ValueError("Fixed CPU evaluation inventory differs")
        if any(not (work / name).resolve().is_relative_to(work) for name in names):
            raise ValueError("Unsafe fixed input path")
        archive.extractall(work)
    for name, sha in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                raise ValueError("Frozen evaluation input changed: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.scripts.curriculum_evaluation import fixed_checkpoints, resolve
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output
    from fog.pretrained.runtime import native_environment

    environment = native_environment(work)
    environment.update(PYTHONPATH=str(work), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1",
                       PYTHONIOENCODING="utf-8", TORCHDYNAMO_DISABLE="1")
    output.mkdir(parents=True)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    status, error, phase, returncode = "failed", None, "verify GPU sources", 1

    def remaining():
        return descriptor["timeout_seconds"]-(time.monotonic()-begun)-120

    try:
        checkpoints = fixed_checkpoints(descriptor, work)
        if len(descriptor["training_sources"]) not in (1, 2):
            raise ValueError("Require one continuation or two matched training conditions")
        conditions = set(checkpoints)
        for source in descriptor["training_sources"]:
            condition = source["experiment"]["condition"]
            if condition in conditions:
                raise ValueError("A training condition overwrites a fixed control")
            conditions.add(condition)
            candidate = resolve(source, Path("/kaggle/input"), output / ("source-" + condition))
            if candidate is None:
                status, returncode = "waiting", 0
                print("Waiting for the completed fixed GPU exports; no evaluation was run.", flush=True)
                return
            if source["experiment"]["initial_sha256"] != descriptor["initial_sha256"]:
                raise ValueError("Evaluation start differs from GPU starting weights")
            checkpoints[condition] = candidate
        if set(checkpoints) != conditions:
            raise ValueError("CPU sources do not contain all fixed training conditions")
        phase = "setup"
        execute([sys.executable, "-m", "fog.scripts.pretrained_setup", "--stage", "evaluate", "--device", "cpu",
                 "--root", str(work), "--out", str(output / "native-environment")],
                environment, output / "native-setup.log", remaining())
        execute([sys.executable, "-m", "fog.scripts.kaggle_setup", "--device", "cpu"],
                environment, output / "setup.log", remaining())
        for condition, checkpoint in checkpoints.items():
            phase = "full-game screen: " + condition
            execute([str(work / ".venv/bin/python"), "-m", "fog.scripts.expert_evaluate", "--config", "config.yaml",
                     "--checkpoint", str(checkpoint), "--pool", "inputs/development.json",
                     "--out", str(output / condition), "--games", str(descriptor["games"]),
                     "--native-python", str(work / ".runtime/native/bin/python"), "--native-checkpoint", "inputs/kakuna34.pt",
                     "--native-games", str(descriptor["native_games"])],
                    environment, output / (condition + ".log"), remaining())
        status, returncode = "complete", 0
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "returncode": returncode, "phase": phase,
                 "elapsed_seconds": time.monotonic()-begun, "gpu_requested": False, "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
