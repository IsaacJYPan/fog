"""Thin private Kaggle entry point for the fixed FOG teacher pilot and CPU screen."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile


def run():
    begun = time.monotonic()
    work = Path("/kaggle/temp/fog")
    output = Path("/kaggle/working/pilot")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require exactly one sealed pilot bundle")
    with zipfile.ZipFile(bundles[0]) as archive:
        for name in archive.namelist():
            if not (work / name).resolve().is_relative_to(work):
                raise ValueError("Unsafe fixed input path")
        archive.extractall(work)
    experiment = json.loads((work / "experiment.json").read_text())
    for name, expected in experiment["files"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work):
            raise ValueError("Unsafe inventory path")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen input hash changed: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1",
                       PYTHONIOENCODING="utf-8", PYTHONPATH=str(work), TORCHDYNAMO_DISABLE="1")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    status, error, returncode = "failed", None, 1
    try:
        source = None
        if experiment["stage"] == "evaluate":
            from fog.scripts.teacher_pilot_queue import resolve_training

            source = resolve_training(experiment["training_source"], Path("/kaggle/input"), output / "source-training")
            if source is None:
                status, returncode = "waiting", 0
                print("Waiting for the fixed GPU pilot export; CPU screen has not run.", flush=True)
                return
            (output / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
        if not shutil.which("node") or not shutil.which("npm"):
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "nodejs", "npm"], check=True)
        if experiment["stage"] == "evaluate":
            subprocess.run([sys.executable, "-m", "fog.scripts.pretrained_setup", "--stage", "evaluate", "--device", "cpu", "--root", str(work),
                            "--out", str(output / "native-environment")], env=environment, check=True)
        subprocess.run([sys.executable, "-m", "fog.scripts.kaggle_setup", "--device", experiment["device"]], env=environment, check=True)
        python = str(work / ".venv/bin/python")
        if experiment["stage"] == "train":
            command = [python, "-m", "fog.training.expert_pilot", "--config", "config.yaml", "--recipe", "recipe.json",
                       "--manifest", "inputs/teacher-data/manifest.json", "--initial", "inputs/random_init.pt",
                       "--out", str(output / "training"), "--device", "cuda"]
        else:
            command = [python, "-m", "fog.scripts.expert_evaluate", "--config", "config.yaml",
                       "--checkpoint", str(output / "source-training/training/latest.pt"), "--pool", "inputs/development.json",
                       "--out", str(output / "evaluation"), "--games", str(experiment["games"]),
                       "--native-python", str(work / ".runtime/native/bin/python"), "--native-checkpoint", "inputs/kakuna34.pt",
                       "--native-games", str(experiment["native_games"])]
        remaining = experiment["timeout_seconds"] - (time.monotonic() - begun) - 120
        if remaining <= 0:
            raise RuntimeError("Setup consumed the frozen job deadline")
        from fog.scripts.kaggle_worker import run_training

        # The training runner prints its fixed progress; periodic saves live in
        # saved outputs throughout the job. Independent CPU evaluation uses its
        # own resource allocation and never extends the GPU pilot.
        print("Starting frozen " + experiment["stage"] + " worker", flush=True)
        returncode = run_training(command, environment, remaining, log_path=output / "worker.log")
        if returncode:
            raise RuntimeError("Pilot worker returned exit code " + str(returncode))
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "stage": experiment["stage"], "error": error,
                 "returncode": returncode, "elapsed_seconds": time.monotonic() - begun,
                 "gpu_requested": experiment["device"] == "cuda", "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        for path in (work / "runs/kaggle_environment").glob("*"):
            if path.is_file():
                destination = output / "environment" / path.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(path, destination)
        inventory = {}
        for path in output.rglob("*"):
            if path.is_file() and path.name != "output_hashes.json":
                with path.open("rb") as stream:
                    inventory[path.relative_to(output).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
        (output / "output_hashes.json").write_text(json.dumps(inventory, indent=2) + "\n")
        shutil.make_archive("/kaggle/working/fog-experiment-output", "zip", output)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
