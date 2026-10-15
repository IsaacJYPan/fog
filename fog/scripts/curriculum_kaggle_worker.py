"""Fixed private GPU entry point for one arm of the online curriculum pilot."""

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
    work, output = Path("/kaggle/temp/fog"), Path("/kaggle/working/pilot")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require one sealed online-pilot bundle")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        if descriptor["format"] != "fog.curriculum-pilot" or descriptor["device"] != "cuda":
            raise ValueError("Only the fixed GPU online-pilot recipe is supported")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(descriptor["files"]) | {"experiment.json"}:
            raise ValueError("Fixed online-pilot inventory differs")
        if any(not (work / name).resolve().is_relative_to(work) for name in names):
            raise ValueError("Unsafe fixed input path")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen online-pilot input changed: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    environment = dict(os.environ, PYTHONPATH=str(work), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TORCHDYNAMO_DISABLE="1")
    output.mkdir(parents=True)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    status, error, phase, returncode = "failed", None, "setup", 1

    def remaining():
        return descriptor["timeout_seconds"] - (time.monotonic()-begun) - 120

    try:
        if not shutil.which("node") or not shutil.which("npm"):
            execute(["apt-get", "update"], environment, output / "apt-update.log", remaining())
            execute(["apt-get", "install", "-y", "nodejs", "npm"], environment, output / "apt-install.log", remaining())
        execute([sys.executable, "-m", "fog.scripts.kaggle_setup", "--device", "cuda"],
                environment, output / "setup.log", remaining())
        python = str(work / ".venv/bin/python")
        initial = descriptor.get("initial_file", "inputs/bc1000.pt")
        if initial not in descriptor["files"]:
            raise ValueError("The initial checkpoint is outside the verified inventory")
        if descriptor.get("parent_state_sha256") and descriptor["parent_state_sha256"] != descriptor["files"].get("inputs/parent.state.pt"):
            raise ValueError("The parent checkpoint is outside its fixed continuation identity")
        if descriptor.get("actor_preflight"):
            phase = "actor and CUDA training-state preflight"
            preflight = [python, "-m", "fog.training.curriculum_actor_preflight", "--config", "config.yaml",
                     "--recipe", "recipe.json", "--bank", "inputs/bank/manifest.json", "--pool", "inputs/train.json",
                     "--checkpoint", initial, "--out", str(output / "ACTOR_PREFLIGHT.json")]
            if descriptor.get("parent_state_sha256"):
                preflight.extend(["--parent", "inputs/parent.state.pt", "--anchor-checkpoint", "inputs/original_anchor.pt"])
            execute(preflight, environment, output / "actor-preflight.log", remaining())
        phase = "online PPO and exact-team auxiliary training"
        command = [python, "-m", "fog.training.curriculum_pilot", "--config", "config.yaml",
                 "--recipe", "recipe.json", "--bank", "inputs/bank/manifest.json", "--pool", "inputs/train.json",
                 "--initial", initial, "--out", str(output / "training"), "--device", "cuda"]
        if descriptor.get("parent_state_sha256"):
            command.extend(["--parent", "inputs/parent.state.pt", "--anchor-checkpoint", "inputs/original_anchor.pt"])
        execute(command, environment, output / "worker.log", remaining())
        report = json.loads((output / "training/RESULTS.json").read_text())
        if not report["complete"] or report["batches"] != descriptor["max_batches"]:
            raise ValueError("Online pilot did not reach its frozen optimizer-update budget")
        status, returncode = "complete", 0
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "phase": phase, "returncode": returncode,
                 "elapsed_seconds": time.monotonic()-begun, "gpu_requested": True, "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        for path in (work / "runs/kaggle_environment").glob("*"):
            if path.is_file():
                (output / "environment").mkdir(exist_ok=True)
                shutil.copy2(path, output / "environment" / path.name)
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
