"""CPU-only, fixed-input worker for the backward-curriculum feasibility screen."""

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
    work, output = Path("/kaggle/temp/fog"), Path("/kaggle/working/probe")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require exactly one sealed curriculum-probe bundle")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        if descriptor["format"] != "fog.backplay-probe" or descriptor["device"] != "cpu":
            raise ValueError("Only the CPU curriculum-feasibility recipe is supported")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(descriptor["files"]) | {"experiment.json"}:
            raise ValueError("Curriculum-probe inventory differs from the fixed input package")
        if any(not (work / name).resolve().is_relative_to(work) for name in names):
            raise ValueError("Unsafe fixed-input path")
        archive.extractall(work)
    for name, sha in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                raise ValueError("Frozen input hash changed: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    environment = dict(os.environ, PYTHONPATH=str(work), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TORCHDYNAMO_DISABLE="1")
    output.mkdir(parents=True)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    status, error, phase = "failed", None, "setup"

    def remaining():
        return descriptor["timeout_seconds"] - (time.monotonic() - begun) - 90

    try:
        if not shutil.which("node") or not shutil.which("npm"):
            execute(["apt-get", "update"], environment, output / "apt-update.log", remaining())
            execute(["apt-get", "install", "-y", "nodejs", "npm"], environment, output / "apt-install.log", remaining())
        execute([sys.executable, "-m", "fog.scripts.kaggle_setup", "--device", "cpu"],
                environment, output / "setup.log", remaining())
        phase = "verified-prefix continuation screen"
        execute([str(work / ".venv/bin/python"), "-m", "fog.scripts.backplay_probe",
                 "--config", "config.yaml", "--initial", "inputs/random_init.pt",
                 "--checkpoint", "inputs/bc1000.pt", "--bank", "inputs/bank/manifest.json",
                 "--out", str(output / "checks"), "--records", str(descriptor["records"]),
                 "--seed", str(descriptor["seed"]), "--horizons", *map(str, descriptor["horizons"])],
                environment, output / "worker.log", remaining())
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "phase": phase,
                 "elapsed_seconds": time.monotonic() - begun, "gpu_requested": False,
                 "neural_training": False, "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        for path in (work / "runs/kaggle_environment").glob("*"):
            if path.is_file():
                (output / "environment").mkdir(exist_ok=True)
                shutil.copy2(path, output / "environment" / path.name)
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
