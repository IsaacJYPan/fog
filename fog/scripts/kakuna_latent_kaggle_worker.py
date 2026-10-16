"""Run a sealed native feature-extraction stage on Kaggle CPU."""

from pathlib import Path
import hashlib
import json
import os
import sys
import time
import zipfile


def run():
    started = time.monotonic()
    work, out = Path("/kaggle/temp/fog"), Path("/kaggle/working/latent-cache")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require one sealed native cache package")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        if descriptor["format"] != "fog.kakuna-latent-cache" or descriptor["device"] != "cpu":
            raise ValueError("Only the fixed CPU extraction recipe is supported")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(descriptor["files"]) | {"experiment.json"}:
            raise ValueError("Incomplete or duplicate cache package inventory")
        if any(not (work / name).resolve().is_relative_to(work) for name in names):
            raise ValueError("Unsafe cache package member")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Changed cache input: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.pretrained.runtime import native_environment
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    environment = native_environment(work)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1",
                       PYTHONIOENCODING="utf-8", TORCHDYNAMO_DISABLE="1")
    out.mkdir(parents=True)
    (out / "experiment.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    status, error, phase = "failed", None, "native CPU setup"

    def remaining():
        return descriptor["timeout_seconds"] - (time.monotonic() - started) - 120

    try:
        execute([sys.executable, "-m", "fog.scripts.pretrained_setup", "--stage", "train",
                 "--device", "cpu", "--root", str(work), "--out", str(out / "environment")],
                environment, out / "setup.log", remaining())
        phase = "complete-history feature extraction"
        execute([str(work / ".runtime/native/bin/python"), "-m", "fog.scripts.kakuna_latent_cache",
                 "--config", "inputs/cache.json", "--out", str(out / "cache")],
                environment, out / "cache.log", remaining())
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "returncode": 0 if status == "complete" else 1,
                 "phase": phase, "elapsed_seconds": time.monotonic() - started,
                 "gpu_requested": False, "automatic_promotion": False}
        (out / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(out, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
