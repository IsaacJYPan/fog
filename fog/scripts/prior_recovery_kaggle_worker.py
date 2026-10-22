"""Run one sealed CPU diagnostic capsule; no training or native-policy weights."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile


def extract_bundle(bundle, work):
    work = Path(work).resolve()
    with zipfile.ZipFile(bundle) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        names = archive.namelist()
        if (
            descriptor["format"] != "fog.prior-recovery-experiment"
            or descriptor["version"] != 1
            or descriptor["device"] != "cpu"
            or descriptor["stage"] != "evaluate"
            or len(names) != len(set(names))
            or set(names) != set(descriptor["files"]) | {"experiment.json"}
            or any(not (work / name).resolve().is_relative_to(work) for name in names)
        ):
            raise ValueError("Unsafe or inconsistent CPU capsule")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Changed sealed CPU input: " + name)
    return descriptor


def run():
    started = time.monotonic()
    work, output = Path("/kaggle/temp/fog"), Path("/kaggle/working/prior-recovery")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require exactly one fixed CPU capsule")
    descriptor = extract_bundle(bundles[0], work)
    schema = json.loads((work / "inputs/LABEL_SCHEMA_AUDIT.json").read_text())
    if not schema["passed"] or any(
        descriptor["files"].get(name) != expected
        for name, expected in schema["source_files"].items()
    ):
        raise ValueError("Cached categorical labels differ from this runtime schema")
    os.chdir(work)
    sys.path.insert(0, str(work))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    environment = dict(os.environ, PYTHONPATH=str(work), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                       TORCHDYNAMO_DISABLE="1", PYTHONIOENCODING="utf-8")
    status, error, phase = "failed", None, "runtime and numerical control checks"
    try:
        import torch
        import numpy

        platform = {"python": sys.version, "torch": torch.__version__,
                    "numpy": numpy.__version__, "GPU_available": torch.cuda.is_available(),
                    "requested_device": "cpu"}
        (output / "PLATFORM.json").write_text(json.dumps(platform, indent=2) + "\n")
        if torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.11.0":
            raise ValueError("Require CPU allocation and the tested PyTorch 2.11 runtime")
        # Match the preceding evaluator's NumPy without replacing base Torch or
        # installing unrelated native/opponent environments.
        from fog.scripts.pretrained_setup import environment as make_environment

        runtime = make_environment(work, "recovery-evaluator")
        execute([str(runtime), "-m", "pip", "install", "numpy==2.5.3"], environment,
                output / "setup.log", 600, isolated=False)
        execute([str(runtime), "-m", "fog.scripts.compare_prior_recovery", "--config",
                 "inputs/comparison.json", "--out", str(output / "probability-probe"),
                 "--probe-reference", "inputs/REFERENCE_PROBE.json"], environment,
                output / "probe.log", 600, isolated=False)
        for phase, module, filename, target in [
            ("matched sampler recovery comparison", "fog.scripts.compare_prior_recovery",
             "comparison.json", "comparison"),
            ("production prefix support audit", "fog.scripts.audit_prior_prefixes",
             "audit.json", "audit"),
        ]:
            remaining = descriptor["timeout_seconds"] - (time.monotonic() - started) - 120
            execute([str(runtime), "-m", module, "--config", "inputs/" + filename,
                     "--out", str(output / target)], environment, output / f"{target}.log",
                    remaining, isolated=False,
                    progress=lambda: {"phase": phase, "elapsed_seconds": time.monotonic() - started})
            report = json.loads((output / target / "RESULTS.json").read_text())
            if not report["complete"] or report["native_policy_changed"] or report["search_connected"]:
                raise ValueError("CPU diagnostic result failed its completion/isolation gate")
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {"status": status, "error": error, "phase": phase,
                 "returncode": 0 if status == "complete" else 1,
                 "elapsed_seconds": time.monotonic() - started, "gpu_requested": False,
                 "native_policy_changed": False, "training_performed": False,
                 "automatic_promotion": False}
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
