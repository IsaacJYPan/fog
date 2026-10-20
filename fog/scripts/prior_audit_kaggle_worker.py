"""A sealed CPU-only prior-data experiment; no neural training or model promotion."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile


def run():
    started = time.monotonic()
    work = Path("/kaggle/temp/fog")
    output = Path("/kaggle/working/prior-audit")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require one sealed prior-data bundle")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        names = archive.namelist()
        if (
            descriptor["format"] != "fog.kakuna-prior-audit"
            or descriptor["device"] != "cpu"
            or len(names) != len(set(names))
            or set(names) != set(descriptor["files"]) | {"experiment.json"}
            or any(not (work / name).resolve().is_relative_to(work) for name in names)
        ):
            raise ValueError("Unsafe or incomplete prior experiment")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Changed frozen prior input: " + name)
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.scripts.kaggle_worker import run_training
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(work), str(work / "upstream/poke-env/src"))),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
    )
    status, error, phase = "failed", None, "setup"

    def child(command, log):
        seconds = descriptor["timeout_seconds"] - (time.monotonic() - started) - 120
        code = run_training(command, environment, seconds, log_path=output / log)
        if code:
            raise RuntimeError("Prior worker failed: " + str(code))

    try:
        if not shutil.which("node") or not shutil.which("npm"):
            child(["apt-get", "update"], "apt-update.log")
            child(["apt-get", "install", "-y", "nodejs", "npm"], "apt-install.log")
        child(
            [
                "npm",
                "--prefix",
                "server/pokemon-showdown",
                "ci",
                "--omit=dev",
                "--omit=optional",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
            ],
            "npm.log",
        )
        venv = work / ".runtime/prior"
        child(
            [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(venv)],
            "venv.log",
        )
        python = str(venv / "bin/python")
        child(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-r",
                "configs/prior-audit-requirements.txt",
            ],
            "dependencies.log",
        )
        child(
            [
                python,
                "-c",
                "import json,numpy,platform;print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__}))",
            ],
            "platform.log",
        )
        child(["node", "--version"], "node-version.log")
        child(
            [
                python,
                "-c",
                "import json;from pathlib import Path;from fog.pretrained.data import composition;"
                "from fog.pretrained.prior_data import validate_export;"
                "team=json.loads(Path('inputs/pools/train.json').read_text())['teams'][0];"
                "assert validate_export([team],Path('server/pokemon-showdown'))[0]['valid'];"
                "print(json.dumps({'roster_parser':composition(team),'validator':True}))",
            ],
            "parser-preflight.log",
        )
        phase = "bounded prior construction and matched reconstruction benchmark"
        child(
            [
                python,
                "-m",
                "fog.scripts.pretrained_prior_audit",
                "--config",
                "inputs/recipe.json",
                "--out",
                str(output / "audit"),
            ],
            "audit.log",
        )
        report = json.loads((output / "audit/RESULTS.json").read_text())
        if (
            not report["complete"]
            or not report["baseline_reproduced"]
            or not report["heldout_rosters_excluded"]
        ):
            raise ValueError("The fixed prior comparison did not pass its proof gates")
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {
            "status": status,
            "error": error,
            "phase": phase,
            "returncode": 0 if status == "complete" else 1,
            "elapsed_seconds": time.monotonic() - started,
            "gpu_requested": False,
            "training_performed": False,
            "automatic_promotion": False,
        }
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
