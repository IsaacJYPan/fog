"""Thin Kaggle entry point; all experiment inputs arrive in a frozen private dataset."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def _signal_training(process, sig) -> None:
    # Saved Kaggle scripts run on Linux. All descendants join the dedicated process group,
    # including the training driver, learner, CPU actors and Showdown processes.
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def run_training(command: list[str], environment: dict, seconds: float) -> int:
    """Leave time to export saved checkpoints before Kaggle's session deadline."""
    if seconds <= 0:
        raise RuntimeError("No training time remains before the export deadline")
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        _signal_training(process, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _signal_training(process, getattr(signal, "SIGKILL", signal.SIGTERM))
            process.wait()
        return 124


def run() -> None:
    started = time.monotonic()
    # Keep the environment and corpus out of notebook outputs.
    work = Path("/kaggle/temp/fog")
    # Keep checkpoints in saved outputs throughout training, even if the platform
    # interrupts the session before the final export archive can be assembled.
    output = Path("/kaggle/working/training")
    archives = list(Path("/kaggle/input").rglob("bundle.zip"))
    manifests = list(Path("/kaggle/input").rglob("experiment.json"))
    if archives:
        if len(archives) != 1:
            raise RuntimeError("Expected exactly one experiment bundle")
        with zipfile.ZipFile(archives[0]) as archive:
            for name in archive.namelist():
                if not (work.parent / name).resolve().is_relative_to(work):
                    raise ValueError("Unsafe bundle path")
            archive.extractall(work.parent)
    else:
        if len(manifests) != 1:
            raise RuntimeError("Expected exactly one frozen experiment")
        shutil.copytree(manifests[0].parent, work)
    experiment = json.loads((work / "experiment.json").read_text())
    for name, expected in experiment["files"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work):
            raise ValueError("Unsafe inventory path")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Frozen input changed: {name}")
    os.chdir(work)
    output.mkdir(parents=True, exist_ok=True)
    output_link = work / "runs/experiment"
    output_link.parent.mkdir(parents=True, exist_ok=True)
    output_link.symlink_to(output, target_is_directory=True)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    code = 1
    try:
        if not shutil.which("node") or not shutil.which("npm"):
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "nodejs", "npm"], check=True)
        subprocess.run([sys.executable, "-m", "fog.scripts.kaggle_setup"], check=True)
        python = str(work / ".venv/bin/python")
        job = work / experiment["job"]
        import yaml
        recipe = yaml.safe_load(job.read_text())
        manifest = recipe.get("overrides", {}).get("replay_policy.manifest")
        if manifest:
            subprocess.run([python, "-m", "fog.scripts.validate_replay_decisions",
                            "--manifest", manifest, "--out",
                            str(output / "input_validation.json")], check=True)
        if recipe.get("safety", {}).get("policy_contract"):
            subprocess.run([python, "-m", "fog.scripts.check_policy_contract", "--job", str(job),
                            "--out", str(output / "policy_contract.json")],
                           env=environment, check=True)
        remaining = int(experiment["timeout_seconds"]) - (time.monotonic() - started) - 300
        code = run_training([python, "-m", "fog.scripts.automated_training", "run",
                             "--job", str(job), "--max-blocks", "1"], environment, remaining)
        if code == 124:
            state_path = output / "pipeline_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                state.update(status="interrupted", error="Training deadline reached; saved checkpoints retained")
                for block in state.get("blocks", []):
                    if block.get("status") == "running":
                        block.update(status="interrupted", error=state["error"])
                state_path.write_text(json.dumps(state, indent=2))
    finally:
        for log in sorted(output.rglob("*.log")):
            print(log.relative_to(output), flush=True)
            print(log.read_text(errors="replace")[-12000:], flush=True)
        state = output / "pipeline_state.json"
        if not state.exists():
            state.write_text(json.dumps({"status": "failed", "blocks": [],
                                         "error": "Setup failed before training"}))
        runtime = work / "runs/kaggle_environment"
        if runtime.exists():
            shutil.copytree(runtime, output / "environment", dirs_exist_ok=True)
        (output / "execution.json").write_text(json.dumps({"returncode": code, "timed_out": code == 124}))
        shutil.make_archive("/kaggle/working/fog-experiment-output", "zip", output)
    if code:
        raise RuntimeError(f"Training failed with exit code {code}; inspect the exported logs")


if __name__ == "__main__":
    run()
