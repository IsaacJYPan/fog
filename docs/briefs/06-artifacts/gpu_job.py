"""Private GPU measurement job; inject FOG_COMMIT before submitting.

Use FOG_STAGE=collection (1,800-second platform timeout), training (3,000),
or combined (3,600). Combined measures the missing MaxDamage matrix, selects
its best GPU batch, then runs the full training loop and evaluation. It completes
the self-play matrix afterward only when the remaining deadline fits its measured cost.
Internal deadlines reserve 300 seconds for export. Both include setup.
Completed probe rows and training checkpoints survive a rerun of the script
with the same output directory. Budget any new GPU session separately.
FOG_PLATFORM_TIMEOUT may adjust collection up to 3,600 seconds or training
up to 3,000. The submission must request that same platform timeout, and the
sum of prior allocations and the new cap must fit three allocated GPU-hours.
Account settings belong only in the private submission directory.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    started = time.monotonic()
    output = Path("/kaggle/working/06-artifacts")
    output.mkdir(parents=True, exist_ok=True)
    root = Path("/kaggle/temp/fog")
    root.parent.mkdir(parents=True, exist_ok=True)
    commit = os.environ["FOG_COMMIT"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("FOG_COMMIT must identify the pushed commit")
    stage = os.environ.get("FOG_STAGE", "collection")
    if stage not in {"collection", "training", "combined"}:
        raise ValueError("FOG_STAGE must be collection, training or combined")
    default = {"collection": 1800, "training": 3000, "combined": 3600}[stage]
    maximum = 3000 if stage == "training" else 3600
    platform_limit = int(os.environ.get("FOG_PLATFORM_TIMEOUT", default))
    if not 600 <= platform_limit <= maximum:
        raise ValueError("FOG_PLATFORM_TIMEOUT must be between 600 and this stage's maximum")
    deadline = started + platform_limit - 300
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    monitor = None
    gpu_count = 2  # Conservative until nvidia-smi reports the allocation.
    status = "failed"
    # Attached prior private outputs allow another bounded job to restore saved PPO state.
    resume_roots = [path.parent for path in Path("/kaggle/input").rglob("collect_gpu_check.txt")
                    if (path.parent / "training/checkpoints").is_dir()]
    if len(resume_roots) > 1:
        raise ValueError("attach at most one preceding training job")
    if resume_roots:
        shutil.copytree(resume_roots[0], output, dirs_exist_ok=True)
        for directory, _, files in os.walk(output):
            Path(directory).chmod(0o755)
            for name in files:
                (Path(directory) / name).chmod(0o644)

    def run(command: list[str], name: str, *, cwd: Path = root) -> str:
        path = output / (name + ".txt")
        record = {"command": command, "commit": commit, "status": "running",
                  "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / (name + "-execution.json"), record)
        begin = time.monotonic()
        with path.open("w") as stream:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        record.update(seconds=time.monotonic() - begin, exit_code=code,
                      status="complete" if code == 0 else "failed",
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / (name + "-execution.json"), record)
        print(name, record["status"], flush=True)
        if code:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            print(path.read_text()[-6000:], flush=True)
            raise RuntimeError(f"{name} exited {code}; stop before further measurements")
        return path.read_text()

    try:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader",
        ], text=True).strip().splitlines()
        gpu_count = len(gpu)
        if gpu_count not in {1, 2}:
            raise RuntimeError("Unexpected GPU allocation; stop to preserve the budget")
        with (output / "gpu-memory.txt").open("w") as stream:
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,utilization.gpu",
                "--format=csv", "--loop=5",
            ], stdout=stream, stderr=subprocess.STDOUT)
        if not root.exists():
            run(["git", "clone", "https://github.com/IsaacJYPan/fog.git", str(root)],
                "clone", cwd=root.parent)
        run(["git", "fetch", "origin", commit], "fetch")
        run(["git", "checkout", "--detach", commit], "checkout")
        run(["git", "submodule", "update", "--init", "--recursive"], "submodule")
        run(["npm", "--prefix", "server/pokemon-showdown", "ci", "--ignore-scripts",
             "--no-audit", "--no-fund"], "showdown-install")
        run(["npm", "--prefix", "server/pokemon-showdown", "run", "build"], "showdown-build")
        # Reuse the established installer; it retains Kaggle's CUDA PyTorch.
        run([sys.executable, "-m", "fog.scripts.kaggle_setup"], "setup")
        python = str(root / ".venv/bin/python")
        workers = len(os.sched_getaffinity(0))
        quota_path = Path("/sys/fs/cgroup/cpu.max")
        quota = quota_path.read_text().strip() if quota_path.exists() else "unavailable"
        if quota != "unavailable" and not quota.startswith("max"):
            amount, period = map(int, quota.split())
            workers = min(workers, max(1, (amount + period - 1) // period))
        cpu = next(line.split(":", 1)[1].strip() for line in
                   Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        write_json(output / "environment.json", {
            "commit": commit, "cpu_model": cpu, "cpu_count": os.cpu_count(),
            "affinity_count": len(os.sched_getaffinity(0)), "cpu_quota": quota,
            "workers": workers, "gpu": gpu, "label": "smoke",
        })
        if stage == "training":
            train(root, output, python, workers, gpu_count, environment, run)
            status = "complete"
            return
        rows_path = output / "collection.json"
        saved = json.loads(rows_path.read_text()) if rows_path.exists() else {"rows": []}
        opponents = ("MaxDamage",) if stage == "combined" else ("self", "MaxDamage")
        def matrix(opponents):
            for opponent in opponents:
                for device, batch in (("cpu", 32), ("cuda", 32), ("cuda", 64), ("cuda", 128)):
                    for repeat in (1, 2):
                        name = f"probe-{opponent}-{device}-{batch}-{repeat}"
                        if any(row["name"] == name for row in saved["rows"]):
                            continue
                        command = [python, "-m", "fog.eval.showdown_rollout_probe", "--workers",
                                   str(workers), "--batch", str(batch), "--games", "512",
                                   "--opponent", opponent, "--pool", "data/teams/pool_a",
                                   "--device", device]
                        text = run(command, name)
                        line = next(line for line in text.splitlines() if line.startswith("workers="))
                        values = dict(field.split("=", 1) for field in line.split())
                        saved["rows"].append({"name": name, "label": "smoke", "commit": commit,
                                              "command": command, "repeat": repeat, **values})
                        write_json(rows_path, saved)

        matrix(opponents)
        if stage == "combined":
            import yaml

            config = root / "configs/collect_gpu_check.yaml"
            previous = output / "collect_gpu_check.txt"
            if previous.exists():
                recipe = yaml.safe_load(previous.read_text())
            else:
                recipe = yaml.safe_load(config.read_text())
                speeds = {}
                for batch in (32, 64, 128):
                    measured = [row for row in saved["rows"] if row["opponent"] == "MaxDamage"
                                and row["device"] == "cuda" and int(row["batch"]) == batch]
                    assert len(measured) == 2
                    speeds[batch] = sum(int(row["learner_decisions"]) for row in measured) / sum(
                        float(row["seconds"]) for row in measured)
                best = max(speeds, key=speeds.get)
                recipe["env"].update(rollout_workers=workers, rollout_batch=best)
                recipe["ppo"]["games_per_chunk"] = workers * best
                write_json(output / "batch-selection.json", {
                    "label": "smoke", "basis": "aggregate learner decisions / aggregate seconds, two MaxDamage repeats",
                    "rows_per_second": speeds, "selected_batch": best, "workers": workers,
                    "self_play_matrix": "primary selection uses MaxDamage; see collection-completeness.json for the final same-job status",
                })
            config.write_text(yaml.safe_dump(recipe, sort_keys=False))
            train(root, output, python, workers, gpu_count, environment, run)
            summary = output / "training-summary.json"
            if summary.exists():
                print("[end-to-end] " + json.dumps(json.loads(summary.read_text())), flush=True)
            reference = root / "docs/briefs/06-artifacts/collection-attempt2"
            measurements = [reference / f"probe-self-{device}-{batch}-{repeat}-execution.json"
                            for device, batch in (("cpu", 32), ("cuda", 32), ("cuda", 64), ("cuda", 128))
                            for repeat in (1, 2)]
            reference_seconds = (sum(json.loads(path.read_text())["seconds"] for path in measurements)
                                 if all(path.exists() for path in measurements) else None)
            needed = reference_seconds * max(1.0, 4 / workers) * 1.2 + 60 if reference_seconds else None
            remaining = deadline - time.monotonic()
            finish_self = needed is not None and needed <= remaining
            write_json(output / "same-job-self-plan.json", {
                "label": "smoke planning, not measured completion", "reference_seconds": reference_seconds,
                "planning_seconds_with_margin": needed, "remaining_internal_seconds": remaining,
                "run_after_training_and_evaluation": finish_self,
            })
            if finish_self:
                matrix(("self",))
        requested = {f"probe-{opponent}-{device}-{batch}-{repeat}"
                     for opponent in ("self", "MaxDamage")
                     for device, batch in (("cpu", 32), ("cuda", 32), ("cuda", 64), ("cuda", 128))
                     for repeat in (1, 2)}
        write_json(output / "collection-completeness.json", {
            "label": "smoke", "requested_rows": len(requested), "completed_rows": len(saved["rows"]),
            "all_requested_rows_complete": requested.issubset({row["name"] for row in saved["rows"]}),
            "same_job_complete": not resume_roots and requested.issubset({row["name"] for row in saved["rows"]}),
        })
        status = "complete"
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        seconds = time.monotonic() - started
        write_json(output / "budget.json", {"label": "smoke", "status": status,
                   "commit": commit, "allocated_gpus": gpu_count, "job_seconds": seconds,
                   "gpu_hours": gpu_count * seconds / 3600,
                   "platform_timeout_seconds": platform_limit})


def train(root, output, python, workers, allocated_gpus, environment, run) -> None:
    import csv

    import yaml

    config = root / "configs/collect_gpu_check.yaml"
    recipe = yaml.safe_load(config.read_text())
    previous = output / "collect_gpu_check.txt"
    if previous.exists():
        recipe = yaml.safe_load(previous.read_text())
        config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    if recipe["env"]["rollout_workers"] != workers:
        raise ValueError("Training hardware differs from the recorded collection hardware")
    shutil.copy2(config, output / "collect_gpu_check.txt")
    # Preserve the inherited log.dir setting while saving checkpoints in Kaggle outputs.
    log = root / "runs/gen9ou_typed_token_policy_v1"
    log.parent.mkdir(parents=True, exist_ok=True)
    (output / "training").mkdir(exist_ok=True)
    if not log.exists():
        log.symlink_to(output / "training", target_is_directory=True)
    shim = root.parent / "timing"
    shim.mkdir(exist_ok=True)
    shutil.copy2(root / "docs/briefs/06-artifacts/training_timing.py", shim / "sitecustomize.py")
    previous_environment = {key: environment.get(key) for key in ("PYTHONPATH", "FOG_TIMING_OUTPUT")}
    environment.update(PYTHONPATH=str(shim) + os.pathsep + str(root),
                       FOG_TIMING_OUTPUT=str(output / "training-timing.jsonl"))
    command = [python, "-m", "fog.train", "--config", "configs/collect_gpu_check.yaml",
               "--trainer", "ppo"]
    states = sorted((output / "training/checkpoints").glob("ppo_[0-9]*.state.pt"),
                    key=lambda path: int(path.stem.split("_")[1].split(".")[0]))
    if states:
        trim_measurements(output, int(states[-1].stem.split("_")[1].split(".")[0]))
        command += ["--set", "ppo.resume_ckpt=" + str(states[-1])]
    oom_path = output / "oom-attempts.json"
    oom_attempts = json.loads(oom_path.read_text())["attempts"] if oom_path.exists() else []
    attempts_path = output / "trainer-attempts.json"
    attempts = json.loads(attempts_path.read_text())["attempts"] if attempts_path.exists() else []
    while True:
        try:
            run(command, "trainer")
            attempts.append(json.loads((output / "trainer-execution.json").read_text()))
            write_json(attempts_path, {"attempts": attempts})
            break
        except RuntimeError:
            attempts.append(json.loads((output / "trainer-execution.json").read_text()))
            write_json(attempts_path, {"attempts": attempts})
            text = (output / "trainer.txt").read_text()
            if "out of memory" not in text.lower() or "cuda" not in text.lower():
                raise
            # An update-boundary checkpoint requires its original exact-resume config.
            if recipe["ppo"].get("activation_checkpointing", False) or list(
                (output / "training/checkpoints").glob("*.state.pt")
            ):
                raise
            shutil.copy2(output / "trainer.txt", output / "trainer-oom-1024.txt")
            execution = json.loads((output / "trainer-execution.json").read_text())
            execution.update(minibatch_size=1024, error="CUDA out of memory")
            oom_attempts.append(execution)
            write_json(output / "oom-attempts.json", {"attempts": oom_attempts})
            timing = output / "training-timing.jsonl"
            if timing.exists():
                shutil.copy2(timing, output / "trainer-oom-1024-timing.jsonl")
                timing.write_text("")
            recipe["ppo"]["activation_checkpointing"] = True
            config.write_text(yaml.safe_dump(recipe, sort_keys=False))
            shutil.copy2(config, output / "collect_gpu_check.txt")
    for key, value in previous_environment.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    lines = (output / "trainer.txt").read_text().splitlines()
    (output / "trainer-tail.txt").write_text("\n".join(lines[-60:]) + "\n")
    shutil.copy2(output / "training/metrics.csv", output / "trainer-metrics.csv")
    shutil.copy2(config, output / "collect_gpu_check.txt")
    with (output / "trainer-metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    native_collection = sum(float(row["collection_seconds"]) for row in metrics)
    native_update = sum(float(row["update_seconds"]) for row in metrics)
    native_optimizer = sum(float(row["optimization_seconds"]) for row in metrics)
    events = [json.loads(line) for line in (output / "training-timing.jsonl").read_text().splitlines()]
    phases = [row for row in events if row["kind"] == "update_phase"]
    collections = [row for row in events if row["kind"] == "collection"]
    optimizers = [row for row in events if row["kind"] == "update"]
    collection = sum(row["seconds"] for row in collections)
    optimizer = sum(row["seconds"] for row in optimizers)
    update = sum(row["seconds"] for row in phases)
    if len(phases) != len(metrics):
        raise ValueError("Update-phase timers and trainer metrics disagree")
    rows = sum(row["rows"] for row in collections)
    games = sum(row["games"] for row in collections)
    assert rows >= 100000, "partial training, do not claim completed 100k run"
    assert rows == int(float(metrics[-1]["decisions"]))
    wall = sum(attempt["seconds"] for attempt in attempts)
    summary = {"label": "smoke", "decisions": rows, "games": games,
               "collection_seconds": collection, "update_seconds": update,
               "optimizer_update_seconds": optimizer,
               "native_collection_seconds": native_collection,
               "native_update_seconds": native_update,
               "native_optimization_seconds": native_optimizer,
               "update_assembly_diagnostics_logging_seconds": update - optimizer,
               "update_forward_seconds": sum(row["forward_seconds"] for row in optimizers),
               "update_loss_backward_optimizer_seconds": sum(
                   row["loss_backward_optimizer_seconds"] for row in optimizers),
               "trainer_wall_seconds": wall, "collection_decisions_per_second": rows / collection,
               "trainer_attempts": len(attempts),
               "latest_trainer_attempt_seconds": attempts[-1]["seconds"],
               "timing_scope": "committed update prefix for phase sums; all trainer attempts, including OOM or repeated unfinished work, in end-to-end wall",
               "failed_oom_attempt_seconds": sum(attempt["seconds"] for attempt in oom_attempts),
               "update_unique_rows_per_second": rows / update,
               "update_optimized_rows_per_second": sum(
                   row["optimized_rows"] for row in optimizers) / update,
               "optimizer_optimized_rows_per_second": sum(
                   row["optimized_rows"] for row in optimizers) / optimizer,
               "end_to_end_decisions_per_second": rows / wall,
               "end_to_end_games_per_second": games / wall,
               "effective_epochs": sum(row["optimized_rows"] for row in optimizers) / rows,
               "collection_wall_share": collection / wall, "update_wall_share": update / wall,
               "other_wall_share": (wall - collection - update) / wall,
               "allocated_gpus": allocated_gpus, "week_gpu_hours": 30,
               "week_session_hours": 30 / allocated_gpus,
               "week_decisions_projection": rows / wall * 30 * 3600 / allocated_gpus,
               "week_games_projection": games / wall * 30 * 3600 / allocated_gpus}
    write_json(output / "training-summary.json", summary)
    run([python, "-m", "fog.eval.showdown_direct_selection", "--pool",
         "data/teams/pool_a/core_v1_offensive.txt", "--device", "cuda", "--config",
         "configs/collect_gpu_check.yaml", "--checkpoint", "final=" + str(log / "checkpoints/ppo_final.pt"),
         "--checkpoint-config", "final=configs/collect_gpu_check.yaml", "--opponent",
         "maxdamage", "--games", "200", "--temperature", "0", "--out",
         str(output / "evaluation.json")], "evaluation")
    if collection > update:
        run([python, "-m", "fog.eval.showdown_rollout_probe", "--batch",
             str(recipe["env"]["rollout_batch"]), "--games", "128", "--opponent",
             "MaxDamage", "--device", "cuda"], "collection-split")


def trim_measurements(output: Path, decisions: int) -> None:
    """Drop work after the last saved PPO state before a continuation repeats it."""
    import csv

    timing = output / "training-timing.jsonl"
    lines = timing.read_text().splitlines(keepends=True)
    completed = 0
    cut = None
    for index, line in enumerate(lines):
        event = json.loads(line)
        if event["kind"] == "update_phase":
            completed += event["rows"]
            if completed == decisions:
                cut = index + 1
                break
    if cut is None:
        raise ValueError("Saved model state has no matching completed timing prefix; timing is partial")
    shutil.copy2(timing, output / "timing-before-resume.jsonl")
    timing.write_text("".join(lines[:cut]))
    metrics = output / "training/metrics.csv"
    with metrics.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(float(row["decisions"])) <= decisions]
    shutil.copy2(metrics, output / "metrics-before-resume.csv")
    with metrics.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
