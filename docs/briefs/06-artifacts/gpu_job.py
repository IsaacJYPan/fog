"""Private GPU measurement job; inject FOG_COMMIT before submitting.

Use FOG_STAGE=collection (1,800-second platform timeout) or training (3,000).
Internal deadlines reserve 300 seconds for export. Both include setup.
Completed probe rows and training checkpoints survive a rerun of the script
with the same output directory. Budget any new GPU session separately.
These caps leave 600 session seconds for the failed initial submission within
three GPU-hours even if Kaggle allocates two T4s. FOG_PLATFORM_TIMEOUT may
lower a cap further; the submission must request that same platform timeout.
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
    if stage not in {"collection", "training"}:
        raise ValueError("FOG_STAGE must be collection or training")
    maximum = 1800 if stage == "collection" else 3000
    platform_limit = int(os.environ.get("FOG_PLATFORM_TIMEOUT", maximum))
    if not 600 <= platform_limit <= maximum:
        raise ValueError("FOG_PLATFORM_TIMEOUT must be between 600 and this stage's maximum")
    deadline = started + platform_limit - 300
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    monitor = None
    gpu_count = 2  # Conservative until nvidia-smi reports the allocation.
    status = "failed"

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
        for opponent in ("self", "MaxDamage"):
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
    if recipe["env"]["rollout_workers"] != workers:
        raise ValueError("Training hardware differs from the recorded collection hardware")
    # Preserve the inherited log.dir setting while saving checkpoints in Kaggle outputs.
    log = root / "runs/gen9ou_typed_token_policy_v1"
    log.parent.mkdir(parents=True, exist_ok=True)
    (output / "training").mkdir(exist_ok=True)
    if not log.exists():
        log.symlink_to(output / "training", target_is_directory=True)
    shim = root.parent / "timing"
    shim.mkdir(exist_ok=True)
    shutil.copy2(root / "docs/briefs/06-artifacts/training_timing.py", shim / "sitecustomize.py")
    environment.update(PYTHONPATH=str(shim) + os.pathsep + str(root),
                       FOG_TIMING_OUTPUT=str(output / "training-timing.jsonl"))
    command = [python, "-m", "fog.train", "--config", "configs/collect_gpu_check.yaml",
               "--trainer", "ppo"]
    states = sorted((output / "training/checkpoints").glob("ppo_[0-9]*.state.pt"),
                    key=lambda path: int(path.stem.split("_")[1].split(".")[0]))
    if states:
        command += ["--set", "ppo.resume_ckpt=" + str(states[-1])]
    oom_attempts = []
    while True:
        try:
            run(command, "trainer")
            break
        except RuntimeError:
            text = (output / "trainer.txt").read_text()
            if "out of memory" not in text.lower() or "cuda" not in text.lower():
                raise
            minibatch = int(recipe["ppo"]["minibatch_size"])
            # An update-boundary checkpoint requires its original exact-resume config.
            if minibatch <= 1 or list((output / "training/checkpoints").glob("*.state.pt")):
                raise
            shutil.copy2(output / "trainer.txt", output / f"trainer-oom-{minibatch}.txt")
            execution = json.loads((output / "trainer-execution.json").read_text())
            execution.update(minibatch_size=minibatch, error="CUDA out of memory")
            oom_attempts.append(execution)
            write_json(output / "oom-attempts.json", {"attempts": oom_attempts})
            recipe["ppo"]["minibatch_size"] = minibatch // 2
            config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    environment.pop("FOG_TIMING_OUTPUT")
    environment.pop("PYTHONPATH")
    lines = (output / "trainer.txt").read_text().splitlines()
    (output / "trainer-tail.txt").write_text("\n".join(lines[-60:]) + "\n")
    shutil.copy2(output / "training/metrics.csv", output / "trainer-metrics.csv")
    shutil.copy2(config, output / "collect_gpu_check.txt")
    with (output / "trainer-metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    collection = sum(float(row["collection_seconds"]) for row in metrics)
    update = sum(float(row["update_seconds"]) for row in metrics)
    rows = sum(int(float(row["collection_rows"])) for row in metrics)
    games = sum(int(float(row["collection_games"])) for row in metrics)
    wall = json.loads((output / "trainer-execution.json").read_text())["seconds"]
    summary = {"label": "smoke", "decisions": rows, "games": games,
               "collection_seconds": collection, "update_seconds": update,
               "update_forward_seconds": sum(float(row["update_forward_seconds"]) for row in metrics),
               "update_loss_backward_optimizer_seconds": sum(
                   float(row["update_other_seconds"]) for row in metrics),
               "trainer_wall_seconds": wall, "collection_decisions_per_second": rows / collection,
               "update_unique_rows_per_second": rows / update,
               "end_to_end_decisions_per_second": rows / wall,
               "collection_wall_share": collection / wall, "update_wall_share": update / wall,
               "other_wall_share": (wall - collection - update) / wall,
               "allocated_gpus": allocated_gpus, "week_gpu_hours": 30,
               "week_session_hours": 30 / allocated_gpus,
               "week_decisions_projection": rows / wall * 30 * 3600 / allocated_gpus,
               "week_games_projection": games / wall * 30 * 3600 / allocated_gpus}
    write_json(output / "training-summary.json", summary)
    if collection > update:
        run([python, "-m", "fog.eval.showdown_rollout_probe", "--batch",
             str(recipe["env"]["rollout_batch"]), "--games", "128", "--opponent",
             "MaxDamage", "--pool", "data/teams/pool_a", "--device", "cuda"], "collection-split")
    run([python, "-m", "fog.eval.showdown_direct_selection", "--pool",
         "data/teams/pool_a/core_v1_offensive.txt", "--device", "cuda", "--config",
         "configs/collect_gpu_check.yaml", "--checkpoint", "final=" + str(log / "checkpoints/ppo_final.pt"),
         "--checkpoint-config", "final=configs/collect_gpu_check.yaml", "--opponent",
         "maxdamage", "--games", "200", "--temperature", "0", "--out",
         str(output / "evaluation.json")], "evaluation")


if __name__ == "__main__":
    main()
