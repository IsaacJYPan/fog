"""Private single-GPU collection job; inject FOG_COMMIT before submitting.

Submit with a single P100 and a 3,600-second platform timeout. This script's
3,300-second deadline includes setup. Completed probe rows survive a rerun.
Account settings belong only in the private submission directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
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
    commit = os.environ["FOG_COMMIT"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("FOG_COMMIT must identify the pushed commit")
    deadline = started + 3300
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    monitor = None
    status = "failed"

    def run(command: list[str], name: str, *, cwd: Path = root) -> str:
        path = output / (name + ".txt")
        record = {"command": command, "commit": commit, "status": "running"}
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
                      status="complete" if code == 0 else "failed")
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
        if len(gpu) != 1:
            raise RuntimeError("Request exactly one GPU to keep the GPU-hour budget explicit")
        with (output / "gpu-memory.txt").open("w") as stream:
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,name,memory.used,utilization.gpu",
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
                   "commit": commit, "allocated_gpus": 1, "job_seconds": seconds,
                   "gpu_hours": seconds / 3600, "platform_timeout_seconds": 3600})


if __name__ == "__main__":
    main()
