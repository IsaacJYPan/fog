"""Kill and restart the genuine 600-game pipeline; run from the repository root.

The cached immutable source shard may be copied into work/cache before invocation.
Other conversion work may run concurrently; these walls are not idle CPU profiles.
"""
import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

work = Path("runs/blueprint_runner_smoke")
command = [sys.executable, "-u", "-m", "fog.scripts.run_replay_blueprint",
           "--games", "600", "--work-dir", str(work), "--device", "cpu",
           "--batch-size", "256", "--epochs", "2", "--chronological-holdout-games", "100"]
records = {"command": command,
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "source": "pinned public corpus; first source shard cached from main import",
           "timing_context": "main 22000-game converter also running"}
started = time.perf_counter()
with (work / "first-attempt.txt").open("w") as output:
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    state = work / "training/checkpoints/bc_state.pt"
    while not state.is_file():
        if child.poll() is not None:
            raise RuntimeError(f"runner exited before epoch state: {child.returncode}")
        time.sleep(1)
    child.send_signal(signal.SIGTERM)
    records["interrupted_exit_code"] = child.wait(timeout=30)
records["first_attempt_seconds"] = time.perf_counter() - started
metrics = work / "training/metrics.csv"
before = list(csv.DictReader(metrics.open()))
records["metrics_epochs_after_interrupt"] = [int(row["epoch"]) for row in before]
records["final_checkpoint_after_interrupt"] = (work / "training/checkpoints/bc_final.pt").exists()
(work / "metrics-before-resume.csv").write_text(metrics.read_text())
(work / "interruption-test.json").write_text(json.dumps(records, indent=2) + "\n")
started = time.perf_counter()
with (work / "resumed-attempt.txt").open("w") as output:
    completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
records["resumed_exit_code"] = completed.returncode
records["resumed_seconds"] = time.perf_counter() - started
after = list(csv.DictReader(metrics.open()))
records["metrics_epochs_after_resume"] = [int(row["epoch"]) for row in after]
records["earlier_metrics_unchanged"] = after[:len(before)] == before
(work / "interruption-test.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2), flush=True)
assert records["interrupted_exit_code"] == 143, records
assert not records["final_checkpoint_after_interrupt"], records
assert records["metrics_epochs_after_interrupt"] == [0, 1], records
assert records["metrics_epochs_after_resume"] == [0, 1, 2], records
assert records["earlier_metrics_unchanged"], records
raise SystemExit(completed.returncode)
