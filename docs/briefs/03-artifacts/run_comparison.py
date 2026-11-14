"""Run one brief 03 comparison, retaining commands, walls and the A interruption.

Invoke from the repository root with R, A, S, or B, or R-default for the required
diagnostic without resume/selection options. This calls the existing trainer and
holdout reporter. It does not modify their configuration beyond the brief.
"""
import csv
import json
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

name = sys.argv[1]
games, epochs, directory = {
    "R": (1000, 6, "g1000e6"), "A": (1000, 18, "g1000e18"),
    "S": (1000, 6, "g1000e6s2"), "B": (3000, 18, "g3000e18"),
    "R-default": (1000, 6, "g1000e6-default"),
}[name]
work = Path("runs/blueprint") / directory
work.mkdir(parents=True, exist_ok=True)
artifacts = Path("docs/briefs/03-artifacts")
metadata_path = artifacts / f"{name}-execution.json"
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
    "run": name, "games": games, "epochs": epochs,
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "attempts": [], "reports": [],
}
command = [sys.executable, "-u", "-m", "fog.training.bc", "--config",
           "configs/replay_blueprint_cpu.yaml", "--set",
           f"bc.manifest=runs/blueprint/split/train_{games}.parquet",
           "bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet",
           "bc.streaming=true", f"bc.epochs={epochs}", "bc.early_stopping_patience=0"]
if name != "R-default":
    command.extend(["bc.selection_metric=action_ce", "bc.resume=true"])
command.append(f"log.dir={work}")
if name == "S":
    command.append("seed=20261202")
metadata["command"] = shlex.join(command)


def save():
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def metric_rows():
    path = work / "metrics.csv"
    # The trainer rewrites this file after each epoch. During polling, an empty
    # file, incomplete header or partial row means the next snapshot is not ready.
    try:
        contents = path.read_text()
        if not contents.endswith("\n"):
            return []
        reader = csv.DictReader(contents.splitlines())
        if not {"epoch", "eval_action_ce"} <= set(reader.fieldnames or []):
            return []
        rows = list(reader)
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                return []
            int(row["epoch"])
            float(row["eval_action_ce"])
        return rows
    except (OSError, ValueError, csv.Error):
        return []


while True:
    attempt = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    metadata["attempts"].append(attempt)
    save()
    started = time.perf_counter()
    with (work / "train-output.txt").open("a") as output:
        output.write("\n$ " + shlex.join(command) + "\n")
        output.flush()
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        interrupted = False
        while child.poll() is None:
            if name == "A" and not metadata.get("interruption_performed"):
                rows = metric_rows()
                state = work / "checkpoints/bc_state.pt"
                if rows and int(rows[-1]["epoch"]) >= 2 and state.is_file():
                    # The CSV precedes the atomic state write. Check the completed state
                    # before killing, so the recorded restart really begins after epoch 2.
                    import torch

                    completed = torch.load(state, weights_only=False, map_location="cpu")["epoch"]
                    if completed >= 2:
                        # Capture the already-read snapshot. A later CSV row can
                        # precede its checkpoint and must be repeated on resume.
                        before = [row for row in rows if int(row["epoch"]) <= completed]
                        child.send_signal(signal.SIGTERM)
                        child.wait(timeout=30)
                        metadata["interruption_performed"] = True
                        metadata["metrics_before_resume"] = before
                        metadata["completed_epoch_at_interruption"] = completed
                        interrupted = True
                        print(f"A killed after completed epoch {completed}; restarting", flush=True)
                        save()
                        break
            time.sleep(1)
        code = child.wait()
    attempt.update(exit_code=code, seconds=time.perf_counter() - started,
                   deliberately_interrupted=interrupted)
    save()
    if interrupted:
        continue
    if code:
        raise SystemExit(code)
    break

rows = metric_rows()
if not rows:
    raise RuntimeError("training exited without a complete metrics.csv")
metadata["completed_epochs"] = [int(row["epoch"]) for row in rows]
if name == "A":
    before = metadata["metrics_before_resume"]
    metadata["earlier_metrics_unchanged"] = rows[:len(before)] == before
    assert metadata["earlier_metrics_unchanged"]
(artifacts / f"{name}-metrics.csv").write_text((work / "metrics.csv").read_text())
with (work / "train-output.txt").open() as stream:
    (artifacts / f"{name}-tail.txt").write_text("".join(deque(stream, maxlen=60)))
save()
for kind, checkpoint in (("best", "bc_best.pt"), ("final", "bc_final.pt")):
    report = work / f"holdout-{kind}.json"
    score = [sys.executable, "-u", "-m", "fog.eval.replay_holdout_report",
             "--config", "configs/replay_blueprint_cpu.yaml", "--manifest",
             "runs/blueprint/split/holdout_manifest.parquet", "--checkpoint",
             str(work / "checkpoints" / checkpoint), "--out", str(report)]
    started = time.perf_counter()
    with (artifacts / f"{name}-holdout-{kind}-output.txt").open("w") as output:
        subprocess.run(score, stdout=output, stderr=subprocess.STDOUT, check=True)
    metadata["reports"].append({"command": shlex.join(score),
                                "seconds": time.perf_counter() - started})
    (artifacts / f"{name}-holdout-{kind}.json").write_text(report.read_text())
    save()
if name in {"R", "R-default"}:
    old = list(csv.DictReader(Path("docs/briefs/01-artifacts/g1000-metrics.csv").open()))
    previous = [format(float(row["eval_action_ce"]), ".3f") for row in old]
    current = [format(float(row["eval_action_ce"]), ".3f") for row in rows]
    metadata["brief01_policy_nll_rounded"] = previous
    metadata["policy_nll_rounded"] = current
    metadata["reproduces_brief01_to_three_decimals"] = previous == current
    save()
    if name == "R" and previous != current:
        raise SystemExit("R did not reproduce brief 01; diagnose before other runs")
if name == "R-default":
    reference = list(csv.DictReader((artifacts / "R-metrics.csv").open()))
    ignored = {"seconds_per_epoch", "train_rows_per_second"}
    current_metrics = [{key: value for key, value in row.items() if key not in ignored}
                       for row in rows]
    reference_metrics = [{key: value for key, value in row.items() if key not in ignored}
                         for row in reference]
    metadata["matches_R_metrics_except_elapsed_time"] = current_metrics == reference_metrics
    save()
print(json.dumps(metadata, indent=2), flush=True)
