"""Run a bounded, audited PPO trial and verify the resulting weights and optimizer state."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def run(device: str, output: Path) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator; CUDA is unavailable.")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "trial_report.json"
    if report_path.exists():
        raise FileExistsError(f"Trial already reported at {report_path}; choose a new --out.")
    job = yaml.safe_load(Path("configs/automated_gen9ou_kaggle_trial.yaml").read_text())
    job["output_dir"] = output.as_posix()
    job["safety"]["device"] = device
    job["overrides"].update({
        "device": device,
        "preflight.report": (output / "audits/direct/summary.json").as_posix(),
        "preflight.replay_report": (output / "audits/replay/summary.json").as_posix(),
    })
    job_path = output / "trial_job.yaml"
    text = yaml.safe_dump(job, sort_keys=False)
    if job_path.exists() and job_path.read_text() != text:
        raise RuntimeError("Existing trial settings differ; choose a new --out.")
    job_path.write_text(text, encoding="utf-8")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "fog.scripts.automated_training", "run",
               "--job", str(job_path), "--max-blocks", "1"]
    print("Running audited PPO trial; detailed logs are under", output, flush=True)
    subprocess.run(command, env=environment, check=True)
    checkpoints = list(output.rglob("ppo_final.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one final checkpoint, found {len(checkpoints)}")
    checkpoint = checkpoints[0]
    before = torch.load(job["initial_checkpoint"], map_location="cpu", weights_only=True)
    after = torch.load(checkpoint, map_location="cpu", weights_only=True)
    changed = [key for key in before if key in after and not torch.equal(before[key], after[key])]
    finite = all(torch.isfinite(value).all().item() for value in after.values())
    state = torch.load(checkpoint.with_suffix(".state.pt"), map_location="cpu", weights_only=False)
    progress = state["progress"]
    metrics = progress["metrics"]
    finite_metrics = all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
    audits = {name: json.loads((output / f"audits/{name}/summary.json").read_text())
              for name in ("direct", "replay")}
    passed = bool(changed and finite and finite_metrics and state["optimizer"]["state"]
                  and progress["decisions_seen"] >= 32
                  and all(audit["passed"] for audit in audits.values()))
    report = {
        "passed": passed, "purpose": "pipeline trial, not evidence of playing-strength improvement",
        "device": device, "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "initial_checkpoint": job["initial_checkpoint"], "checkpoint": str(checkpoint),
        "decisions": progress["decisions_seen"], "changed_tensors": len(changed),
        "all_weights_finite": finite, "all_metrics_finite": finite_metrics,
        "metrics": metrics,
        "audits": {name: {key: audit.get(key) for key in
                            ("passed", "games_completed", "optimizer_steps", "error_count")}
                   for name, audit in audits.items()},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError(f"Trial validation failed; see {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/gen9ou_kaggle_trial"))
    args = parser.parse_args()
    run(args.device, args.out)


if __name__ == "__main__":
    main()
