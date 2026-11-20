"""Score policy/value checkpoints, test lead retention, then run brief 13 play cells.

Existing reports and the trainer's epoch state allow interruption and continuation.
Run from the repository root, with the locked Python environment and PYTHONPATH=.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import polars as pl


def run(command, destination, *, threads=4):
    started = time.monotonic()
    with destination.with_suffix(".txt").open("a") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    record = {"command": command, "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "math_threads": threads, "wall_seconds": time.monotonic() - started,
        "exit_code": result.returncode}
    destination.with_name(destination.stem + "-command.json").write_text(
        json.dumps(record, indent=2) + "\n")
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--play-only", action="store_true",
                        help="resume play after completing the three checkpoint scoring stages")
    parser.add_argument("--validation-workers", type=int, default=1, choices=range(1, 5),
                        help="parallel validation cells, with one math thread per process")
    parser.add_argument("--anchor-checkpoint", type=Path,
                        help="recovered retained imitation checkpoint for the G2 head-to-head")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/briefs/13-artifacts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    configs = {"original": "configs/replay_mechanical_cpu.yaml",
               "lead": "configs/replay_preview_cpu.yaml"}
    checkpoints = {"original": args.checkpoint_dir / "bc_best.pt",
                   "value": args.checkpoint_dir / "bc_best_value.pt",
                   "lead": args.work / "lead/checkpoints/bc_best.pt"}

    if args.anchor_checkpoint:
        anchor_digest = hashlib.sha256(args.anchor_checkpoint.read_bytes()).hexdigest()
        if anchor_digest != "17de023acd346c8ac3490aa3a8a75b4139cc9c58b691723d4a8e0d30322664bf":
            raise ValueError("the selected checkpoint does not match the retained A3 anchor")
        configs["A3"] = "configs/gen9ou_typed_token_selfplay.yaml"
        checkpoints["A3"] = args.anchor_checkpoint

    def report(label, checkpoint, config):
        for module, suffix in (("replay_slot_order_check", "slot"),
                               ("replay_holdout_report", "holdout")):
            output = args.out / f"{args.run}-{label}-{suffix}.json"
            if output.exists():
                json.loads(output.read_text())
                continue
            run([sys.executable, "-m", f"fog.eval.{module}", "--config", config,
                 "--manifest", str(args.holdout_manifest), "--checkpoint", str(checkpoint),
                 "--batch-size", "256", "--out", str(output)], output)

    if args.play_only:
        for label in ("original", "value", "lead"):
            for suffix in ("slot", "holdout"):
                saved = json.loads((args.out / f"{args.run}-{label}-{suffix}.json").read_text())
                if Path(saved["checkpoint"]).resolve() != checkpoints[label].resolve():
                    raise ValueError(f"{label} report belongs to another checkpoint")
    else:
        report("original", checkpoints["original"], configs["original"])
        report("value", checkpoints["value"], configs["original"])
        # Match brief 09's short lead stage: the nested 1,000-game prefix, eight epochs.
        preview_manifest = args.work / "lead_train_1000.parquet"
        pl.read_parquet(args.train_manifest).head(1000).write_parquet(preview_manifest)
        run([sys.executable, "-m", "fog.training.bc", "--config", configs["lead"], "--set",
             f"bc.init_ckpt={checkpoints['original']}", f"bc.manifest={preview_manifest}",
             f"bc.holdout_manifest={args.holdout_manifest}", "bc.epochs=8", "bc.resume=true",
             "loss.belief_coef=0", "loss.belief_chain_coef=0", "loss.skip_unweighted_belief=true",
             f"log.dir={args.work / 'lead'}"], args.out / f"{args.run}-lead-training.json")
        for name in ("metrics.csv", "scope_verification.json"):
            (args.out / f"{args.run}-lead-{name}").write_bytes((args.work / "lead" / name).read_bytes())
        report("lead", checkpoints["lead"], configs["lead"])

    def play(pool, games, temperature, labels, suffix, *, against_anchor=False, threads=4):
        output = args.out / f"{args.run}-{suffix}.json"
        command = [sys.executable, "-m", "fog.eval.showdown_direct_selection", "--config",
                   configs["original"], "--device", "cpu", "--pool", pool, "--games", str(games),
                   "--temperature", str(temperature), "--action-stats", "--resume", "--out", str(output)]
        if against_anchor:
            command += ["--head-to-head", "--compatible-checkpoint", "A3"]
        else:
            command += ["--opponent", "simpleheuristics", "--opponent", "maxdamage"]
        for label in labels:
            command += ["--checkpoint", f"{label}={checkpoints[label]}",
                        "--checkpoint-config", f"{label}={configs[label]}"]
        run(command, output, threads=threads)
        data = json.loads(output.read_text())
        assert data["status"] == "complete"
        for row in data["rows"]:
            label = row["a"]
            row["checkpoint_sha256"] = hashlib.sha256(checkpoints[label].read_bytes()).hexdigest()
            row["temperature"] = temperature
            row["lead_rule"] = ("argmax preview logits" if temperature == 0
                                else "sample preview logits at temperature 0.5")
            row["lead_supervision"] = "brief 09 preview stage" if label == "lead" else "none"
        output.write_text(json.dumps(data, indent=2) + "\n")
        return data

    smoke = play("data/teams/pool_b", 400, 0, ["original", "lead"], "pool-b-play")
    before = json.loads((args.out / f"{args.run}-original-slot.json").read_text())
    after = json.loads((args.out / f"{args.run}-lead-slot.json").read_text())
    delta_nll = after["policy"]["all"]["nll_stored"] - before["policy"]["all"]["nll_stored"]
    rows = {r["a"]: r for r in smoke["rows"] if r["b"] == "simpleheuristics"}
    assert all(r["failures"] == r["truncated"] == 0 for r in rows.values())
    # Pair the same four-game team/side blocks; interval is for the lead-minus-original score.
    differences = (np.array(rows["lead"]["game_scores"], dtype=float)
                   - np.array(rows["original"]["game_scores"], dtype=float)).reshape(-1, 4).mean(1)
    rng = np.random.default_rng(0)
    interval = np.quantile(differences[rng.integers(len(differences), size=(20000, len(differences)))].mean(1),
                           [0.025, 0.975]).tolist()
    selected = "lead" if abs(delta_nll) <= 0.005 and interval[1] >= 0 else "original"
    decision = {"policy_nll_delta_lead_minus_original": delta_nll,
                "simpleheuristics_score_delta": float(differences.mean()),
                "paired_block_bootstrap_interval95": interval, "bootstrap_samples": 20000,
                "rule": "absolute NLL change <= 0.005 and no significant paired-block score decrease",
                "selected": selected, "checkpoint": str(checkpoints[selected])}
    (args.out / f"{args.run}-lead-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if Path("data/teams/replay_v1/validation").exists():
        # Each cell has its own simulator, seeded schedule and resumable report. Keep
        # independent cells on separate cores without changing team or side assignments.
        threads = 4 if args.validation_workers == 1 else 1
        tasks = []
        with ThreadPoolExecutor(max_workers=args.validation_workers) as pool:
            for temperature in (0, 0.5):
                tasks.append(pool.submit(play, "data/teams/replay_v1/validation", 1000, temperature,
                                         [selected], f"validation-t{temperature}-play", threads=threads))
                if args.anchor_checkpoint:
                    tasks.append(pool.submit(play, "data/teams/replay_v1/validation", 1000, temperature,
                                             [selected, "A3"], f"validation-t{temperature}-A3",
                                             against_anchor=True, threads=threads))
            for task in tasks:
                task.result()


if __name__ == "__main__":
    main()
