"""Time two normal PPO updates with the brief 06 T4 collection recipe."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from fog.config import load_config_with_args
from fog.training import ppo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--belief-fraction", type=float, required=True)
    parser.add_argument("--mixed-precision", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    config = load_config_with_args("configs/collect_gpu_check.yaml", [
        f"log.dir={args.out}", "league.snapshot_every_decisions=8000",
        f"env.rollout_workers={min(4, len(os.sched_getaffinity(0)))}",
        f"ppo.belief_row_fraction={args.belief_fraction}",
        f"ppo.mixed_precision={str(args.mixed_precision).lower()}",
    ])
    original = ppo._maybe_chunk_update
    updates = []
    started = time.monotonic()

    def measure(setup, *positional, **kwargs):
        result = original(setup, *positional, **kwargs)
        if result != "updated":
            return result
        fields = ("optimization_seconds", "optimizer_steps", "optimizer_rows",
                  "collection_seconds", "update_seconds", "decisions")
        updates.append({key: setup.metrics[key] for key in fields})
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        seconds = sum(row["optimization_seconds"] for row in updates)
        steps = sum(row["optimizer_steps"] for row in updates)
        rows = sum(row["optimizer_rows"] for row in updates)
        report = {
            "smoke": True, "updates": updates, "seconds_through_checkpoint": elapsed,
            "seconds_per_optimizer_step": seconds / steps, "processed_rows_per_second": rows / seconds,
            "end_to_end_decisions_per_second": setup.decisions_seen / elapsed,
            "peak_learner_allocated_bytes": torch.cuda.max_memory_allocated(),
            "belief_row_fraction": args.belief_fraction, "mixed_precision": args.mixed_precision,
            "torch": str(torch.__version__), "device": torch.cuda.get_device_name(0),
        }
        (args.out / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
        return "halt" if len(updates) == 2 else result

    ppo._maybe_chunk_update = measure
    torch.cuda.reset_peak_memory_stats()
    try:
        ppo.train(config)
    finally:
        ppo._maybe_chunk_update = original
    assert len(updates) == 2


if __name__ == "__main__":
    main()
