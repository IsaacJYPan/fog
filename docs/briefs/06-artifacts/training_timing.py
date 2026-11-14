"""Optional sitecustomize for timing this measurement, without editing the trainer.

Copy to a temporary directory as sitecustomize.py and add that directory to
PYTHONPATH for `python -m fog.train` only. Set FOG_TIMING_OUTPUT to a JSONL
path. Spawned actors inherit FOG_TIMING_PARENT and do not install these hooks.
Times include explicit CUDA synchronization at learner update boundaries.
Update rows/s counts unique collected rows, including the configured epochs.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import time


def install(path: Path) -> None:
    import torch

    from fog.training import ppo
    from fog.training.dashboard import TrainingDashboard
    from fog.training.direct_parallel import PersistentProcessPool

    interval = {"collection_seconds": 0.0, "collection_rows": 0, "collection_games": 0}

    def record(row: dict) -> None:
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("[timing] " + json.dumps(row), flush=True)

    original_map = PersistentProcessPool.map

    @functools.wraps(original_map)
    def timed_map(self, *args, **kwargs):
        start = time.perf_counter()
        results = original_map(self, *args, **kwargs)
        seconds = time.perf_counter() - start
        games = [game for buffer, _ in results for game in buffer._games.values()]
        rows = sum(len(game.actions) for game in games)
        interval["collection_seconds"] += seconds
        interval["collection_rows"] += rows
        interval["collection_games"] += len(games)
        record({"kind": "collection", "seconds": seconds, "rows": rows, "games": len(games)})
        return results

    PersistentProcessPool.map = timed_map
    original_update = ppo.ppo_update

    @functools.wraps(original_update)
    def timed_update(agent, optimizer, batch, *args, **kwargs):
        torch.cuda.synchronize()
        forward_seconds = 0.0
        forward_start = 0.0

        def before_forward(_module, _args):
            nonlocal forward_start
            torch.cuda.synchronize()
            forward_start = time.perf_counter()

        def after_forward(_module, _args, _output):
            nonlocal forward_seconds
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - forward_start

        before = agent.register_forward_pre_hook(before_forward)
        after = agent.register_forward_hook(after_forward)
        start = time.perf_counter()
        try:
            metrics = original_update(agent, optimizer, batch, *args, **kwargs)
            torch.cuda.synchronize()
        finally:
            before.remove()
            after.remove()
        seconds = time.perf_counter() - start
        rows = len(batch["actions"])
        config = args[0] if args else kwargs["cfg"]
        minibatch = int(config.get("ppo", {}).get("minibatch_size", 2048))
        batches_per_epoch = (rows + minibatch - 1) // minibatch
        steps = int(metrics["optimizer_steps"])
        epochs, partial = divmod(steps, batches_per_epoch)
        optimized_rows = epochs * rows + min(partial * minibatch, rows)
        interval.update(update_seconds=seconds, update_rows=rows,
                        update_optimized_rows=optimized_rows,
                        update_forward_seconds=forward_seconds,
                        update_other_seconds=seconds - forward_seconds)
        record({"kind": "update", "seconds": seconds, "rows": rows,
                "forward_seconds": forward_seconds,
                "loss_backward_optimizer_seconds": seconds - forward_seconds,
                "rows_per_second": rows / seconds,
                "optimized_rows": optimized_rows,
                "optimized_rows_per_second": optimized_rows / seconds,
                "optimizer_steps": metrics.get("optimizer_steps"),
                "peak_learner_allocated_bytes": torch.cuda.max_memory_allocated()})
        return metrics

    ppo.ppo_update = timed_update
    original_log = TrainingDashboard.log

    @functools.wraps(original_log)
    def timed_log(self, step, metrics):
        if "update_seconds" in interval:
            metrics.update(interval)
            metrics["collection_rows_per_second"] = (
                interval["collection_rows"] / interval["collection_seconds"]
            )
            metrics["update_rows_per_second"] = interval["update_rows"] / interval["update_seconds"]
            metrics["update_optimized_rows_per_second"] = (
                interval["update_optimized_rows"] / interval["update_seconds"]
            )
            metrics["peak_learner_allocated_bytes"] = torch.cuda.max_memory_allocated()
            interval.clear()
            interval.update(collection_seconds=0.0, collection_rows=0, collection_games=0)
        return original_log(self, step, metrics)

    TrainingDashboard.log = timed_log


if os.environ.get("FOG_TIMING_OUTPUT") and not os.environ.get("FOG_TIMING_PARENT"):
    os.environ["FOG_TIMING_PARENT"] = str(os.getpid())
    install(Path(os.environ["FOG_TIMING_OUTPUT"]))
