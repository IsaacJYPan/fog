"""Profile the CPU replay recipe without changing its batches, losses or optimizer.

Run while the machine is idle, before and after a loader optimisation::

    python -m fog.scripts.profile_replay_training --out runs/blueprint/profile-before

The JSON includes the first 20 losses for a fixed-seed comparison. The separate
batch-only pass includes shard loading; neither pass retains batches in memory.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import pstats
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch

from fog.training.bc import bc_loss


def _batches_forever(dataset, batch_size: int, seed: int):
    epoch = 0
    while True:
        yielded = False
        for batch in dataset.batches(batch_size, shuffle=True, seed=seed + epoch, device="cpu"):
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("training manifest yielded no batches")
        epoch += 1


def measure_steps(agent, dataset, cfg: dict, *, warmup: int, steps: int,
                  loss_fn=bc_loss) -> dict:
    """Time the ordinary full-scope CPU update, with passive model/head hooks."""
    if warmup < 0 or steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")
    bc = cfg.get("bc", {})
    batch_size = int(bc.get("batch_size", 512))
    epochs = int(bc.get("epochs", 10))
    lr = float(bc.get("lr", 3e-4))
    opt = torch.optim.AdamW(agent.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, math.ceil(len(dataset) / batch_size)) * epochs,
        eta_min=float(bc.get("lr_final", 3e-5)),
    )
    clocks = {"model_forward": 0.0, "belief_heads": 0.0}
    starts: dict[str, list[float]] = {name: [] for name in clocks}
    handles = []

    def attach(module, name):
        handles.append(module.register_forward_pre_hook(
            lambda _module, _args: starts[name].append(time.perf_counter())
        ))

        def done(_module, _args, _result):
            clocks[name] += time.perf_counter() - starts[name].pop()

        handles.append(module.register_forward_hook(done))

    attach(agent, "model_forward")
    # A single belief module is called for both marginal and teacher-forced heads.
    seen = set()
    for name in ("belief", "self_belief"):
        module = getattr(agent, name, None)
        if module is not None and id(module) not in seen:
            attach(module, "belief_heads")
            seen.add(id(module))
    iterator = _batches_forever(dataset, batch_size, int(cfg.get("seed", 0)))
    totals = {"batch_build": 0.0, "model_forward": 0.0,
              "backward_optimizer": 0.0, "other": 0.0, "total": 0.0,
              "belief_heads": 0.0}
    losses = []
    rows = 0
    agent.train()
    try:
        for step in range(warmup + steps):
            started = time.perf_counter()
            batch = next(iterator)
            built = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            before = dict(clocks)
            total, _ = loss_fn(agent, batch, cfg.get("loss", {}))
            backward_start = time.perf_counter()
            total.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            backward_end = time.perf_counter()
            schedule.step()
            ended = time.perf_counter()
            if len(losses) < 20:
                losses.append(float(total.detach()))
            if step >= warmup:
                elapsed = ended - started
                building = built - started
                forward = clocks["model_forward"] - before["model_forward"]
                backward = backward_end - backward_start
                totals["batch_build"] += building
                totals["model_forward"] += forward
                totals["backward_optimizer"] += backward
                totals["other"] += elapsed - building - forward - backward
                totals["total"] += elapsed
                totals["belief_heads"] += clocks["belief_heads"] - before["belief_heads"]
                rows += len(batch["action"])
    finally:
        iterator.close()
        for handle in handles:
            handle.remove()
    return {
        "warmup_steps": warmup, "measured_steps": steps, "measured_rows": rows,
        "seconds_per_step": {name: seconds / steps for name, seconds in totals.items()},
        "belief_share_of_model_forward": (
            totals["belief_heads"] / totals["model_forward"]
            if totals["model_forward"] else None
        ),
        "training_rows_per_second": rows / totals["total"],
        "first_20_training_losses": losses,
        "torch_threads": torch.get_num_threads(),
    }


def measure_batch_building(dataset, batch_size: int, seed: int, steps: int) -> dict:
    if steps < 1:
        raise ValueError("batch-only steps must be positive")
    iterator = _batches_forever(dataset, batch_size, seed)
    rows = 0
    started = time.perf_counter()
    try:
        for _ in range(steps):
            batch = next(iterator)
            rows += len(batch["action"])
            del batch
    finally:
        iterator.close()
    seconds = time.perf_counter() - started
    return {"steps": steps, "rows": rows, "seconds": seconds,
            "rows_per_second": rows / seconds}


def main() -> None:
    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent
    from fog.training.dataset import StreamingReplayDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/replay_blueprint_cpu.yaml"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("runs/blueprint/split/train_1000.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config_with_args(args.config, args.set)
    bc = cfg.get("bc", {})
    if bc.get("scope", "full") != "full" or bc.get("auxiliary_manifest"):
        parser.error("this profiler covers the single-source full-scope replay recipe")
    if min(args.steps, args.profile_steps) < 1 or args.warmup < 0:
        parser.error("steps must be positive; warmup must be non-negative")
    if bc.get("torch_threads") is not None:
        torch.set_num_threads(int(bc["torch_threads"]))
    ds = StreamingReplayDataset(args.manifest, shard_buffer=int(bc.get("shard_buffer", 64)))
    args.out.mkdir(parents=True, exist_ok=True)

    def agent():
        torch.manual_seed(int(cfg.get("seed", 0)))
        return FogAgent.from_tables(cfg).cpu()

    result = measure_steps(agent(), ds, cfg, warmup=args.warmup, steps=args.steps)
    result["batch_building_only"] = measure_batch_building(
        ds, int(bc.get("batch_size", 512)), int(cfg.get("seed", 0)), args.steps
    )
    profile = cProfile.Profile()
    profile.runcall(measure_steps, agent(), ds, cfg, warmup=0, steps=args.profile_steps)
    output = io.StringIO()
    pstats.Stats(profile, stream=output).strip_dirs().sort_stats("cumulative").print_stats(15)
    (args.out / "cprofile.txt").write_text(output.getvalue(), encoding="utf-8")
    result.update(device="cpu", torch_version=str(torch.__version__),
                  cuda_available=torch.cuda.is_available(), manifest=str(args.manifest),
                  command=shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    (args.out / "timing.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
