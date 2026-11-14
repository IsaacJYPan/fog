"""Compare the original and current trainer for exactly 20 real replay updates.

Run serially, after the full default-option reproduction run has finished::

    uv run --no-sync python docs/briefs/03-artifacts/check_real_updates.py

Both trainers evaluate the complete fixed holdout normally, then stop immediately
before the 21st training forward. Their recipe retains six epochs so the original
learning-rate schedule is unchanged. This is a 20-update diagnostic, not a full
training run or a replacement for the historical per-epoch reproduction check.
Results and partial diagnostic outputs go under the ignored WORK directory.
"""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from fog.config import load_config_with_args
from fog.training import bc

BASE = "f0dc3d16eeecab8fbad545aed93af2b2de20a2c4"
CONFIG = "configs/replay_blueprint_cpu.yaml"
TRAIN = Path("runs/blueprint/split/train_1000.parquet")
HOLDOUT = Path("runs/blueprint/split/holdout_manifest.parquet")
WORK = Path("runs/blueprint/original-current-real20")
UPDATES = 20
EXTERNAL_SEED = 123
COMMAND = "uv run --no-sync python docs/briefs/03-artifacts/check_real_updates.py"


class _TwentyUpdatesComplete(Exception):
    """Stop before the next forward, after all 20 optimizer/schedule steps."""


def run(module, name: str) -> tuple[list[float], dict[str, torch.Tensor], dict]:
    overrides = [
        f"bc.manifest={TRAIN}",
        f"bc.holdout_manifest={HOLDOUT}",
        "bc.streaming=true",
        "bc.epochs=6",
        "bc.early_stopping_patience=0",
        f"log.dir={WORK / name}",
    ]
    config = load_config_with_args(CONFIG, overrides)
    for option in ("resume", "selection_metric", "torch_threads"):
        if option in config["bc"]:
            raise ValueError(f"diagnostic requires bc.{option} to be absent")

    losses: list[float] = []
    captured: dict[str, torch.Tensor] | None = None
    original_loss = module.bc_loss

    def record(agent, batch, loss_cfg, **kwargs):
        nonlocal captured
        if agent.training and len(losses) == UPDATES:
            # The trainer has completed backward, optimizer and schedule updates
            # for the preceding 20 calls. Do not perform a 21st model forward.
            captured = {
                key: tensor.detach().cpu().clone()
                for key, tensor in agent.state_dict().items()
            }
            raise _TwentyUpdatesComplete
        result = original_loss(agent, batch, loss_cfg, **kwargs)
        if agent.training:
            losses.append(float(result[0].detach()))
        return result

    # The trainer itself sets the Torch seed from the unchanged recipe.
    random.seed(EXTERNAL_SEED)
    np.random.seed(EXTERNAL_SEED)
    started = time.perf_counter()
    module.bc_loss = record
    try:
        try:
            module.train(config)
        except _TwentyUpdatesComplete:
            pass
        else:
            raise RuntimeError("trainer finished without reaching the 20-update stop")
    finally:
        module.bc_loss = original_loss

    if captured is None or len(losses) != UPDATES:
        raise RuntimeError("20-update diagnostic did not capture its completed state")
    return losses, captured, {
        "config": CONFIG,
        "overrides": overrides,
        "new_options_absent": ["bc.resume", "bc.selection_metric", "bc.torch_threads"],
        "torch_threads": torch.get_num_threads(),
        "wall_seconds_including_full_baseline_evaluation": time.perf_counter() - started,
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "original_bc.py"
        source.write_bytes(subprocess.check_output(["git", "show", f"{BASE}:fog/training/bc.py"]))
        spec = importlib.util.spec_from_file_location("original_bc", source)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        before, old_state, old_run = run(original, "original")
    after, new_state, new_run = run(bc, "current")

    differing_names = sorted(
        name for name in old_state.keys() | new_state.keys()
        if name not in old_state or name not in new_state
        or not torch.equal(old_state[name], new_state[name])
    )
    losses_identical = before == after
    tensors_identical = not differing_names
    identical = losses_identical and tensors_identical
    result = {
        "status": "passed_20_update_diagnostic" if identical else "failed_20_update_diagnostic",
        "full_training_run": False,
        "scope": "20 CPU updates on the real 1000-game training split, not a full reproduction",
        "baseline_evaluation": "complete fixed holdout evaluated normally in each trainer",
        "base_commit": BASE,
        "current_commit": current_commit,
        "command": COMMAND,
        "torch_version": str(torch.__version__),
        "external_python_numpy_seed": EXTERNAL_SEED,
        "completed_updates_per_trainer": UPDATES,
        "original_run": old_run,
        "current_run": new_run,
        "first_20_losses_before": before,
        "first_20_losses_after": after,
        "all_losses_identical": losses_identical,
        "maximum_loss_absolute_difference": max(abs(a - b) for a, b in zip(before, after)),
        "state_dict_tensor_count_before": len(old_state),
        "state_dict_tensor_count_after": len(new_state),
        "all_parameters_and_persistent_buffers_identical": tensors_identical,
        "differing_state_dict_tensor_names": differing_names,
    }
    output = WORK / "comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Diagnostic JSON: {output}", flush=True)
    if not identical:
        raise SystemExit("original/current 20-update diagnostic differs; see comparison.json")


if __name__ == "__main__":
    main()
