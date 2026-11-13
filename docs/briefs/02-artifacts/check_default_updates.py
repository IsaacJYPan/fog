"""Compare the old and new trainers on the generated fixture, never on real replays.

    uv run --no-sync python docs/briefs/02-artifacts/check_default_updates.py

Requires the fixture runner's converted shards/split. Leaves both diagnostic
checkpoints in ignored runs/blueprint_default_equivalence.
"""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from fog.config import load_config_with_args
from fog.training import bc

BASE = "f0dc3d16eeecab8fbad545aed93af2b2de20a2c4"
FIXTURE = Path("runs/blueprint_runner_fixture_smoke")
WORK = Path("runs/blueprint_default_equivalence")


def run(module, name):
    config = load_config_with_args("configs/replay_blueprint_cpu.yaml", [
        f"bc.manifest={FIXTURE / 'split/train_manifest.parquet'}",
        f"bc.holdout_manifest={FIXTURE / 'split/holdout_manifest.parquet'}",
        "bc.streaming=true", "bc.epochs=3", "bc.early_stopping_patience=0",
        f"log.dir={WORK / name}",
    ])
    values = []
    original = module.bc_loss

    def record(agent, batch, loss_cfg, **kwargs):
        result = original(agent, batch, loss_cfg, **kwargs)
        if agent.training:
            values.append(float(result[0].detach()))
        return result

    module.bc_loss = record
    random.seed(123)
    np.random.seed(123)
    try:
        module.train(config)
    finally:
        module.bc_loss = original
    state = torch.load(WORK / name / "checkpoints/bc_final.pt", weights_only=True)
    return values, state


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "original_bc.py"
        source.write_bytes(subprocess.check_output(["git", "show", f"{BASE}:fog/training/bc.py"]))
        spec = importlib.util.spec_from_file_location("original_bc", source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        before, old_state = run(old, "before")
    after, new_state = run(bc, "after")
    assert len(before) >= 20 and before == after
    assert old_state.keys() == new_state.keys()
    assert all(torch.equal(value, new_state[key]) for key, value in old_state.items())
    result = {
        "fixture": "600 generated two-turn protocol logs, not the pinned corpus",
        "base_commit": BASE,
        "current_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": "uv run --no-sync python docs/briefs/02-artifacts/check_default_updates.py",
        "torch_threads": torch.get_num_threads(),
        "steps": len(before), "first_20_losses_before": before[:20],
        "first_20_losses_after": after[:20],
        "all_losses_identical": before == after,
        "all_final_state_tensors_identical": True,
    }
    (WORK / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
