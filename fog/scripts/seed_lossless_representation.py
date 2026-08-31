"""Migrate the improved encoders into the lossless structured-memory architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.models.agent import FogAgent


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261057)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(args.seed)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = FogAgent.from_tables(load_config(args.config))
    target = agent.state_dict()
    replaced = []
    loaded = []
    for name, value in source.items():
        if name.startswith("policy_value.conditional_action_scorer."):
            replaced.append(name)
            continue
        if name not in target or tuple(target[name].shape) != tuple(value.shape):
            raise RuntimeError(f"unexpected incompatible tensor {name}: {tuple(value.shape)}")
        target[name] = value
        loaded.append(name)
    agent.load_state_dict(target)
    state = agent.state_dict()
    changed_shared = [name for name in loaded if _digest(state[name]) != _digest(source[name])]
    if changed_shared:
        raise RuntimeError(f"shared tensors changed: {changed_shared[:5]}")
    # The new action-conditioned route is exactly zero at its output, so migration cannot perturb
    # policy logits before it is deliberately trained.
    final = agent.policy_value.conditional_action_scorer[-1]
    if bool(final.weight.count_nonzero()) or bool(final.bias.count_nonzero()):
        raise RuntimeError("new conditional action output is not zero initialized")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    report = {
        "status": "complete", "source": str(args.checkpoint), "output": str(args.out),
        "shared_tensor_count": len(loaded), "shared_tensors_byte_identical": True,
        "reinitialized_action_tensors": sorted(replaced),
        "new_action_output_zero": True,
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
