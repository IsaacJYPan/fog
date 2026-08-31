"""Create a separate behavior-preserving checkpoint with the token-aware action path enabled."""
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
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--new-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261052)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(args.seed)
    base = FogAgent.from_tables(load_config(args.base_config))
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(source)
    seeded = FogAgent.from_tables(load_config(args.new_config))
    result = seeded.load_state_dict(source, strict=False)
    expected = {
        name for name in seeded.state_dict()
        if name.startswith("policy_value.conditional_action_scorer.")
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(
            f"unexpected migration difference missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )
    seeded_state = seeded.state_dict()
    changed_existing = [
        name for name, value in source.items()
        if name not in expected and _digest(value) != _digest(seeded_state[name])
    ]
    if changed_existing:
        raise RuntimeError(f"existing tensors changed during seed: {changed_existing[:5]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(seeded_state, args.out)
    report = {
        "status": "complete",
        "source": str(args.checkpoint),
        "output": str(args.out),
        "existing_tensors_byte_identical": True,
        "new_tensors": sorted(expected),
        "new_output_layer_zero_initialized": True,
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
