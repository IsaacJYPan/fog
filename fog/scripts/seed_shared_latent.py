"""Create a traceable shared-latent checkpoint while preserving compatible trained tensors."""
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
    parser.add_argument("--seed", type=int, default=20261068)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(args.seed)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = FogAgent.from_tables(load_config(args.config))
    manifest = agent.load_compatible_state_dict(source)
    state = agent.state_dict()
    loaded = manifest["loaded"]
    changed = [name for name in loaded if _digest(state[name]) != _digest(source[name])]
    if changed:
        raise RuntimeError(f"compatible source tensors changed: {changed[:10]}")
    initialized = manifest["initialized"]
    if not any(name.startswith("shared_latent.") for name in initialized):
        raise RuntimeError("shared latent was not initialized as a new module")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    report = {
        "status": "complete",
        "source": str(args.checkpoint),
        "output": str(args.out),
        "parameter_count": agent.n_params(),
        "loaded_tensor_count": len(loaded),
        "loaded_tensors_byte_identical": True,
        "initialized_tensor_count": len(initialized),
        "shape_mismatch_count": len(manifest["mismatched"]),
        "shared_latent_initialized": True,
        "policy_and_value_require_retraining": True,
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
