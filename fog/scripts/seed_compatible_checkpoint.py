"""Migrate a checkpoint into the current config by loading all shape-compatible tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.config import load_config_with_args
from fog.models.agent import FogAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    agent = FogAgent.from_tables(load_config_with_args(args.config))
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    migration = agent.load_compatible_state_dict(source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), args.out)
    report = {
        "status": "complete",
        "source": str(args.checkpoint),
        "checkpoint": str(args.out),
        "seed": args.seed,
        "migration": migration,
    }
    report_path = args.out.with_suffix(args.out.suffix + ".migration.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
