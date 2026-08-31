"""Read-only aggregate belief/value evaluation on one replay-shard manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.config import load_config_with_args
from fog.models.agent import FogAgent
from fog.training.bc import _fixed_eval
from fog.training.dataset import StreamingReplayDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-buffer", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261218)
    args = parser.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)
    )
    dataset = StreamingReplayDataset(args.manifest, shard_buffer=args.shard_buffer)
    metrics = _fixed_eval(
        agent,
        dataset,
        args.batch_size,
        cfg.get("loss", {}),
        device,
        args.seed,
        args.max_batches,
    )
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "rows": len(dataset),
        "metrics": metrics,
        "read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
