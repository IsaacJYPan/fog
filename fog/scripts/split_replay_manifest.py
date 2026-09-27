"""Split converted replays by game so decisions from one battle stay together."""
from __future__ import annotations

import argparse
from pathlib import Path

from fog.training.bc import _split_replay_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    for name in ("train_manifest.parquet", "holdout_manifest.parquet"):
        if (args.out / name).exists():
            raise FileExistsError(f"refusing to replace an existing split: {args.out / name}")
    paths = _split_replay_manifest(args.manifest, args.out, args.holdout_fraction, args.seed)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
