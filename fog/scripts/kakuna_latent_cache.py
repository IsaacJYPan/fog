"""Extract frozen lower-layer features using only player-visible native inputs."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-games", type=int)
    args = parser.parse_args()
    if args.maximum_games is not None and args.maximum_games < 1:
        parser.error("Local trial needs at least one game")
    from fog.pretrained.latent_cache import build

    report = build(json.loads(args.config.read_text()), args.out, maximum_games=args.maximum_games)
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "recipe")}), flush=True)


if __name__ == "__main__":
    main()
