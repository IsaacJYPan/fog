"""Build public joint-set donors from a training-only raw replay directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fog.scripts.parse_replays import replay_files
from fog.training.replay_candidates import build_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exclude-ids", type=Path, help="JSON list of held-out replay IDs")
    args = parser.parse_args()
    excluded = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()
    print(json.dumps(build_bank(replay_files(args.raw, args.limit), args.out,
                               excluded_ids=excluded), indent=2))


if __name__ == "__main__":
    main()
