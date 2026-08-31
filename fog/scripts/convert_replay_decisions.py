"""Convert high-rated public replays into uncertainty-audited policy decisions."""
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from fog.scripts.parse_replays import (
    parse_replay_policy_versions,
    replay_files,
    rows_to_arrays,
)


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw_replays"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--imputations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    if (args.out / "manifest.parquet").exists():
        raise FileExistsError(f"refusing to overwrite existing conversion: {args.out}")
    shard_dir = args.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    quarantine: list[dict] = []
    filter_counts: Counter[str] = Counter()
    total_source_rows = 0
    for path in replay_files(args.raw, args.limit):
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            rating = int(replay.get("rating") or 0)
            if rating < args.min_rating:
                filter_counts["below_rating"] += 1
                continue
            result = parse_replay_policy_versions(
                replay,
                replay_id=path.stem,
                imputations=args.imputations,
                seed=args.seed,
            )
            filter_counts.update(result.get("filter_counts", {}))
            rows = result["rows"]
            total_source_rows += sum(result.get("filter_counts", {}).values())
            if not rows:
                quarantine.append({"id": path.stem, "reason": "no accepted policy decisions"})
                continue
            arrays = rows_to_arrays(rows)
            if not np.isfinite(arrays["action_features"]).all():
                raise ValueError("non-finite reconstructed action features")
            if not arrays["policy_valid"].all():
                raise ValueError("filtered policy shard retained an invalid decision")
            shard = shard_dir / f"{path.stem}.npz"
            np.savez_compressed(shard, **arrays)
            manifest.append({
                "id": path.stem,
                "n_rows": len(rows),
                "shard": str(shard),
                "rating": rating,
                "p1_win": result["p1_win"],
                "imputations": args.imputations,
            })
        except Exception as error:  # noqa: BLE001 - quarantine preserves the corpus
            quarantine.append({
                "id": path.stem,
                "reason": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc()[-1200:],
            })

    if manifest:
        pl.DataFrame(manifest).write_parquet(args.out / "manifest.parquet")
    (args.out / "quarantine.json").write_text(
        json.dumps(quarantine, indent=2), encoding="utf-8"
    )
    accepted = int(filter_counts.get("accepted", 0))
    quality = {
        "status": "complete",
        "source_replays_considered": len(replay_files(args.raw, args.limit)),
        "parsed_replays": len(manifest),
        "quarantined_replays": len(quarantine),
        "accepted_decision_versions": accepted,
        "all_decision_versions": total_source_rows,
        "accepted_fraction": accepted / max(1, total_source_rows),
        "imputations_per_replay": args.imputations,
        "min_rating": args.min_rating,
        "provenance_codes": {"unknown": 0, "inferred": 1, "exact": 2},
        "public_state_reconstruction": "chronological_prefix_only",
        "future_opponent_fields_used": False,
        "filter_counts": dict(sorted(filter_counts.items())),
    }
    (args.out / "quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, indent=2), flush=True)


if __name__ == "__main__":
    main()
