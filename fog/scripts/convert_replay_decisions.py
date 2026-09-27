"""Convert high-rated public replays into uncertainty-audited policy decisions."""
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np

from fog.scripts.parse_replays import (
    parse_replay_policy_versions,
    replay_files,
    rows_to_arrays,
)

_BANK = None


def initialize_worker(bank_path: Path | None) -> None:
    from fog.training.replay_candidates import ReplayCandidateBank

    global _BANK
    _BANK = ReplayCandidateBank(bank_path) if bank_path else None


def convert_one(path: Path, *, shard_dir: Path, min_rating: int, imputations: int,
                seed: int) -> dict:
    output = {"filters": {}, "imputation_report": {}, "source_rows": 0}
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
        rating = int(replay.get("rating") or 0)
        if rating < min_rating:
            output["filters"] = {"below_rating": 1}
            return output
        result = parse_replay_policy_versions(replay, replay_id=path.stem,
                                              imputations=imputations, seed=seed,
                                              candidate_bank=_BANK)
        output["imputation_report"] = result.get("imputation_report", {})
        output["filters"] = result.get("filter_counts", {})
        output["source_rows"] = sum(output["filters"].values())
        rows = result["rows"]
        if not rows:
            output["quarantine"] = {"id": path.stem, "reason": "no accepted policy decisions"}
            return output
        arrays = rows_to_arrays(rows)
        if not np.isfinite(arrays["action_features"]).all():
            raise ValueError("non-finite reconstructed action features")
        if not arrays["policy_valid"].all():
            raise ValueError("filtered policy shard retained an invalid decision")
        shard = shard_dir / f"{path.stem}.npz"
        np.savez_compressed(shard, **arrays)
        output["manifest"] = {
            "id": path.stem, "n_rows": len(rows), "shard": shard.as_posix(),
            "rating": rating, "p1_win": result["p1_win"], "imputations": imputations,
            "uploadtime": int(replay.get("uploadtime") or 0),
            "source_provider": replay.get("_fog_source", {}).get("provider", "showdown_replay"),
            "source_revision": replay.get("_fog_source", {}).get("revision", ""),
            "source_shard_sha256": replay.get("_fog_source", {}).get("shard_sha256", ""),
            "private_truth": False,
            "imputation_method": "replay_candidates_v1" if _BANK else "usage_baseline",
            "candidate_bank_sha256": _BANK.sha256 if _BANK else "",
        }
    except Exception as error:  # noqa: BLE001 - quarantine preserves the corpus
        output["quarantine"] = {"id": path.stem,
                                "reason": f"{type(error).__name__}: {error}",
                                "trace": traceback.format_exc()[-1200:]}
    return output


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw_replays"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--imputations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidate-bank", type=Path,
                        help="training-only replay donor bank; omit for the usage baseline")
    args = parser.parse_args()
    if args.workers < 1 or args.imputations < 1:
        parser.error("workers and imputations must be positive")
    initialize_worker(args.candidate_bank)
    bank = _BANK

    if (args.out / "manifest.parquet").exists():
        raise FileExistsError(f"refusing to overwrite existing conversion: {args.out}")
    shard_dir = args.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    quarantine: list[dict] = []
    filter_counts: Counter[str] = Counter()
    total_source_rows = 0
    imputation_report: Counter = Counter()
    paths = replay_files(args.raw, args.limit)
    worker = partial(convert_one, shard_dir=shard_dir, min_rating=args.min_rating,
                     imputations=args.imputations, seed=args.seed)
    context = (ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                   initargs=(args.candidate_bank,))
               if args.workers > 1 else nullcontext())
    with context as pool:
        results = pool.map(worker, paths) if pool else map(worker, paths)
        for index, result in enumerate(results, 1):
            filter_counts.update(result["filters"])
            imputation_report.update(result["imputation_report"])
            total_source_rows += result["source_rows"]
            if "manifest" in result:
                manifest.append(result["manifest"])
            if "quarantine" in result:
                quarantine.append(result["quarantine"])
            if index % 25 == 0 or index == len(paths):
                print(f"Replay conversion: {index}/{len(paths)}, {len(manifest)} accepted, "
                      f"{len(quarantine)} quarantined", flush=True)

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
        "private_truth": False,
        "source_providers": sorted({row["source_provider"] for row in manifest}),
        "filter_counts": dict(sorted(filter_counts.items())),
        "imputation_method": "replay_candidates_v1" if bank else "usage_baseline",
        "candidate_bank_sha256": bank.sha256 if bank else None,
        "imputation_report": dict(imputation_report),
    }
    (args.out / "quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, indent=2), flush=True)


if __name__ == "__main__":
    main()
