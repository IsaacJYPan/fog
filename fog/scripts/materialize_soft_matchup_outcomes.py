"""Create non-destructive replay shards whose value target is an empirical matchup mean.

Repeated stochastic games from the same ordered team matchup share a Bayes-style soft outcome
target. Belief labels and every input tensor are copied byte-for-byte. This reduces avoidable value
gradient noise without leaking an evaluated game's future trajectory into its inputs; splits must
remain disjoint and are preserved from the source manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--target-rows", choices=("all", "initial"), default="all",
        help="Replace every outcome or only rows with history_len == 0.",
    )
    parser.add_argument(
        "--output-rows", choices=("all", "targeted"), default="all",
        help="Keep every row or retain only rows selected by --target-rows.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    shard_dir = args.out / "shards"
    shard_dir.mkdir()
    manifest = pl.read_parquet(args.manifest)
    required = {"shard", "p1_team", "p2_team", "outcome", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    stats = manifest.group_by(["p1_team", "p2_team"]).agg(
        pl.len().alias("repeat_count"), pl.col("outcome").mean().alias("soft_outcome")
    )
    if int(stats["repeat_count"].min()) < 2:
        raise ValueError("every matchup must have at least two repetitions")
    joined = manifest.join(stats, on=["p1_team", "p2_team"], how="left")
    rows = []
    for index, row in enumerate(joined.iter_rows(named=True)):
        source = Path(row["shard"])
        with np.load(source) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        row_count = int(arrays["outcome"].shape[0])
        if args.target_rows == "all":
            target_mask = np.ones(row_count, dtype=bool)
        else:
            if "history_len" not in arrays:
                raise ValueError(f"{source} lacks history_len required for initial targets")
            target_mask = np.asarray(arrays["history_len"]) == 0
            if int(target_mask.sum()) <= 0:
                raise ValueError(f"{source} has no initial row")
        outcomes = np.array(arrays["outcome"], copy=True)
        outcomes[target_mask] = float(row["soft_outcome"])
        arrays["outcome"] = outcomes
        if args.output_rows == "targeted":
            arrays = {
                name: value[target_mask]
                if value.ndim > 0 and int(value.shape[0]) == row_count
                else value
                for name, value in arrays.items()
            }
        target = shard_dir / f"game_{index:06d}.npz"
        np.savez_compressed(target, **arrays)
        output = {key: value for key, value in row.items() if key != "shard"}
        output["shard"] = str(target.resolve())
        output["hard_outcome"] = float(row["outcome"])
        output["soft_outcome"] = float(row["soft_outcome"])
        output["soft_target_rows"] = int(target_mask.sum())
        output["n_rows"] = int(arrays["outcome"].shape[0])
        if args.target_rows == "all" or args.output_rows == "targeted":
            output["outcome"] = float(row["soft_outcome"])
        rows.append(output)
    result = pl.DataFrame(rows)
    result.write_parquet(args.out / "manifest.parquet")
    for split in result["split"].unique().to_list():
        result.filter(pl.col("split") == split).write_parquet(
            args.out / f"{split}_manifest.parquet"
        )
    report = {
        "status": "complete",
        "source_manifest": str(args.manifest),
        "games": len(rows),
        "matchups": stats.height,
        "minimum_repetitions": int(stats["repeat_count"].min()),
        "maximum_repetitions": int(stats["repeat_count"].max()),
        "soft_outcome_min": float(stats["soft_outcome"].min()),
        "soft_outcome_max": float(stats["soft_outcome"].max()),
        "target_rows": args.target_rows,
        "output_rows": args.output_rows,
        "soft_target_rows": int(result["soft_target_rows"].sum()),
        "inputs_and_belief_truth_unchanged": True,
        "future_information_added_to_inputs": False,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
