"""Combine broad hard-state simulation data with minority initial soft-value rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def build(hard_path: Path, initial_path: Path, output_path: Path) -> dict:
    hard = pl.read_parquet(hard_path)
    initial = pl.read_parquet(initial_path)
    if set(initial["split"].unique().to_list()) != {"train"}:
        raise ValueError("initial-soft manifest must contain only training rows")
    if not initial.select(
        (pl.col("soft_target_rows") == pl.col("n_rows")).all()
    ).item():
        raise ValueError("every initial-soft row must carry a soft target")

    hard = hard.with_columns(
        pl.lit("hard_exact").alias("value_target_kind")
    )
    initial = initial.with_columns(
        pl.concat_str([pl.lit("initial-soft-"), pl.col("id")]).alias("id"),
        pl.lit("initial_soft").alias("curriculum_source"),
        pl.lit("initial_soft").alias("value_target_kind"),
    )
    combined = pl.concat([hard, initial], how="diagonal_relaxed")
    missing = [path for path in combined["shard"].to_list() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced shards are missing")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    hard_rows = int(hard["n_rows"].sum())
    initial_rows = int(initial["n_rows"].sum())
    report = {
        "status": "complete",
        "hard_manifest": str(hard_path),
        "initial_soft_manifest": str(initial_path),
        "output_manifest": str(output_path),
        "hard_shards": hard.height,
        "hard_rows": hard_rows,
        "initial_soft_shards": initial.height,
        "initial_soft_rows": initial_rows,
        "total_shards": combined.height,
        "total_rows": hard_rows + initial_rows,
        "initial_soft_row_fraction": initial_rows / (hard_rows + initial_rows),
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard", required=True, type=Path)
    parser.add_argument("--initial-soft", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.hard, args.initial_soft, args.out), indent=2))


if __name__ == "__main__":
    main()
