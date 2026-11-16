"""Build brief 13 splits, excluding acting validation compositions row by row.

Run from the repository root with PYTHONPATH=. and the locked environment.
The converted corpus is immutable; filtered shards and manifests go to a new directory.
Completed games are reusable after interruption. Keep that directory to resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from fog.constants.tables import TABLES
from fog.scripts.split_replay_manifest import _fixed_holdout_split, _nested_training_sets


def filter_arrays(arrays, forbidden):
    """Use submitted own species, including fainted members, rather than battle forms."""
    own = np.sort(arrays["self_belief_targets"][:, :, 0], axis=1)
    keep = np.ones(len(own), dtype=bool)
    for composition in forbidden:
        keep &= ~np.all(own == composition, axis=1)
    result = {}
    preview_keep = np.array([
        tuple(sorted(TABLES.species_index(str(s)) for s in team)) not in forbidden
        for team in arrays["preview_species"]
    ])
    for key, value in arrays.items():
        selector = np.ones(2, dtype=bool) if key.startswith("preview_") else keep
        if len(value) != len(selector):
            raise ValueError(f"unexpected array row count: {key}")
        result[key] = value[selector]
    # The existing preview loader requires two slots/game and selects preview_valid.
    result["preview_valid"] &= preview_keep
    return result, int((~keep).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-ids", type=Path,
                        default=Path("data/replay_splits/p2_holdout_2000_ids.txt"))
    parser.add_argument("--validation", type=Path,
                        default=Path("data/teams/replay_v1/validation/manifest.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[20000, 100000])
    args = parser.parse_args()
    teams = json.loads(args.validation.read_text())["teams"]
    forbidden = {tuple(sorted(TABLES.species_index(s) for s in t["composition"]))
                 for t in teams}
    settings = {"manifest": str(args.manifest.resolve()),
                "holdout_ids": args.holdout_ids.read_text(),
                "validation": json.loads(args.validation.read_text()), "sizes": args.sizes}
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path = args.out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("split settings changed; use a fresh directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    training, holdout, newer = _fixed_holdout_split(
        pl.read_parquet(args.manifest), args.holdout_ids.read_text().split())
    assert len(holdout) == 2000
    shards = args.out / "shards"
    shards.mkdir(exist_ok=True)
    counts = {"input_training_games": len(training),
              "input_training_rows": int(training["n_rows"].sum()),
              "newer_games_excluded": newer, "validation_compositions": len(forbidden),
              "games_affected": 0, "games_removed": 0, "rows_removed": 0}
    retained = []
    for row in training.iter_rows(named=True):
        target = shards / f"{row['id']}.npz"
        receipt = shards / f"{row['id']}.json"
        if receipt.exists():
            entry = json.loads(receipt.read_text())
            if entry["row"] is not None and not Path(entry["row"]["shard"]).exists():
                raise FileNotFoundError(entry["row"]["shard"])
        else:
            with np.load(row["shard"], allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            assert len(arrays["action"]) == row["n_rows"]
            if not all(key in arrays for key in ("terminal_alive", "terminal_valid")):
                raise ValueError("dense outcome targets missing; reconvert with merged converter")
            filtered, removed = filter_arrays(arrays, forbidden)
            entry = {"removed": removed, "row": None}
            if len(filtered["action"]):
                kept = dict(row)
                if removed or not np.array_equal(filtered["preview_valid"], arrays["preview_valid"]):
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **filtered)
                    temporary.replace(target)
                    kept["shard"] = str(target.resolve())
                else:
                    kept["shard"] = str(Path(row["shard"]).resolve())
                kept["n_rows"] = len(filtered["action"])
                kept["n_preview_rows"] = len(filtered["preview_valid"])
                kept["n_preview_valid"] = int(filtered["preview_valid"].sum())
                entry["row"] = kept
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(entry))
            temporary.replace(receipt)
        counts["games_affected"] += int(entry["removed"] > 0)
        counts["rows_removed"] += entry["removed"]
        counts["games_removed"] += int(entry["row"] is None)
        if entry["row"] is not None:
            retained.append(entry["row"])
    clean = pl.DataFrame(retained, schema=training.schema)
    holdout = holdout.with_columns(pl.col("shard").map_elements(
        lambda p: str(Path(p).resolve()), return_dtype=pl.String))
    splits = {"train_manifest": clean, "holdout_manifest": holdout}
    sizes = [size for size in args.sizes if size <= len(clean)]
    splits.update({f"train_{size}": frame for size, frame in
                   _nested_training_sets(clean, sizes, 20260804).items()})
    counts["splits"] = {}
    for name, frame in splits.items():
        temporary = args.out / f"{name}.parquet.tmp"
        frame.write_parquet(temporary)
        temporary.replace(args.out / f"{name}.parquet")
        counts["splits"][name] = {"games": len(frame), "rows": int(frame["n_rows"].sum())}
    counts["skipped_oversized_subsets"] = sorted(set(args.sizes) - set(sizes))
    (args.out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
