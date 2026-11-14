"""Verify the recreated brief 01 corpus before any comparison training."""
from datetime import UTC, datetime
from pathlib import Path
import json

import polars as pl

root = Path("runs/blueprint/split")
expected = {40: 2020, 1000: 52983, 3000: 156597, 10000: 514103, 20000: 1023525}
result = {}
for games, rows in expected.items():
    frame = pl.read_parquet(root / f"train_{games}.parquet")
    result[f"train_{games}"] = {"games": len(frame), "rows": frame["n_rows"].sum()}
    assert len(frame) == games and frame["n_rows"].sum() == rows, result
h = pl.read_parquet(root / "holdout_manifest.parquet")
ids = set(Path("data/replay_splits/p2_holdout_2000_ids.txt").read_text().split())
result["holdout"] = {
    "games": len(h), "rows": h["n_rows"].sum(),
    "fixed_ids_match": set(h["id"].cast(pl.String).to_list()) == ids,
    "oldest_upload_utc": datetime.fromtimestamp(h["uploadtime"].min(), UTC).isoformat(),
    "newest_upload_utc": datetime.fromtimestamp(h["uploadtime"].max(), UTC).isoformat(),
}
print(json.dumps(result, indent=2), flush=True)
assert len(h) == 2000 and h["n_rows"].sum() == 104037 and result["holdout"]["fixed_ids_match"]
assert result["holdout"]["oldest_upload_utc"].startswith("2026-02-20")
assert result["holdout"]["newest_upload_utc"].startswith("2026-05-19")
fixed = Path("runs/blueprint/split_fixed")
if fixed.exists():
    for name in ("holdout_manifest", "train_20000"):
        a = pl.read_parquet(root / f"{name}.parquet")
        b = pl.read_parquet(fixed / f"{name}.parquet")
        assert set(a["id"]) == set(b["id"]), name
    print("fixed and chronological holdout/train_20000 IDs match")
