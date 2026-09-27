"""Freeze a chronological replay split before building imputation donors."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from fog.scripts.parse_replays import replay_files
from fog.training.replay_candidates import replay_digest


def donor_exclusions(raw: Path, cutoff: int) -> list[str]:
    excluded = []
    for path in replay_files(raw, 0):
        replay = json.loads(path.read_text(encoding="utf-8"))
        if int(replay.get("uploadtime") or 0) >= cutoff:
            excluded.append(path.stem)
    return sorted(excluded)


def prepare(raw: Path, out: Path, *, games: int = 2000, min_rating: int = 1650,
            holdout_fraction: float = 0.1) -> dict:
    if out.exists():
        raise FileExistsError(out)
    if games < 2 or not 0 < holdout_fraction < 1:
        raise ValueError("need at least two games and a holdout fraction between zero and one")
    unique = {}
    for path in replay_files(raw, 0):
        replay = json.loads(path.read_text(encoding="utf-8"))
        if (replay.get("formatid") != "gen9ou" or int(replay.get("rating") or 0) < min_rating
                or int(replay.get("uploadtime") or 0) <= 0):
            continue
        digest = replay_digest(replay)
        unique.setdefault(digest, {"id": path.stem, "path": path.as_posix(),
                                  "uploadtime": int(replay["uploadtime"]), "log_sha256": digest,
                                  "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    ordered = sorted(unique.values(), key=lambda row: (row["uploadtime"], row["id"]))
    if len(ordered) < games:
        raise ValueError(f"requested {games} games, only {len(ordered)} eligible unique games")
    selected = ordered[-games:]
    n_holdout = max(1, min(games - 1, round(games * holdout_fraction)))
    cutoff = selected[-n_holdout]["uploadtime"]
    # Keep equal timestamps on the same side of the boundary.
    for row in selected:
        row["split"] = "holdout" if row["uploadtime"] >= cutoff else "train"
    if not any(row["split"] == "train" for row in selected):
        raise ValueError("timestamps cannot produce a chronological training split")
    out.mkdir(parents=True)
    (out / "raw").mkdir()
    for row in selected:
        shutil.copyfile(row["path"], out / "raw" / (row["id"] + ".json"))
    report = {"selection": "latest unique public logs, chronological holdout",
              "min_rating": min_rating, "eligible_unique_games": len(ordered),
              "holdout_cutoff": cutoff, "private_truth": False, "games": selected}
    (out / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "holdout_ids.json").write_text(json.dumps(
        [row["id"] for row in selected if row["split"] == "holdout"], indent=2), encoding="utf-8")
    (out / "donor_excluded_ids.json").write_text(
        json.dumps(donor_exclusions(raw, cutoff), indent=2), encoding="utf-8")
    return report


def finalize(out: Path) -> dict:
    import numpy as np
    import polars as pl

    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    held = {row["id"] for row in selection["games"] if row["split"] == "holdout"}
    hashes = {row["log_sha256"] for row in selection["games"] if row["split"] == "holdout"}
    with np.load(out / "bank.npz", allow_pickle=False) as bank:
        if held & set(bank["replay_ids"].tolist()) or hashes & set(bank["log_hashes"].tolist()):
            raise ValueError("held-out evidence leaked into candidate bank")
        if (bank["timestamps"] >= selection["holdout_cutoff"]).any():
            raise ValueError("donor bank crosses chronological holdout cutoff")
    manifest = pl.read_parquet(out / "manifest.parquet")
    selected_ids = {row["id"] for row in selection["games"]}
    if set(manifest["id"].to_list()) - selected_ids:
        raise ValueError("conversion contains unselected games")
    result = {}
    for split in ("train", "holdout"):
        path = out / f"{split}_manifest.parquet"
        if path.exists():
            raise FileExistsError(path)
        mask = pl.col("id").is_in(sorted(held))
        frame = manifest.filter(mask if split == "holdout" else ~mask)
        if not len(frame):
            raise ValueError(f"empty {split} partition after conversion")
        frame.write_parquet(path)
        result[split] = {"games": len(frame), "decision_versions": int(frame["n_rows"].sum())}
    result["holdout_donor_overlap"] = 0
    result["private_truth"] = False
    (out / "split_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "finalize"))
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--min-rating", type=int, default=1650)
    args = ap.parse_args()
    if args.mode == "prepare":
        if not args.raw:
            ap.error("prepare requires --raw")
        result = prepare(args.raw, args.out, games=args.games, min_rating=args.min_rating)
        print(json.dumps({k: v for k, v in result.items() if k != "games"}, indent=2))
    else:
        print(json.dumps(finalize(args.out), indent=2))


if __name__ == "__main__":
    main()
