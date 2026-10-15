"""Freeze a portable verified subset for curriculum feasibility measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from fog.training.teacher_replay import digest


def prepare(manifest, train, development, out, *, train_games=0, development_games=8):
    source = json.loads(manifest.read_text())
    if source.get("format") != "fog.teacher-replay-data" or not source.get("complete"):
        raise ValueError("Require the complete verified teacher-replay manifest")
    if out.exists():
        raise FileExistsError("Preserve existing curriculum banks")
    if any(type(n) is not int or n < 0 for n in (train_games, development_games)) or not train_games + development_games:
        raise ValueError("Choose a nonempty bounded subset")
    out.mkdir(parents=True)
    records = []
    groups = {split: {key for row in source["records"] if row["split"] == split for key in row["compositions"]}
              for split in ("train", "development")}
    if groups["train"] & groups["development"]:
        raise ValueError("Source training and development compositions overlap")
    for split, directory, count in (("train", train, train_games), ("development", development, development_games)):
        selected = [row for row in source["records"] if row["split"] == split][:count]
        if len(selected) != count:
            raise ValueError("Requested more verified games than the source contains")
        for row in selected:
            if not all(row.get(k) is True for k in ("exact_public_trace", "native_legality_exact", "opponent_truth_is_label_only", "teacher_answer_features_zeroed")):
                raise ValueError("A selected source lacks exact-replay validation")
            files = {"source": (directory / row["source"], "sources/" + split + "-" + row["source"]),
                     "shard": (manifest.parent / row["shard"], row["shard"])}
            saved = dict(row)
            for key, (path, target) in files.items():
                if not path.resolve().is_relative_to(directory.resolve() if key == "source" else manifest.parent.resolve()):
                    raise ValueError("Source path escapes its declared input folder")
                if digest(path) != row[key + "_sha256"]:
                    raise ValueError("Verified source changed before curriculum packaging")
                destination = out / target
                if not destination.resolve().is_relative_to(out.resolve()):
                    raise ValueError("Portable bank path escapes its output folder")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                saved[key] = target
            records.append(saved)
    result = {"format": "fog.backplay-probe-bank", "version": 1, "source_replays_verified": True,
              "source_manifest_sha256": digest(manifest), "composition_disjoint": True,
              "records": records, "training_uses_development": False, "final_cohort_used": False}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "train", "development", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train-games", type=int, default=0)
    parser.add_argument("--development-games", type=int, default=8)
    args = parser.parse_args()
    prepare(args.manifest, args.train, args.development, args.out,
            train_games=args.train_games, development_games=args.development_games)


if __name__ == "__main__":
    main()
