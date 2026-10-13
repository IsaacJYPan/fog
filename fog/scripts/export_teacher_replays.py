"""Export exact-team teacher shards from trace-verified Showdown reconstructions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from fog.training.teacher_replay import digest, read_record, replay_game


def export(train, development, pools, out, *, train_games=None, development_games=None):
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.pretrained.kakuna import KAKUNA_34_SHA256
    from fog.team_identity import team_group_key

    if out.exists():
        raise FileExistsError("Use a new teacher-data identity")
    inputs, compositions, records = {}, {}, []
    for split, source, limit in (("train", train, train_games), ("development", development, development_games)):
        collection = json.loads((source / "collection.json").read_text(encoding="utf-8"))
        if collection["split"] != split or collection["pool_sha256"] != digest(pools / (split + ".json")):
            raise ValueError("Teacher source cohort has a different pool or split")
        if collection["native_sha256"] != KAKUNA_34_SHA256:
            raise ValueError("Teacher source is not the frozen Kakuna34 collection")
        teams = json.loads((pools / (split + ".json")).read_text(encoding="utf-8"))["teams"]
        compositions[split] = {team_group_key(paste) for paste in teams}
        paths = sorted(source.glob("*.json.gz"))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise ValueError("An exact teacher cohort is empty")
        inputs[split] = {
            "collection_sha256": digest(source / "collection.json"),
            "pool_sha256": collection["pool_sha256"],
            "records": {path.name: digest(path) for path in paths},
        }
        records.extend((split, path) for path in paths)
    if compositions["train"] & compositions["development"]:
        raise ValueError("Teacher train/development compositions overlap")
    out.mkdir(parents=True)
    (out / "shards").mkdir()
    recipe = {"created_at": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
              "format": "gen9ou", "replay_verification": "exact_public_trace_except_clock_timestamps",
              "source_corrections": "exclude_recorded_request_rejection_games",
              "teacher_answers_in_features": False, "opponent_truth_is_label_only": True}
    (out / "INPUTS.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    manifest, exclusions = [], []
    started = time.monotonic()
    with ShowdownDirectClient() as client:
        for index, (split, path) in enumerate(records):
            source_hash = digest(path)
            if source_hash != inputs[split]["records"][path.name]:
                raise ValueError("Source changed during teacher export")
            record = read_record(path)
            keys = [team_group_key(paste) for paste in record["submitted_teams"].values()]
            if any(key not in compositions[split] for key in keys):
                raise ValueError("Submitted composition lies outside its frozen pool")
            if record.get("choice_rejections"):
                exclusions.append({"split": split, "source": path.name, "sha256": source_hash,
                                   "reason": "recorded_request_corrections", "corrections": record["choice_rejections"]})
                continue
            try:
                arrays, report = replay_game(client, record, tag=f"teacher-{split}-{index}")
            except Exception as error:
                failure = {"split": split, "source": path.name, "sha256": source_hash,
                           "error": type(error).__name__ + ": " + str(error),
                           "status": "failed", "completed_games": len(manifest)}
                (out / "FAILURE.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                raise
            name = f"{split}-{path.stem.removesuffix('.json')}.npz"
            shard = out / "shards" / name
            np.savez_compressed(shard, **arrays)
            manifest.append({"id": f"{split}-{path.name}", "split": split, "source": path.name,
                             "source_sha256": source_hash, "shard": "shards/" + name,
                             "shard_sha256": digest(shard), "compositions": keys, **report})
            if index < 3 or (index + 1) % 50 == 0:
                print(json.dumps({"exported": len(manifest), "sources_checked": index + 1,
                                  "total": len(records), "rows": sum(row["rows"] for row in manifest),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    counts = Counter()
    for row in manifest:
        counts[row["split"] + "_games"] += 1
        counts[row["split"] + "_rows"] += row["rows"]
        counts[row["split"] + "_policy_rows"] += row["policy_rows"]
    if not counts["train_games"] or not counts["development_games"]:
        raise ValueError("Exclusions emptied a teacher-data split")
    result = {"format": "fog.teacher-replay-data", "version": 1, "complete": True,
              "recipe": recipe, "records": manifest, "exclusions": exclusions,
              "counts": dict(counts), "elapsed_seconds": time.monotonic() - started,
              "training_performed": False, "gpu_requested": False}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "counts": dict(counts), "exclusions": len(exclusions),
                      "elapsed_seconds": result["elapsed_seconds"], "manifest": str(out / "manifest.json")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-games", type=int)
    parser.add_argument("--development-games", type=int)
    args = parser.parse_args()
    for count in (args.train_games, args.development_games):
        if count is not None and count < 1:
            raise ValueError("Teacher cohort limits must be positive")
    export(args.train, args.development, args.pools, args.out,
           train_games=args.train_games, development_games=args.development_games)


if __name__ == "__main__":
    main()
