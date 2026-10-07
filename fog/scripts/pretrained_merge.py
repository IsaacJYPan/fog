"""Merge hash-verified private collection shards and audit both native perspectives."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from fog.pretrained.kakuna import load_kakuna, sha256
from fog.pretrained.loading import fresh_files, read_record, SequenceLoader


def merge(results: list[Path], out: Path, checkpoint: Path, *, minimum_games=1000):
    if out.exists():
        raise FileExistsError("Choose a new merged collection identity")
    experiments = [json.loads((root / "experiment.json").read_text()) for root in results]
    collections = [json.loads((root / "fresh/collection.json").read_text()) for root in results]
    first = experiments[0]
    common = {key: value for key, value in first["collection"].items() if key != "shard"}
    if len(results) != common["shards"] or {e["collection"]["shard"] for e in experiments} != set(
        range(common["shards"])
    ):
        raise ValueError("Require every unique shard of the same frozen collection")
    for experiment, collection in zip(experiments, collections):
        recipe = {key: value for key, value in experiment["collection"].items() if key != "shard"}
        if (
            experiment["stage"] != "collect"
            or recipe != common
            or experiment["files"] != first["files"]
            or collection != collections[0]
        ):
            raise ValueError("Collection recipes, source inventories or opponent identities differ")
    if common["games"] < minimum_games or sha256(checkpoint) != collections[0]["native_sha256"]:
        raise ValueError("Insufficient planned games or different native checkpoint")
    out.mkdir(parents=True)
    (out / "collection.json").write_text(json.dumps(collections[0], indent=2) + "\n")
    inventory, sources, indices = {}, [], set()
    for root, experiment in zip(results, experiments):
        hashes = json.loads((root / "output_hashes.json").read_text())
        for name, expected in hashes.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
                raise ValueError("Returned output changed: " + name)
        for path in sorted((root / "fresh").glob("*.json.gz")):
            index = int(path.name.split(".")[0])
            if (
                not 0 <= index < common["games"]
                or index % common["shards"] != experiment["collection"]["shard"]
                or index in indices
            ):
                raise ValueError("Duplicate or incorrectly assigned collection slot")
            if sha256(path) != hashes["fresh/" + path.name]:
                raise ValueError("Fresh game was not in the returned output inventory")
            indices.add(index)
            shutil.copy2(path, out / path.name)
            inventory[path.name] = sha256(path)
        for path in (root / "fresh/failed_attempts").glob("*.json.gz"):
            target = out / "failed_attempts" / path.name
            target.parent.mkdir(exist_ok=True)
            if target.exists() or sha256(path) != hashes["fresh/failed_attempts/" + path.name]:
                raise ValueError("Duplicate or unverified failed-attempt evidence")
            shutil.copy2(path, target)
            inventory[target.relative_to(out).as_posix()] = sha256(path)
        sources.append(
            {
                "experiment": experiment["id"],
                "experiment_sha256": sha256(root / "experiment.json"),
                "output_hashes_sha256": sha256(root / "output_hashes.json"),
            }
        )
    if indices != set(range(common["games"])):
        raise ValueError(
            "Collection has missing slots; preserve these outputs and resume the frozen package"
        )
    files, audit = fresh_files(out, minimum_games=minimum_games)
    if audit["failures"] or len(files) != common["games"]:
        raise ValueError("Permanently failed slots cannot enter the matched training freeze")
    import torch

    torch.set_num_threads(1)
    loader = SequenceLoader(load_kakuna(checkpoint))
    seats = Counter()
    for path in files:
        record = read_record(Path(path))
        if set(record["submitted_teams"]) != {"p1", "p2"} or set(record["perspectives"]) != {
            "p1",
            "p2",
        }:
            raise ValueError("Fresh game lost submitted teams or a perspective")
        seats[(record["opponent"], record["schedule"]["p1_is_a"])] += 1
        # These fixed seeds select opposite perspectives, exercising both stored
        # live observation sequences against chronological native reconstruction.
        loader.load(("fresh", path, 0))
        loader.load(("fresh", path, 1))
    if any(
        seats[(opponent, True)] != seats[(opponent, False)] for opponent in audit["opponent_counts"]
    ):
        raise ValueError("Opponent schedule lost seat balance")
    report = {
        "passed": True,
        "games": len(files),
        "native_load_perspectives": 2 * len(files),
        "collection_sha256": sha256(out / "collection.json"),
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(first["files"], sort_keys=True).encode()
        ).hexdigest(),
        "sources": sources,
        "files": inventory,
        "audit": audit,
        "seat_balance": {
            opponent: [seats[(opponent, True)], seats[(opponent, False)]]
            for opponent in audit["opponent_counts"]
        },
        "failed_attempts": sum(name.startswith("failed_attempts/") for name in inventory),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "games",
                    "native_load_perspectives",
                    "seat_balance",
                    "failed_attempts",
                )
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--minimum-games", type=int, default=1000)
    args = parser.parse_args()
    merge(args.result, args.out, args.checkpoint, minimum_games=args.minimum_games)


if __name__ == "__main__":
    main()
