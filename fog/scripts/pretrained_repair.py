"""Overlay explicit collection recoveries without replacing successful parent games."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from fog.pretrained.kakuna import load_kakuna, sha256
from fog.pretrained.loading import fresh_files, read_record, SequenceLoader


def verify_result(root):
    hashes = json.loads((root / "output_hashes.json").read_text())
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError("Returned output changed: " + name)
    experiment = json.loads((root / "experiment.json").read_text())
    collection = json.loads((root / "fresh/collection.json").read_text())
    if experiment["stage"] != "collect":
        raise ValueError("Recovery requires collection outputs")
    paths = {int(p.name.split(".")[0]): p for p in (root / "fresh").glob("*.json.gz")}
    for path in paths.values():
        if hashes.get("fresh/" + path.name) != sha256(path):
            raise ValueError("Game absent from returned inventory")
    return experiment, collection, paths


def compatible_collection(first, other):
    keys = (
        "seed",
        "games",
        "split",
        "pool_sha256",
        "native_sha256",
        "temperature",
        "opponent_mix",
        "foul_play_search_ms",
        "foul_play_sampling",
        "smoke_opponent",
    )
    if any(first.get(key) != other.get(key) for key in keys):
        raise ValueError("Recovery changed data, schedule, or opponent recipe")
    if list(first["learners"].values()) != list(other["learners"].values()):
        raise ValueError("Recovery changed frozen learners or their ordering")


def validate_recovery_inputs(first, other):
    protected = {
        name: digest
        for name, digest in first["files"].items()
        if name.startswith("inputs/")
        or "/pkmn_sets_cache/" in name
        or "/smogon_stats_cache/" in name
    }
    if first["upstream_revisions"] != other["upstream_revisions"] or any(
        other["files"].get(name) != digest for name, digest in protected.items()
    ):
        raise ValueError(
            "Recovery changed immutable inputs, opponent priors, or upstream revisions"
        )


def validate_replacement(original, repaired):
    keys = ("p1_is_a", "p1_team", "p2_team", "controller_seed")
    if (
        original["opponent"] != repaired["opponent"]
        or original["submitted_teams"] != repaired["submitted_teams"]
        or any(original["schedule"][key] != repaired["schedule"][key] for key in keys)
    ):
        raise ValueError("Recovery changed assigned opponent, team, seat, or controller")


def repair(parents, recoveries, out, checkpoint):
    if out.exists():
        raise FileExistsError("Choose a new repaired collection identity")
    inventory, sources, chosen, evidence = {}, [], {}, []
    first_experiment, first_collection, _ = verify_result(parents[0])
    common = {k: v for k, v in first_experiment["collection"].items() if k != "shard"}
    shards = set()
    for root in parents:
        experiment, collection, paths = verify_result(root)
        if (
            collection != first_collection
            or experiment["files"] != first_experiment["files"]
            or {k: v for k, v in experiment["collection"].items() if k != "shard"} != common
        ):
            raise ValueError("Original shard identities differ")
        shard = experiment["collection"]["shard"]
        if shard in shards:
            raise ValueError("Duplicate original shard")
        shards.add(shard)
        for slot, path in paths.items():
            if slot in chosen or slot % common["shards"] != shard:
                raise ValueError("Duplicate or incorrectly assigned original slot")
            chosen[slot] = path
        sources.append((root, experiment))
    if shards != set(range(common["shards"])) or set(chosen) != set(range(common["games"])):
        raise ValueError("Original collection is incomplete")
    if common["games"] < 1000 or sha256(checkpoint) != first_collection["native_sha256"]:
        raise ValueError("Require the full schedule and original checkpoint")
    for root in recoveries:
        experiment, collection, paths = verify_result(root)
        validate_recovery_inputs(first_experiment, experiment)
        compatible_collection(first_collection, collection)
        pending = {
            slot for slot, path in chosen.items() if not read_record(path)["training_eligible"]
        }
        if set(experiment["collection"].get("slots", [])) != pending or set(paths) != pending:
            raise ValueError("Recovery must cover exactly the remaining failed slots")
        bundle = root.parent.parent / "dataset/bundle.bin"
        if not bundle.exists():
            bundle = root.parent.parent / "dataset/bundle.zip"
        with zipfile.ZipFile(bundle) as frozen:
            provenance_bytes = frozen.read("fog/inputs/recovery_provenance.json")
        if (
            hashlib.sha256(provenance_bytes).hexdigest()
            != experiment["files"]["inputs/recovery_provenance.json"]
        ):
            raise ValueError("Recovery provenance changed after freezing")
        provenance = json.loads(provenance_bytes)
        expected_parents = {e["id"]: sha256(p / "output_hashes.json") for p, e in sources}
        actual_parents = {
            row["experiment"]: row["output_hashes_sha256"] for row in provenance["parents"]
        }
        if actual_parents != expected_parents:
            raise ValueError("Recovery lineage does not match the preceding outputs")
        proofs = {row["slot"]: row for row in provenance["slots"]}
        if set(proofs) != pending:
            raise ValueError("Recovery failed-slot proof differs")
        for slot, path in paths.items():
            old = chosen[slot]
            if sha256(old) != proofs[slot]["sha256"]:
                raise ValueError("Recovery parent slot changed")
            validate_replacement(read_record(old), read_record(path))
            source = next(e["id"] for p, e in sources if old.is_relative_to(p))
            evidence.append((old, source + "-parent"))
            chosen[slot] = path
        sources.append((root, experiment))
    if any(not read_record(path)["training_eligible"] for path in chosen.values()):
        raise ValueError("Failed slots remain; preserve outputs and diagnose them")
    out.mkdir(parents=True)
    descriptor = dict(
        first_collection, version=7, recovery_source_experiments=[e["id"] for _, e in sources]
    )
    (out / "collection.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    for slot, path in sorted(chosen.items()):
        target = out / f"{slot:06d}.json.gz"
        shutil.copy2(path, target)
        inventory[target.name] = sha256(target)
    for root, experiment in sources:
        for path in (root / "fresh/failed_attempts").glob("*.json.gz"):
            evidence.append((path, experiment["id"]))
    for path, source in evidence:
        target = out / "failed_attempts" / source / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Failed evidence would be overwritten")
        shutil.copy2(path, target)
        inventory[target.relative_to(out).as_posix()] = sha256(target)
    files, audit = fresh_files(out, minimum_games=1000)
    if len(files) != common["games"] or audit["failures"]:
        raise ValueError("Recovered corpus has ineligible games")
    import torch

    torch.set_num_threads(1)
    loader, seats = SequenceLoader(load_kakuna(checkpoint)), Counter()
    for index, path in enumerate(files):
        record = read_record(Path(path))
        seats[(record["opponent"], record["schedule"]["p1_is_a"])] += 1
        for seed in (0, 1):
            loader.load(("fresh", path, seed))
        if (index + 1) % 100 == 0:
            print(json.dumps({"validated_games": index + 1}), flush=True)
    if any(
        seats[(opponent, True)] != seats[(opponent, False)] for opponent in audit["opponent_counts"]
    ):
        raise ValueError("Recovered collection lost seat balance")
    report = {
        "passed": True,
        "games": len(files),
        "native_load_perspectives": 2 * len(files),
        "collection_sha256": sha256(out / "collection.json"),
        "files": inventory,
        "audit": audit,
        "sources": [
            {
                "experiment": e["id"],
                "experiment_sha256": sha256(p / "experiment.json"),
                "output_hashes_sha256": sha256(p / "output_hashes.json"),
                "source_inventory_sha256": hashlib.sha256(
                    json.dumps(e["files"], sort_keys=True).encode()
                ).hexdigest(),
            }
            for p, e in sources
        ],
        "recovery_preserves_successful_parent_bytes": True,
        "seat_balance": {
            o: [seats[(o, True)], seats[(o, False)]] for o in audit["opponent_counts"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "games": len(files), "native_load_perspectives": 2 * len(files)}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, action="append", required=True)
    parser.add_argument("--recovery", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    repair(args.parent, args.recovery, args.out, args.checkpoint)


if __name__ == "__main__":
    main()
