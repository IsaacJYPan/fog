"""Merge validated public team banks into deterministic, leakage-safe packages.

Exact packed-team hashes never cross splits.  Each input manifest must come from
``fog.scripts.collect_team_bank``, which records the pinned Showdown validation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _stratum(origins: list[dict]) -> str:
    if any(
        record.get("source_name") == "current_ou_samples"
        or "recent" in str(record.get("ruleset_epoch", "")).lower()
        for record in origins
    ):
        return "recent_current_legal"
    return "legacy_current_legal"


def build_splits(
    banks: list[Path],
    out_dir: Path,
    *,
    seed: int = 20260729,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    by_hash: dict[str, dict] = {}
    origins: dict[str, list[dict]] = defaultdict(list)
    for bank in banks:
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("source_errors"):
            raise ValueError(f"{bank} has source errors; refusing a claim-bearing split")
        for record in manifest["teams"]:
            packed_hash = record["packed_sha256"]
            paste = (bank / record["file"]).read_text(encoding="utf-8")
            if hashlib.sha256(paste.encode()).hexdigest() != record["paste_sha256"]:
                raise ValueError(f"paste hash mismatch for {bank / record['file']}")
            by_hash.setdefault(packed_hash, {"paste": paste, "record": record})
            origins[packed_hash].append(
                {
                    "bank": str(bank),
                    "source_name": record["source_name"],
                    "source_kind": record["source_kind"],
                    "ruleset_epoch": record["ruleset_epoch"],
                    "quality_tier": record["quality_tier"],
                }
            )

    strata: dict[str, list[str]] = defaultdict(list)
    for packed_hash in sorted(by_hash):
        strata[_stratum(origins[packed_hash])].append(packed_hash)

    split_hashes: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for stratum, hashes in sorted(strata.items()):
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(hashes)
        n_test = round(len(hashes) * test_fraction)
        n_validation = round(len(hashes) * validation_fraction)
        split_hashes["test"].extend(hashes[:n_test])
        split_hashes["validation"].extend(hashes[n_test : n_test + n_validation])
        split_hashes["train"].extend(hashes[n_test + n_validation :])

    out_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict]] = {}
    for split, hashes in split_hashes.items():
        hashes.sort()
        (out_dir / f"{split}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": "gen9ou",
                    "split": split,
                    "teams": [by_hash[value]["paste"] for value in hashes],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        split_records[split] = [
            {
                "packed_sha256": value,
                "stratum": _stratum(origins[value]),
                "origins": origins[value],
            }
            for value in hashes
        ]

    manifest = {
        "schema_version": 1,
        "format": "gen9ou",
        "seed": seed,
        "banks": [str(bank) for bank in banks],
        "unique_team_count": len(by_hash),
        "split_counts": {name: len(values) for name, values in split_hashes.items()},
        "stratum_counts": dict(Counter(_stratum(value) for value in origins.values())),
        "maximum_uniform_train_team_probability": (
            1.0 / len(split_hashes["train"]) if split_hashes["train"] else None
        ),
        "splits": split_records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        action="append",
        type=Path,
        dest="banks",
        required=True,
        help="Validated collector output directory; repeat for multiple banks.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = build_splits(
        args.banks,
        args.out,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    print(
        f"wrote {manifest['unique_team_count']} unique teams: "
        f"{manifest['split_counts']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
