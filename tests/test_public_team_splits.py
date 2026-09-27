from __future__ import annotations

import hashlib
import json

from fog.scripts.build_public_team_splits import build_splits


def _bank(path, records):
    teams = path / "teams"
    teams.mkdir(parents=True)
    manifest_records = []
    for index, (paste, epoch) in enumerate(records):
        paste_hash = hashlib.sha256(paste.encode()).hexdigest()
        packed_hash = hashlib.sha256(f"packed:{paste}".encode()).hexdigest()
        filename = f"{index}.txt"
        (teams / filename).write_text(paste, encoding="utf-8")
        manifest_records.append(
            {
                "paste_sha256": paste_hash,
                "packed_sha256": packed_hash,
                "source_name": "recent" if "recent" in epoch else "legacy",
                "source_kind": "rmt_forum",
                "ruleset_epoch": epoch,
                "quality_tier": "candidate",
                "file": f"teams/{filename}",
            }
        )
    (path / "manifest.json").write_text(
        json.dumps({"source_errors": [], "teams": manifest_records}), encoding="utf-8"
    )


def test_build_splits_deduplicates_and_never_leaks(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _bank(first, [(f"team-{i}", "legacy") for i in range(20)])
    _bank(
        second,
        [("team-0", "recent-current"), *[(f"new-{i}", "recent-current") for i in range(20)]],
    )
    manifest = build_splits([first, second], tmp_path / "out", seed=7)
    assert manifest["unique_team_count"] == 40
    hashes = {
        split: {record["packed_sha256"] for record in records}
        for split, records in manifest["splits"].items()
    }
    assert not hashes["train"] & hashes["validation"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["validation"] & hashes["test"]
    assert sum(manifest["split_counts"].values()) == 40
    assert manifest["maximum_uniform_train_team_probability"] == 1 / 32


def test_composition_variants_stay_together_without_erasing_party_order(tmp_path):
    records = []
    for speed in range(12):
        first = (f"Nickname (Gholdengo) @ Leftovers\nAbility: Good as Gold\nEVs: {speed * 4} Spe\n"
                 "- Recover\n- Shadow Ball")
        second = "Dragapult @ Choice Specs\nAbility: Infiltrator\n- Shadow Ball\n- Draco Meteor"
        records.extend([(first + "\n\n" + second, "recent"),
                        (second + "\n\n" + first.replace("Nickname", "Different"), "recent")])
    bank = tmp_path / "bank"
    _bank(bank, records)
    manifest = build_splits([bank], tmp_path / "out", validation_fraction=.25, test_fraction=.25)
    groups = {name: {row["group_sha256"] for row in rows}
              for name, rows in manifest["splits"].items()}
    assert manifest["group_count"] == 12 and manifest["unique_team_count"] == 24
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["validation"]
    assert not groups["test"] & groups["validation"]
    assert manifest["split_counts"] == {"train": 12, "validation": 6, "test": 6}
