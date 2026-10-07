"""Build legal composition-disjoint pools from pinned local team archives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import tarfile

from fog.pretrained.data import assert_disjoint, composition, file_hashes, split_teams


def build(archives: list[Path], out: Path, showdown: Path, seed: int) -> dict:
    if out.exists():
        raise FileExistsError("Choose a new pool identity; existing splits are immutable")
    records, seen = [], set()
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".gen9ou_team"):
                    continue
                if member.size > 64 * 1024:
                    raise ValueError("Unexpected oversized team record")
                paste = tar.extractfile(member).read().decode("utf-8")
                # Roster grouping is stricter than exact-set composition grouping.
                try:
                    group = composition(paste)
                except ValueError:
                    continue
                if group not in seen:
                    records.append((paste, str(archive), member.name, group))
                    seen.add(group)
    random.Random(seed).shuffle(records)
    # Process manageable chunks to avoid pipe deadlock and bound validator memory.
    accepted, errors = [], Counter()
    worker = Path(__file__).resolve().parents[1] / "env/team_validation_worker.js"
    for offset in range(0, len(records), 300):
        chunk = records[offset : offset + 300]
        run = subprocess.run(
            ["node", str(worker), str(showdown), "gen9ou"],
            input="".join(json.dumps(row[0]) + "\n" for row in chunk),
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        answers = [json.loads(line) for line in run.stdout.splitlines()]
        if len(answers) != len(chunk):
            raise ValueError("Validator did not return one record per team")
        for record, answer in zip(chunk, answers):
            if answer["valid"]:
                accepted.append(record)
            else:
                errors.update(answer["problems"])
        print(
            f"Validated {offset + len(chunk)} candidates; {len(accepted)} legal rosters", flush=True
        )
        if len(accepted) >= 700:
            break
    splits = split_teams([row[0] for row in accepted], seed=seed)
    assert_disjoint(splits)
    out.mkdir(parents=True)
    for name, pool in splits.items():
        (out / (name + ".json")).write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")
    revision = subprocess.check_output(
        ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "seed": seed,
        "grouping": "species roster, independent of sets and ordering",
        "showdown_revision": revision,
        "format": "gen9ou",
        "archives": file_hashes(archives),
        "counts": {name: len(pool["teams"]) for name, pool in splits.items()},
        "submitted_team_truth": "Exact only after these chosen pastes are submitted to new games",
        "origin": "Published Metamon teams; some source sets are reconstructed",
        "accepted_provenance": [
            {"archive": row[1], "member": row[2], "composition": row[3]} for row in accepted
        ],
        "rejection_reasons": dict(errors),
        "candidate_bank": "train.json only",
        "final_cohorts": 2,
        "pool_hashes": file_hashes(sorted(out.glob("*.json"))),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--showdown", type=Path, default=Path("server/pokemon-showdown"))
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.out, args.showdown.resolve(), args.seed)["counts"]))


if __name__ == "__main__":
    main()
