"""Freeze complete, recent human Gen9OU replay pairs with conservative holdouts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

from fog.pretrained.actions import identifier
from fog.pretrained.kakuna import sha256
from fog.env.teams import to_packed

MATCH_ID = re.compile(r"^(?:smogtours-)?gen9ou-\d+")


def freeze(
    archive,
    pools,
    out,
    showdown,
    *,
    max_matches=50000,
    byte_cap=15_000_000_000,
    min_date=date(2025, 4, 1),
    min_rating=1500,
):
    source = json.loads(archive.with_suffix(archive.suffix + ".json").read_text())
    if sha256(archive) != source["sha256"]:
        raise ValueError("Human replay archive hash mismatch")
    if out.exists():
        raise FileExistsError(out)
    protected = set()
    heldout = []
    for split in ("development", "final"):
        for team in json.loads((pools / (split + ".json")).read_text())["teams"]:
            roster = [
                identifier(row.split("|")[1] or row.split("|")[0])
                for row in to_packed(team).split("]")
            ]
            heldout.append(roster)
    # Formes can be concealed at preview. Exclude matching base-species rosters,
    # which is deliberately stricter than the legal split's species compositions.
    dex = str((showdown / "dist/sim").resolve())
    script = "const {Dex}=require(process.argv[1]); console.log(JSON.stringify(Object.fromEntries(Dex.species.all().map(p=>[p.id,p.baseSpecies]))));"
    result = subprocess.run(
        ["node", "-e", script, dex], text=True, encoding="utf-8", capture_output=True, check=True
    )
    bases = {key: identifier(value) for key, value in json.loads(result.stdout).items()}

    def roster_key(roster):
        # Unknown forme spellings are mapped by the actual Dex below where needed.
        return tuple(sorted(bases.get(identifier(name), identifier(name)) for name in roster))

    protected.update(roster_key(roster) for roster in heldout)
    out.mkdir(parents=True)
    staging = out / "staging"
    staging.mkdir()
    groups, counts, months = {}, Counter(), Counter()
    blocked = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            match = MATCH_ID.match(name)
            if not member.isfile() or not name.endswith(".json.lz4") or not match:
                continue
            match_id = match.group()
            counts["seen_perspectives"] += 1
            if match_id in blocked:
                continue
            suffix = name[:-9].rsplit("_", 2)
            try:
                month, day, year = map(int, suffix[-2].split("-"))
                played = date(year, month, day)
                months[f"{year}-{month:02d}"] += 1
                rating = name[len(match_id) + 1 :].split("_", 1)[0]
                strong = match_id.startswith("smogtours-") or (
                    rating.isdigit() and int(rating) >= min_rating
                )
            except (IndexError, ValueError):
                counts["unreadable_metadata"] += 1
                continue
            if played < min_date:
                counts["before_date"] += 1
                continue
            if not strong:
                # Require both rated POVs to clear the threshold; otherwise archive
                # order could preferentially retain upsets with a high-rated loser.
                counts["below_rating"] += 1
                blocked.add(match_id)
                for old in groups.pop(match_id, []):
                    (staging / old["path"]).unlink()
                continue
            if match_id not in groups and len(groups) >= max_matches:
                counts["match_cap"] += 1
                continue
            if member.size > 32 * 1024 * 1024:
                raise ValueError("Oversized human trajectory")
            payload = tar.extractfile(member).read()
            import lz4.frame

            record = json.loads(lz4.frame.decompress(payload))
            states, actions = record["states"], record["actions"]
            reason = None
            if len(states) != len(actions) or len(states) < 3 or actions[-1] != -1:
                reason = "invalid_trajectory"
            elif not (bool(states[-1]["battle_won"]) ^ bool(states[-1]["battle_lost"])):
                reason = "unverified_terminal"
            else:
                own = [states[0]["player_active_pokemon"]["base_species"]] + [
                    p["base_species"] for p in states[0]["available_switches"]
                ]
                if len(own) != 6:
                    reason = "unknown_roster"
                elif roster_key(own) in protected:
                    reason = "protected_roster"
                elif any(
                    identifier(move["name"])
                    in {"terablast", "batonpass", "lastrespects", "shedtail"}
                    for state in states
                    for mon in [state["player_active_pokemon"], *state["available_switches"]]
                    for move in mon["moves"]
                ):
                    reason = "historically_banned_move"
            if reason:
                counts[reason] += 1
                blocked.add(match_id)
                for old in groups.pop(match_id, []):
                    (staging / old["path"]).unlink()
                continue
            digest = hashlib.sha256(payload).hexdigest()
            destination = digest + ".json.lz4"
            (staging / destination).write_bytes(payload)
            groups.setdefault(match_id, []).append(
                {
                    "path": destination,
                    "sha256": digest,
                    "bytes": len(payload),
                    "member": member.name,
                    "played": str(played),
                    "rating": rating,
                    "won": bool(states[-1]["battle_won"]),
                    "decisions": len(actions) - 1,
                }
            )
            if counts["seen_perspectives"] % 10000 == 0:
                print(
                    "Scanned",
                    counts["seen_perspectives"],
                    "perspectives; candidate matches",
                    len(groups),
                    flush=True,
                )
    destination = out / "train/gen9ou"
    destination.mkdir(parents=True)
    matches, records, total = [], [], 0
    for match_id, pair in sorted(groups.items()):
        if len(pair) != 2 or {row["won"] for row in pair} != {True, False}:
            counts["incomplete_or_ambiguous_pair"] += 1
            continue
        size = sum(row["bytes"] for row in pair)
        if total + size > byte_cap:
            counts["byte_cap"] += 1
            continue
        for row in pair:
            (staging / row["path"]).replace(destination / row["path"])
            row["path"] = "train/gen9ou/" + row["path"]
            row["match_id"] = match_id
            records.append(row)
        total += size
        matches.append(match_id)
    # Remove only this run's unselected staging files, using verified local paths.
    for path in staging.iterdir():
        if path.is_file() and path.suffix == ".lz4":
            path.unlink()
    staging.rmdir()
    manifest = {
        "source": source,
        "matches": len(matches),
        "perspectives": len(records),
        "bytes": total,
        "records": records,
        "counts": dict(counts),
        "min_date": str(min_date),
        "min_rating": min_rating,
        "months_in_source": dict(sorted(months.items())),
        "max_matches": max_matches,
        "byte_cap": byte_cap,
        "split": "complete human POV pairs are training-only; evaluation uses fresh held-out games",
        "belief_labels": "All human reconstructed set fields are unverified and masked.",
        "outcome_labels": "Explicit terminal won/lost, opposite outcomes in each complete pair.",
        "historical_exposure": "May overlap Kakuna pretraining; published archive ends June 2025.",
        "protected_base_rosters": [list(roster) for roster in sorted(protected)],
        "pool_manifest_sha256": sha256(pools / "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not matches:
        raise ValueError(
            "No complete eligible recent human match pairs; inspect counters before changing recipe"
        )
    print(
        json.dumps({key: manifest[key] for key in ("matches", "perspectives", "bytes", "counts")}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--showdown", type=Path, default=Path("server/pokemon-showdown"))
    parser.add_argument("--max-matches", type=int, default=50000)
    args = parser.parse_args()
    freeze(args.archive, args.pools, args.out, args.showdown, max_matches=args.max_matches)


if __name__ == "__main__":
    main()
