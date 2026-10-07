"""Verify historical replay pairs and original rosters against protected team pools."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

from fog.env.teams import to_packed
from fog.pretrained.actions import identifier
from fog.pretrained.kakuna import sha256
from fog.pretrained.loading import read_record


def base_species(showdown: Path) -> dict[str, str]:
    script = "const {Dex}=require(process.argv[1]); console.log(JSON.stringify(Object.fromEntries(Dex.species.all().map(p=>[p.id,p.baseSpecies]))));"
    output = subprocess.check_output(
        ["node", "-e", script, str((showdown / "dist/sim").resolve())],
        text=True,
        encoding="utf-8",
    )
    return {key: identifier(value) for key, value in json.loads(output).items()}


def original_roster(record, bases):
    state = record["states"][0]
    own = [state["player_active_pokemon"], *state["available_switches"]]
    # Replay reconstruction can change `name` through Transform or Illusion.
    # Its base_species stores the original had_name; use it for composition isolation.
    names = [identifier(mon["base_species"]) for mon in own]
    if len(names) != 6 or any(name not in bases for name in names):
        raise ValueError("Original replay roster is incomplete or unknown to Showdown")
    return tuple(sorted(bases[name] for name in names))


def audit(published: Path, pools: Path, showdown: Path, out: Path):
    if out.exists():
        raise FileExistsError("Choose a new audit identity")
    manifest = json.loads((published / "manifest.json").read_text())
    bases = base_species(showdown)
    roster_pools = {}
    for split in ("train", "development", "final"):
        rosters = set()
        for team in json.loads((pools / (split + ".json")).read_text())["teams"]:
            roster = [
                identifier(row.split("|")[1] or row.split("|")[0])
                for row in to_packed(team).split("]")
            ]
            rosters.add(tuple(sorted(bases[name] for name in roster)))
        roster_pools[split] = rosters
    names = list(roster_pools)
    if any(roster_pools[a] & roster_pools[b] for i, a in enumerate(names) for b in names[i + 1 :]):
        raise ValueError("Team pools overlap after original base-species normalization")
    protected = roster_pools["development"] | roster_pools["final"]
    declared = {tuple(row) for row in manifest["protected_base_rosters"]}
    if declared != protected or manifest["pool_manifest_sha256"] != sha256(pools / "manifest.json"):
        raise ValueError("Replay exclusion manifest belongs to different held-out teams")
    groups, rejected, reasons = defaultdict(list), set(), Counter()
    for index, row in enumerate(manifest["records"]):
        path = (published / row["path"]).resolve()
        if not path.is_relative_to(published.resolve()) or sha256(path) != row["sha256"]:
            raise ValueError("Historical frozen trajectory changed")
        record = read_record(path)
        try:
            if original_roster(record, bases) in protected:
                rejected.add(row["match_id"])
                reasons["protected_original_roster"] += 1
        except (KeyError, ValueError) as error:
            rejected.add(row["match_id"])
            reasons[str(error)] += 1
        terminal = record["states"][-1]
        if not (bool(terminal["battle_won"]) ^ bool(terminal["battle_lost"])):
            rejected.add(row["match_id"])
            reasons["nondecisive_terminal"] += 1
        groups[row["match_id"]].append(bool(terminal["battle_won"]))
        if (index + 1) % 5000 == 0:
            print(
                json.dumps({"audited_perspectives": index + 1, "rejected_matches": len(rejected)}),
                flush=True,
            )
    for match_id, outcomes in groups.items():
        if len(outcomes) != 2 or set(outcomes) != {False, True}:
            rejected.add(match_id)
            reasons["incomplete_or_inconsistent_pair"] += 1
    report = {
        "passed": not rejected,
        "published_manifest_sha256": sha256(published / "manifest.json"),
        "pool_manifest_sha256": sha256(pools / "manifest.json"),
        "training_pool_sha256": sha256(pools / "train.json"),
        "showdown_revision": subprocess.check_output(
            ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True
        ).strip(),
        "perspectives": len(manifest["records"]),
        "matches": len(groups),
        "protected_base_rosters": len(protected),
        "base_roster_pool_counts": {split: len(rosters) for split, rosters in roster_pools.items()},
        "pool_base_rosters_disjoint": True,
        "rejected_matches": sorted(rejected),
        "reasons": dict(reasons),
        "roster_field": "base_species (original replay had_name), normalized by Showdown Dex",
        "limitation": "Human own sets remain reconstructed; all human belief labels are unverified. Historical pretrained exposure cannot be excluded.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: report[key] for key in ("passed", "matches", "perspectives", "reasons")}),
        flush=True,
    )
    if rejected:
        raise ValueError("Original-roster isolation audit failed; create a new filtered freeze")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--showdown", type=Path, default=Path("server/pokemon-showdown"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.published, args.pools, args.showdown, args.out)


if __name__ == "__main__":
    main()
