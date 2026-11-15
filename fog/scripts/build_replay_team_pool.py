"""Build a validator-clean, composition/time-split pool from public replay evidence.

One usage draw per side; rejected draws are dropped, never repaired or resampled. The optional
work directory checkpoints each replay so an interrupted build can resume. Only species names,
replay ids and numeric evidence counts are exported; nicknames and player metadata are discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fog.constants.schema import NATURES, RARE_SPECIES, SET_MOVE_COLS, TYPE_INDEX
from fog.constants.tables import TABLES
from fog.env.teams import SHOWDOWN_DIR
from fog.scripts.build_generative_pool import render_paste
from fog.training.replay_policy_data import (
    EXACT,
    INFERRED,
    complete_private_team,
    stable_replay_seed,
)
from fog.training.replay_supervision import _preview_species_id, public_replay_truth

SPLITS = ("train", "validation", "test")
CAPS = {"train": 600, "validation": 150, "test": 150}
FIELD_COLUMNS = {
    "species": (0,), "item": (1,), "ability": (2,), "moves": SET_MOVE_COLS,
    "nature": (7,), "evs": tuple(range(8, 14)), "tera": (14,),
}


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


class ShowdownValidator:
    """Reuse the pinned CLI's validator in one Node process instead of booting it per team."""

    def __init__(self, battle_format: str = "gen9ou") -> None:
        self.process = subprocess.Popen(
            ["node", "-e", """
const {Teams} = require('./dist/sim/teams.js');
const {TeamValidator} = require('./dist/sim/team-validator.js');
const validator = TeamValidator.get(process.argv[1]);
require('readline').createInterface({input: process.stdin}).on('line', line => {
    const team = Teams.import(JSON.parse(line));
    console.log(JSON.stringify(validator.validateTeam(team) || []));
});
""", battle_format], cwd=SHOWDOWN_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )

    def __call__(self, paste: str) -> list[str]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(paste) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("pinned Showdown validator exited unexpectedly")
        return json.loads(line)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.process.stdout.close()


def preview_composition(events: list[list[str]], role: str) -> tuple[str, ...] | None:
    species = [_preview_species_id(e[3]) for e in events
               if len(e) >= 4 and e[1:3] == ["poke", role]]
    if len(species) != 6 or any(
        s not in TABLES._species_idx or s == RARE_SPECIES for s in species
    ):
        return None
    return tuple(sorted(species))


def revealed_team(events: list[list[str]], role: str) -> np.ndarray:
    # Same adapter as replay_candidates.revealed_teams, applied independently to each side.
    preview = [SimpleNamespace(species=e[3].split(",")[0]) for e in events
               if len(e) >= 4 and e[1:3] == ["poke", role]]
    battle = SimpleNamespace(
        player_role="p2" if role == "p1" else "p1", opponent_team={},
        teampreview_opponent_team=preview, observations={0: SimpleNamespace(events=events)},
    )
    return public_replay_truth(battle, TABLES)


def tokens_to_paste(team: np.ndarray) -> str:
    if (team < 0).any():
        raise ValueError("completion left unknown fields")
    sets = []
    for row in team:
        item = TABLES.items["ids"][int(row[1])]
        if item.startswith("<") and item != "<NO_ITEM>":
            raise ValueError("completion contains an unknown item")
        sets.append({
            "species": TABLES.species["ids"][int(row[0])],
            "item": "" if item == "<NO_ITEM>" else item,
            "ability": TABLES.abilities["ids"][int(row[2])],
            "moves": [TABLES.moves["ids"][int(row[c])] for c in SET_MOVE_COLS],
            "nature": NATURES[int(row[7])], "evs": [int(v) * 4 for v in row[8:14]],
            "tera": (*TYPE_INDEX, "stellar")[int(row[14])],
        })
    return render_paste(sets, TABLES).replace(" @ \n", "\n")


def field_counts(provenance: np.ndarray, value: int) -> dict[str, int]:
    counts = {name: int((provenance[:, list(cols)] == value).sum())
              for name, cols in FIELD_COLUMNS.items()}
    return {**counts, "total": sum(counts.values())}


def process_replay(replay: dict, *, seed: int, validator) -> dict:
    rid = str(replay["id"])
    if not re.fullmatch(r"gen9ou-[a-z0-9-]+", rid):
        raise ValueError("invalid replay id")
    timestamp, rating = int(replay["uploadtime"]), int(replay["rating"])
    events = [line.split("|") for line in replay["log"].splitlines() if line.startswith("|")]
    counts: Counter = Counter(games=1, sides=2)
    previews, teams = [], []
    for role in ("p1", "p2"):
        composition = preview_composition(events, role)
        if composition is not None:
            previews.append({"composition": composition, "replay_id": rid,
                             "uploadtime": timestamp})
        if rating < 1700:
            counts["below_rating"] += 1
            continue
        counts["rated_sides"] += 1
        if composition is None:
            counts["incomplete_or_unknown_preview"] += 1
            continue
        counts["known_six_species"] += 1
        revealed = revealed_team(events, role)
        if int((revealed[:, list(SET_MOVE_COLS)] >= 0).sum()) < 14:
            counts["fewer_than_14_moves"] += 1
            continue
        counts["at_least_14_moves"] += 1
        completed, provenance = complete_private_team(
            revealed, seed=stable_replay_seed(rid, role, 0, seed),
        )
        try:
            paste = tokens_to_paste(completed)
        except ValueError:
            counts["unrenderable_completion"] += 1
            continue
        errors = validator(paste)
        if errors:
            counts["validator_rejected"] += 1
            continue
        counts["validator_accepted"] += 1
        teams.append({
            "composition": composition, "replay_id": rid, "side": role,
            "uploadtime": timestamp, "rating": rating, "paste": paste,
            "revealed_fields": field_counts(provenance, EXACT),
            "sampled_fields": field_counts(provenance, INFERRED),
        })
    return {"previews": previews, "teams": teams, "counts": dict(counts)}


def split_teams(teams: list[dict], previews: list[dict], holdout_ids: set[str], *,
                seed: int) -> tuple[dict[str, list[dict]], dict]:
    """Deduplicate after validation; older low-rated/uninformative previews still bar test entry."""
    older: set[tuple] = set()
    first_seen: dict[tuple, tuple[int, str]] = {}
    for preview in previews:
        composition = tuple(preview["composition"])
        seen = (preview["uploadtime"], preview["replay_id"])
        first_seen[composition] = min(first_seen.get(composition, seen), seen)
        if preview["replay_id"] not in holdout_ids:
            older.add(composition)
    best: dict[tuple, dict] = {}
    # Stable tie breaking uses replay id and side, independent of source traversal order.
    for team in sorted(teams, key=lambda t: (
        -t["revealed_fields"]["total"], t["replay_id"], t["side"],
    )):
        best.setdefault(tuple(team["composition"]), team)
    test = sorted(c for c in best if c not in older and first_seen[c][1] in holdout_ids)
    remaining = sorted(set(best) - set(test))
    random.Random(f"{seed}:validation").shuffle(remaining)
    n_validation = round(0.15 * len(remaining))
    keys = {"test": test, "validation": remaining[:n_validation],
            "train": remaining[n_validation:]}
    before_caps = {split: len(values) for split, values in keys.items()}
    splits = {}
    for split in SPLITS:
        values = keys[split]
        if len(values) > CAPS[split]:
            values = random.Random(f"{seed}:{split}:cap").sample(values, CAPS[split])
        splits[split] = [dict(best[c], first_seen_uploadtime=first_seen[c][0],
                              first_seen_replay_id=first_seen[c][1]) for c in sorted(values)]
    return splits, {"unique_valid_compositions": len(best),
                    "deduplicated_sides": len(teams) - len(best), "before_caps": before_caps}


def write_pool(out: Path, splits: dict[str, list[dict]], metadata: dict) -> dict:
    manifests = {}
    for split in SPLITS:
        directory = out / split
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("team_*.txt"):
            path.unlink()
        records = []
        for index, team in enumerate(splits[split]):
            filename = f"team_{index:03d}.txt"
            (directory / filename).write_text(team["paste"], encoding="utf-8")
            records.append({key: value for key, value in team.items() if key != "paste"}
                           | {"file": filename})
        manifest = {"format": "gen9ou", "seed": metadata["seed"], "split": split,
                    "teams": records}
        content = _json(manifest)
        (directory / "manifest.json").write_text(content, encoding="utf-8")
        manifests[split] = hashlib.sha256(content.encode()).hexdigest()
    report = {**metadata, "split_sizes": {s: len(splits[s]) for s in SPLITS},
              "split_manifest_sha256": manifests,
              "limits": "Unrevealed items, moves, abilities, Tera and all spreads are usage "
                        "samples; teams may be less coherent than human-built teams."}
    (out / "manifest.json").write_text(_json(report), encoding="utf-8")
    return report


def build_pool(raw: Path, holdout_file: Path, out: Path, *, seed: int = 0,
               work_dir: Path | None = None) -> dict:
    paths = sorted(raw.glob("gen9ou-*.json"))
    if not paths:
        raise ValueError("no imported Gen 9 OU replays")
    holdout_ids = set(holdout_file.read_text(encoding="utf-8").splitlines())
    ids = {p.stem for p in paths}
    if not holdout_ids or not holdout_ids <= ids:
        raise ValueError("holdout ids missing from raw import")
    # Cache version covers the deterministic build options; a different build needs a new dir.
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        options = {"version": 1, "seed": seed, "raw": str(raw.resolve())}
        config = work_dir / "options.json"
        if config.exists() and json.loads(config.read_text()) != options:
            raise ValueError("checkpoint options changed; choose a new work directory")
        config.write_text(_json(options), encoding="utf-8")
    counts: Counter = Counter()
    previews, teams, dates = [], [], {}
    with ShowdownValidator() as validator:
        for index, path in enumerate(paths):
            checkpoint = work_dir / path.name if work_dir is not None else None
            if checkpoint is not None and checkpoint.exists():
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
            else:
                replay = json.loads(path.read_text(encoding="utf-8"))
                if replay.get("id") != path.stem:
                    raise ValueError("replay id and filename differ")
                result = process_replay(replay, seed=seed, validator=validator)
                result["uploadtime"] = int(replay["uploadtime"])
                if checkpoint is not None:
                    temporary = checkpoint.with_suffix(".tmp")
                    temporary.write_text(_json(result), encoding="utf-8")
                    temporary.replace(checkpoint)
            counts.update(result["counts"])
            previews.extend(result["previews"])
            teams.extend(result["teams"])
            dates[path.stem] = result["uploadtime"]
            if (index + 1) % 1000 == 0:
                print(f"pool: {index + 1}/{len(paths)} games; {len(teams)} valid sides", flush=True)
    chronological = sorted(ids, key=lambda rid: (dates[rid], rid))[-len(holdout_ids):]
    if set(chronological) != holdout_ids:
        raise ValueError("fixed holdout is not the newest games in this import")
    splits, dedup = split_teams(teams, previews, holdout_ids, seed=seed)
    return write_pool(out, splits, {
        "seed": seed, "format": "gen9ou", "raw": str(raw),
        "holdout_ids": str(holdout_file), "holdout_games": len(holdout_ids),
        "rating_floor": 1700, "minimum_revealed_moves": 14,
        "validation_fraction": 0.15, "caps": CAPS, "filter_counts": dict(counts),
        **dedup,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/metamon_gen9ou_r1500"))
    parser.add_argument("--holdout-ids", type=Path,
                        default=Path("data/replay_splits/p2_holdout_2000_ids.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/teams/replay_v1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/replay_team_pool/build"))
    args = parser.parse_args()
    print(_json(build_pool(args.raw, args.holdout_ids, args.out, seed=args.seed,
                           work_dir=args.work_dir)), flush=True)


if __name__ == "__main__":
    main()
