"""Bounded training-only set priors; reconstructed archives remain prior evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tarfile

import numpy as np

from fog.constants.schema import NATURE_INDEX, NO_ITEM, RARE_SPECIES, TYPE_INDEX
from fog.constants.tables import TABLES
from fog.pretrained.data import composition
from fog.training.replay_candidates import vocabulary_digest


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identifier(value):
    return re.sub("[^a-z0-9]", "", str(value).lower())


def validated_row(mon, tables=TABLES):
    """Encode an already validator-approved set in the existing 15-field schema.

    Complete categorical priors cannot invent omitted moves, abilities or Tera
    types. These unsupported records are rejected; this is not a replay-label
    reconstruction interface. IVs are deliberately outside the encoded schema.
    """
    species = tables.species_index(identifier(mon["species"]))
    if species == tables.species_index(RARE_SPECIES):
        raise ValueError("Species outside the encoded prior vocabulary")
    moves = [tables.move_index(identifier(move)) for move in mon.get("moves", [])]
    if len(moves) != 4 or len(set(moves)) != 4:
        raise ValueError("Prior requires four represented distinct moves")
    if not tables.learnset_mask(species)[moves].all():
        raise ValueError("Move is outside the modeled species support")
    ability = tables.ability_index(identifier(mon.get("ability", "")))
    if not tables.legal_abilities_mask(species)[ability]:
        raise ValueError("Ability is outside the modeled species support")
    nature = NATURE_INDEX[identifier(mon.get("nature") or "Serious")]
    tera = TYPE_INDEX[identifier(mon["teraType"])]
    evs = [
        int(mon.get("evs", {}).get(stat, 0)) // 4
        for stat in ("hp", "atk", "def", "spa", "spd", "spe")
    ]
    if min(evs) < 0 or max(evs) > 63 or sum(evs) > 127:
        raise ValueError("Invalid represented EV budget")
    item = (
        tables.item_index(identifier(mon["item"]))
        if mon.get("item")
        else tables.item_index(NO_ITEM)
    )
    return np.array([species, item, ability, *sorted(moves), nature, *evs, tera], dtype=np.int32)


def candidates(archive, excluded, *, maximum, seed):
    """Uniform bounded reservoir over permitted archive members, before validation."""
    if maximum < 1:
        raise ValueError("Positive candidate budget required")
    rng, selected, eligible = random.Random(seed), [], 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if not member.isfile() or not member.name.endswith(".gen9ou_team"):
                continue
            if member.size > 64 * 1024:
                raise ValueError("Oversized archived team")
            paste = stream.extractfile(member).read().decode("utf-8")
            try:
                roster = composition(paste)
            except ValueError:
                continue
            if roster in excluded:
                continue
            row = {"paste": paste, "member": member.name, "composition": roster}
            eligible += 1
            if len(selected) < maximum:
                selected.append(row)
            else:
                position = rng.randrange(eligible)
                if position < maximum:
                    selected[position] = row
    rng.shuffle(selected)
    return selected, eligible


def validate_export(pastes, showdown):
    worker = Path(__file__).resolve().parents[1] / "env/team_export_worker.js"
    result = subprocess.run(
        ["node", str(worker), str(Path(showdown).resolve())],
        input="".join(json.dumps(paste) + "\n" for paste in pastes),
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(pastes):
        raise ValueError("Prior validator omitted or added source records")
    return rows


def build_prior(
    archive,
    pools,
    out,
    showdown,
    *,
    maximum_teams=20000,
    minimum_teams=5000,
    maximum_candidates=40000,
    seed=20261019,
):
    """Expand prior evidence without using a protected team roster as a donor.

    Archive sets can be reconstructed. They are explicitly prior-only data,
    never verified historical labels. Dates below express synthetic availability,
    rather than a fabricated historical timestamp for the underlying battle.
    """
    archive, pools, out = Path(archive), Path(pools), Path(out)
    if out.exists() or not 1 <= minimum_teams <= maximum_teams <= maximum_candidates:
        raise ValueError("Require a new output and consistent bounded team budgets")
    pool_data = {
        name: json.loads((pools / (name + ".json")).read_text())
        for name in ("development", "final")
    }
    archive_sha = digest(archive)
    pool_hashes = {name: digest(pools / (name + ".json")) for name in pool_data}
    excluded = {composition(team) for pool in pool_data.values() for team in pool["teams"]}
    rows, eligible = candidates(archive, excluded, maximum=maximum_candidates, seed=seed)
    tokens, rosters, ids, hashes, compositions, sources = [], [], [], [], [], []
    counts, seen = Counter(), set()
    for start in range(0, len(rows), 200):
        chunk = rows[start : start + 200]
        answers = validate_export([row["paste"] for row in chunk], showdown)
        for source, answer in zip(chunk, answers):
            counts["validated_candidates"] += 1
            if not answer["valid"]:
                counts["illegal_teams"] += 1
                continue
            try:
                team = np.stack([validated_row(mon) for mon in answer["team"]])
                if team.shape != (6, 15):
                    raise ValueError("Incomplete team export")
            except (KeyError, ValueError, IndexError):
                counts["unrepresented_teams"] += 1
                continue
            canonical = tuple(sorted(tuple(row) for row in team.tolist()))
            if canonical in seen:
                counts["duplicate_encoded_teams"] += 1
                continue
            seen.add(canonical)
            team_hash = hashlib.sha256(source["paste"].encode()).hexdigest()
            if source["composition"] in excluded:
                raise ValueError("Protected roster entered the prior bank")
            for mon in team:
                tokens.append(mon)
                rosters.append(team[:, 0])
                ids.append(source["member"])
                hashes.append(team_hash)
                compositions.append(source["composition"])
            sources.append({key: source[key] for key in ("member", "composition")})
            counts["accepted_teams"] += 1
            if counts["accepted_teams"] == maximum_teams:
                break
        print(json.dumps(dict(counts)), flush=True)
        if counts["accepted_teams"] == maximum_teams:
            break
    if counts["accepted_teams"] < minimum_teams:
        raise ValueError(
            "Too few valid, supported, training-only prior teams: " + str(dict(counts))
        )
    if digest(archive) != archive_sha or any(
        digest(pools / (name + ".json")) != value for name, value in pool_hashes.items()
    ):
        raise ValueError("Prior sources changed during construction")
    metadata = {
        "schema_version": 1,
        "format": "gen9ou",
        "vocabulary_sha256": vocabulary_digest(),
        "private_truth": False,
        "prior_only": True,
        "label_provenance": "potentially reconstructed published sets",
        "method": "validated_training_archive_prior_v1",
        "donor_rows": len(tokens),
        "counts": dict(counts),
        "eligible_archive_members": eligible,
        "sampled_candidates": len(rows),
        "seed": seed,
        "archive_name": archive.name,
        "archive_sha256": archive_sha,
        "pool_sha256": pool_hashes,
        "excluded_compositions": sorted(excluded),
        "timestamp_scope": "synthetic prior availability: donor=1, evaluation=2; no historical date claim",
        "source_scope": "Deduplicated encoded teams, not calibrated ladder usage frequencies. Individual sets may recur across distinct training rosters.",
    }
    out.mkdir(parents=True)
    with (out / "bank.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            tokens=np.asarray(tokens),
            rosters=np.asarray(rosters),
            replay_ids=np.asarray(ids),
            log_hashes=np.asarray(hashes),
            timestamps=np.ones(len(tokens), dtype=np.int64),
            ratings=np.zeros(len(tokens), dtype=np.int32),
            composition_ids=np.asarray(compositions),
            metadata=np.asarray(json.dumps(metadata)),
        )
    metadata["bank_sha256"] = digest(out / "bank.npz")
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps(sources) + "\n")
    return metadata
