"""Replay-derived joint set donors, with explicit source and temporal exclusion.

Only publicly observed facts enter the bank. One donor supplies correlated fields; residual
unknowns still use the usage baseline. This is a heuristic imputer, not a calibrated posterior.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fog.constants.schema import RARE_SPECIES, SET_MOVE_COLS, SET_N_ATTRS
from fog.constants.tables import TABLES
from fog.training.replay_supervision import public_replay_truth


def vocabulary_digest() -> str:
    values = [getattr(TABLES, field)["ids"] for field in ("species", "items", "abilities", "moves")]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def replay_digest(replay: dict) -> str:
    # Normalize line endings; duplicate public logs must not acquire extra sampling weight.
    log = "\n".join(str(replay["log"]).splitlines())
    return hashlib.sha256(log.encode()).hexdigest()


def revealed_teams(replay: dict) -> dict[str, np.ndarray]:
    """Extract public submitted-set evidence without constructing thousands of observations.

    Reuses the production transfer/Illusion/Transform handling. Requests and private team sheets
    are never inputs. Require Gen9 OU preview rather than inventing species for missing slots.
    """
    events = [line.split("|") for line in replay["log"].splitlines() if line.startswith("|")]
    result = {}
    for role in ("p1", "p2"):
        preview = [SimpleNamespace(species=e[3].split(",")[0]) for e in events
                   if len(e) >= 4 and e[1] == "poke" and e[2] == role]
        if len(preview) != 6:
            raise ValueError("candidate replay requires six preview members on each side")
        battle = SimpleNamespace(
            player_role="p2" if role == "p1" else "p1", opponent_team={},
            teampreview_opponent_team=preview,
            observations={0: SimpleNamespace(events=events)},
        )
        result[role] = public_replay_truth(battle, TABLES)
    return result


def canonical_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.int64).copy()
    moves = sorted(set(int(m) for m in row[3:7] if m >= 0))
    row[3:7] = moves + [-1] * (4 - len(moves))
    return row


def valid_row(row: np.ndarray) -> bool:
    species = int(row[0])
    if species < 0 or species >= TABLES.n_species or species == TABLES.species_index(RARE_SPECIES):
        return False
    moves = row[3:7][row[3:7] >= 0]
    return bool(
        len(moves) >= 2 and len(set(moves.tolist())) == len(moves)
        and np.all(moves < TABLES.n_moves) and TABLES.learnset_mask(species)[moves].all()
        and (row[2] < 0 or (row[2] < TABLES.n_abilities
                           and TABLES.legal_abilities_mask(species)[row[2]]))
    )


def build_bank(paths: list[Path], out: Path, *, excluded_ids: set[str] | None = None) -> dict:
    """Build once from the training partition; no imputed records are ever recycled as evidence."""
    if out.exists():
        raise FileExistsError(out)
    excluded_ids = excluded_ids or set()
    rows, rosters, ids, hashes, dates, ratings = [], [], [], [], [], []
    seen: set[str] = set()
    counts: Counter = Counter()
    sources: set[str] = set()
    # Pre-read excluded logs so their duplicates under another ID are excluded as well.
    excluded_hashes = set()
    if excluded_ids:
        for path in paths:
            replay = json.loads(path.read_text(encoding="utf-8"))
            if path.stem in excluded_ids or str(replay.get("id")) in excluded_ids:
                excluded_hashes.add(replay_digest(replay))
    for index, path in enumerate(paths):
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            rid = str(replay.get("id") or path.stem)
            digest = replay_digest(replay)
            if rid in excluded_ids or path.stem in excluded_ids or digest in excluded_hashes:
                counts["excluded"] += 1
                continue
            if digest in seen:
                counts["duplicate_log"] += 1
                continue
            seen.add(digest)
            if replay.get("formatid", "gen9ou") != "gen9ou":
                counts["other_format"] += 1
                continue
            timestamp = int(replay.get("uploadtime") or 0)
            if timestamp <= 0:
                counts["missing_date"] += 1
                continue
            teams = revealed_teams(replay)
            for team in teams.values():
                for row in team:
                    if not valid_row(row):
                        counts["uninformative_or_invalid_slot"] += 1
                        continue
                    rows.append(canonical_row(row))
                    rosters.append(team[:, 0].copy())
                    ids.append(rid)
                    hashes.append(digest)
                    dates.append(timestamp)
                    ratings.append(int(replay.get("rating") or 0))
            counts["replays"] += 1
            source = replay.get("_fog_source", {})
            sources.add(str(source.get("revision", "local")))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            counts[f"quarantined_{type(error).__name__}"] += 1
        if (index + 1) % 1000 == 0:
            print(f"candidate extraction: {index + 1}/{len(paths)}, {len(rows)} donors", flush=True)
    if not rows:
        raise ValueError("no usable public set donors")
    metadata = {
        "schema_version": 1, "format": "gen9ou", "vocabulary_sha256": vocabulary_digest(),
        "counts": dict(counts), "donor_rows": len(rows), "source_revisions": sorted(sources),
        "excluded_ids": sorted(excluded_ids), "private_truth": False,
        "method": "observed_joint_donors_v1", "timestamp_min": min(dates),
        "timestamp_max": max(dates),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        np.savez_compressed(stream, tokens=np.asarray(rows, dtype=np.int32),
                            rosters=np.asarray(rosters, dtype=np.int32),
                            replay_ids=np.asarray(ids), log_hashes=np.asarray(hashes),
                            timestamps=np.asarray(dates, dtype=np.int64),
                            ratings=np.asarray(ratings, dtype=np.int32),
                            metadata=np.asarray(json.dumps(metadata)))
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class ReplayCandidateBank:
    """Sample an observed, noncontradictory joint donor, then leave holes for the fallback.

    Count actual observations, not the number of incomplete replays compatible with a candidate.
    Prefer donors that add information; group identical sets before a bounded top-k selection.
    Date, rating and teammate overlap are soft weights, not assertions of exact team identity.
    """

    def __init__(self, path: Path, *, top_k: int = 32, half_life_days: float = 180.0):
        if top_k < 1 or half_life_days <= 0:
            raise ValueError("top_k and half_life_days must be positive")
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as bank:
            self.metadata = json.loads(str(bank["metadata"]))
            if (self.metadata["vocabulary_sha256"] != vocabulary_digest()
                    or self.metadata["schema_version"] != 1
                    or self.metadata["format"] != "gen9ou"):
                raise ValueError("candidate bank schema/format/vocabulary mismatch; rebuild it")
            for key in ("tokens", "rosters", "replay_ids", "log_hashes", "timestamps", "ratings"):
                setattr(self, key, bank[key].copy())
        n = len(self.tokens)
        if (self.tokens.shape != (n, SET_N_ATTRS) or self.rosters.shape != (n, 6)
                or any(getattr(self, key).shape != (n,) for key in
                       ("replay_ids", "log_hashes", "timestamps", "ratings"))):
            raise ValueError("candidate bank arrays have inconsistent shapes")
        self.top_k, self.half_life_days = top_k, half_life_days
        self.by_species = {int(s): np.flatnonzero(self.tokens[:, 0] == s)
                           for s in np.unique(self.tokens[:, 0])}
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def propose(self, revealed: np.ndarray, *, rng: np.random.Generator, replay_id: str = "",
                log_hash: str = "", timestamp: int = 0, rating: int = 0) -> tuple[np.ndarray, dict]:
        proposed = np.asarray(revealed, dtype=np.int64).copy()
        report: Counter = Counter()
        roster = proposed[:, 0]
        for slot, row in enumerate(revealed):
            if row[0] < 0:
                continue
            report["slots"] += 1
            indices = self.by_species.get(int(row[0]), np.empty(0, dtype=int))
            # A missing target date cannot establish historical eligibility: fall back explicitly.
            if timestamp <= 0:
                report["missing_target_date"] += 1
                continue
            eligible = ((self.timestamps[indices] < timestamp)
                        & (self.replay_ids[indices] != replay_id)
                        & (self.log_hashes[indices] != log_hash))
            indices = indices[eligible]
            cand = self.tokens[indices]
            known_moves = set(int(m) for m in row[3:7] if m >= 0)
            compatible = np.ones(len(cand), dtype=bool)
            for col in (1, 2, *range(7, SET_N_ATTRS)):
                if row[col] >= 0:
                    compatible &= (cand[:, col] < 0) | (cand[:, col] == row[col])
            # A donor may be partial, but the union must fit four moves. No observed fact is lost.
            additions = np.zeros(len(cand), dtype=int)
            for col in SET_MOVE_COLS:
                additions += (cand[:, col] >= 0) & ~np.isin(cand[:, col], list(known_moves))
            compatible &= additions + len(known_moves) <= 4
            new_scalars = ((cand[:, (1, 2, 14)] >= 0) & (row[[1, 2, 14]] < 0)).sum(axis=1)
            compatible &= additions + new_scalars > 0
            indices = indices[compatible]
            additions, new_scalars = additions[compatible], new_scalars[compatible]
            if not len(indices):
                report["no_compatible_donor"] += 1
                continue
            cand = self.tokens[indices]
            age_days = (timestamp - self.timestamps[indices]) / 86400.0
            weights = np.exp2(-np.minimum(age_days / self.half_life_days, 40))
            if rating > 0:
                weights *= np.exp(-np.abs(self.ratings[indices].astype(float) - rating) / 400)
            overlap = np.isin(self.rosters[indices], roster[roster >= 0]).sum(axis=1) - 1
            weights *= 1 + np.maximum(overlap, 0) / 5
            # Information is rewarded, but partial observed records are not presented as full sets.
            weights *= (1 + additions + new_scalars) ** 2
            unique, inverse = np.unique(cand, axis=0, return_inverse=True)
            mass = np.bincount(inverse, weights=weights, minlength=len(unique))
            top = np.argsort(-mass, kind="stable")[:self.top_k]
            donor = unique[int(rng.choice(top, p=mass[top] / mass[top].sum()))]
            for col in (1, 2, *range(7, SET_N_ATTRS)):
                if proposed[slot, col] < 0:
                    proposed[slot, col] = donor[col]
            extra_moves = [int(m) for m in donor[3:7] if m >= 0 and int(m) not in known_moves]
            for col, move in zip([c for c in SET_MOVE_COLS if row[c] < 0], extra_moves):
                proposed[slot, col] = move
            report["donor_slots"] += 1
            report["donor_fields"] += int(((row < 0) & (proposed[slot] >= 0)).sum())
        return proposed, dict(report)
