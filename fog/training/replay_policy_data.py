"""Private-side reconstruction for policy learning from public Showdown replays.

A replay contains the public protocol, not either player's submitted team sheet.  The acting
player did know its own sheet, however, so replacing those fields with zeros creates a state that
no real player ever saw.  This module completes each player's eventually revealed facts with
usage-prior samples.  Nothing from the opponent's future reveals is copied into the acting
player's observation.

Every submitted-set field carries a provenance code:

* ``UNKNOWN``  - no Pokemon occupies this slot;
* ``INFERRED`` - sampled from a pinned metagame prior;
* ``EXACT``    - species or a private fact eventually made public by this same player.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from fog.constants.schema import (
    NATURE_INDEX,
    SET_EV_COLS,
    SET_MOVE_COLS,
    SET_N_ATTRS,
    SET_NATURE_COL,
    SET_TERA_COL,
    TYPE_INDEX,
)
from fog.constants.tables import GENERATED_DIR, TABLES

UNKNOWN = np.uint8(0)
INFERRED = np.uint8(1)
EXACT = np.uint8(2)


@lru_cache(maxsize=1)
def replay_set_priors() -> dict:
    with (GENERATED_DIR / "priors.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def _weighted_key(values: dict[str, float], rng: np.random.Generator) -> str | None:
    candidates = [(str(key), max(0.0, float(weight))) for key, weight in values.items()]
    candidates = [(key, weight) for key, weight in candidates if key and weight > 0]
    if not candidates:
        return None
    keys, weights = zip(*candidates)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    return str(rng.choice(keys, p=probabilities))


def _sample_spread(species_id: str, rng: np.random.Generator,
                   spread_priors: dict | None = None) -> tuple[int, list[int]]:
    source = TABLES.spreads.get("species", {}) if spread_priors is None else spread_priors
    spreads = list(source.get(species_id, ()))
    if not spreads:
        return NATURE_INDEX["serious"], [0] * 6
    weights = np.asarray(
        [max(0.0, float(spread.get("weight", 0.0))) for spread in spreads],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(spreads), dtype=np.float64)
    weights /= weights.sum()
    spread = spreads[int(rng.choice(len(spreads), p=weights))]
    nature = NATURE_INDEX.get(str(spread.get("nature", "serious")).lower(), NATURE_INDEX["serious"])
    evs = [min(63, max(0, int(value) // 4)) for value in spread.get("evs", [0] * 6)]
    return nature, (evs + [0] * 6)[:6]


def complete_private_team(
    revealed: np.ndarray,
    *,
    seed: int,
    candidate_bank=None,
    replay_id: str = "",
    log_hash: str = "",
    timestamp: int = 0,
    rating: int = 0,
    diagnostics: dict | None = None,
    usage_priors: dict | None = None,
    spread_priors: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one plausible full own sheet and a matching ``[6,15]`` provenance grid."""
    exact = np.asarray(revealed, dtype=np.int64)
    if exact.shape != (6, SET_N_ATTRS):
        raise ValueError(f"revealed private set must have shape (6,{SET_N_ATTRS}); got {exact.shape}")
    rng = np.random.default_rng(seed)
    priors = replay_set_priors() if usage_priors is None else usage_priors
    completed = exact.copy()
    provenance = np.where(exact >= 0, EXACT, UNKNOWN).astype(np.uint8)
    if candidate_bank is not None:
        completed, report = candidate_bank.propose(
            exact, rng=rng, replay_id=replay_id, log_hash=log_hash,
            timestamp=timestamp, rating=rating,
        )
        provenance[(exact < 0) & (completed >= 0)] = INFERRED
        if diagnostics is not None:
            for key, value in report.items():
                diagnostics[key] = diagnostics.get(key, 0) + value

    for slot in range(6):
        species = int(completed[slot, 0])
        if species < 0 or species >= TABLES.n_species:
            continue
        species_id = str(TABLES.species["ids"][species])
        # Separate field streams keep fallback draws paired across imputation methods. A donor
        # supplying an item must not accidentally change the independently sampled EV spread.
        rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 1]))

        if completed[slot, 1] < 0:
            item_id = _weighted_key(priors.get("item_priors", {}).get(species_id, {}), rng)
            if item_id == "nothing":
                item_id = "<NO_ITEM>"
            completed[slot, 1] = TABLES.item_index(item_id or "<NO_ITEM>")
            provenance[slot, 1] = INFERRED

        rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 2]))
        if completed[slot, 2] < 0:
            ability_id = _weighted_key(
                priors.get("ability_priors", {}).get(species_id, {}), rng
            )
            ability = TABLES.ability_index(ability_id or "", default=None)
            if ability is None:
                legal = np.flatnonzero(TABLES.legal_abilities_mask(species))
                ability = int(rng.choice(legal)) if len(legal) else 0
            completed[slot, 2] = ability
            provenance[slot, 2] = INFERRED

        rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 3]))
        known_moves = [
            int(completed[slot, column])
            for column in SET_MOVE_COLS
            if int(completed[slot, column]) >= 0
        ]
        original_moves = set(known_moves)
        move_prior = dict(priors.get("move_priors", {}).get(species_id, {}))
        while len(known_moves) < 4:
            candidates = {
                move_id: weight
                for move_id, weight in move_prior.items()
                if TABLES.move_index(move_id, default=None) not in known_moves
                and TABLES.move_index(move_id, default=None) is not None
                and bool(TABLES.learnset_mask(species)[TABLES.move_index(move_id)])
            }
            move_id = _weighted_key(candidates, rng)
            move = TABLES.move_index(move_id or "", default=None)
            if move is None:
                legal = [
                    int(index)
                    for index in np.flatnonzero(TABLES.learnset_mask(species))
                    if int(index) not in known_moves
                ]
                if not legal:
                    break
                move = int(rng.choice(legal))
            known_moves.append(move)
            move_prior.pop(str(TABLES.moves["ids"][move]), None)
        new_moves = [move for move in known_moves if move not in original_moves]
        empty_move_columns = [column for column in SET_MOVE_COLS if completed[slot, column] < 0]
        for column, move in zip(empty_move_columns, new_moves):
            completed[slot, column] = move
            provenance[slot, column] = INFERRED

        rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 7]))
        nature, ev_units = _sample_spread(species_id, rng, spread_priors)
        if completed[slot, SET_NATURE_COL] < 0:
            completed[slot, SET_NATURE_COL] = nature
            provenance[slot, SET_NATURE_COL] = INFERRED
        for column, value in zip(SET_EV_COLS, ev_units):
            if completed[slot, column] < 0:
                completed[slot, column] = value
                provenance[slot, column] = INFERRED

        rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 14]))
        if completed[slot, SET_TERA_COL] < 0:
            tera_id = _weighted_key(priors.get("tera_priors", {}).get(species_id, {}), rng)
            tera_index = {**TYPE_INDEX, "stellar": 18}
            completed[slot, SET_TERA_COL] = tera_index.get(tera_id or "", int(rng.integers(19)))
            provenance[slot, SET_TERA_COL] = INFERRED

    return completed, provenance


def apply_private_set(mon, tokens: np.ndarray) -> None:
    """Install one sampled submitted set on an own-side poke-env Pokemon object."""
    from poke_env.battle import PokemonType

    row = np.asarray(tokens, dtype=np.int64)
    item_id = str(TABLES.items["ids"][int(row[1])]) if row[1] >= 0 else ""
    mon._item = None if item_id in ("", "<NO_ITEM>") else item_id
    mon._ability = str(TABLES.abilities["ids"][int(row[2])]) if row[2] >= 0 else None
    # Preview Pokemon have no private request moves.  Populate all four in stable sampled order.
    for column in SET_MOVE_COLS:
        if row[column] >= 0:
            mon._add_move(str(TABLES.moves["ids"][int(row[column])]))
    if row[SET_NATURE_COL] >= 0:
        from fog.constants.schema import NATURES

        mon._nature = NATURES[int(row[SET_NATURE_COL])]
    mon._evs = [int(row[column]) * 4 for column in SET_EV_COLS]
    if row[SET_TERA_COL] >= 0:
        type_id = (*TYPE_INDEX, "stellar")[int(row[SET_TERA_COL])]
        mon._terastallized_type = PokemonType.from_name(type_id)


def stable_replay_seed(replay_id: str, role: str, imputation: int, seed: int) -> int:
    import hashlib

    payload = f"{seed}:{replay_id}:{role}:{imputation}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


__all__ = [
    "EXACT",
    "INFERRED",
    "UNKNOWN",
    "apply_private_set",
    "complete_private_team",
    "stable_replay_seed",
]
