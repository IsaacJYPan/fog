"""History records [T, D_turn] (spec §2.3).

One record per completed turn: (compressed public snapshot as of the *start* of the turn, both
players' observed choices, resolution events). The start-of-turn snapshot is what preserves context at
decision time — the requirement from the design discussion. T is capped at 60 and padded/truncated
from the FRONT (keep the most recent turns).

Each turn record concatenates:
  * snapshot   [hist_snapshot]  both actives' species/hp/status/boost sums + ~30 field-hash features
  * my_action  [hist_my_action] 14-way one-hot
  * opp_action [hist_opp_action] observed category: which move / switched-to species / tera flag /
                                 <UNSEEN> if pre-empted (pre-emption itself is informative)
  * resolution [hist_resolution] dmg dealt/taken (HP frac), crit, effectiveness bucket, miss,
                                 items/abilities revealed (indices), faints, who-moved-first (speed
                                 evidence), exact damage frac x255 quantized (enables damage-calc
                                 inference on spreads/items)

Extraction runs off poke-env's `battle.observations[turn].events` via the events.py mini-parser.
Start-of-turn snapshots and the acting player's switch slots require the incremental
``BattleTracker`` used by live collection. A caller that omits the tracker receives only facts that
can be reconstructed from the censored protocol and must treat the missing fields as unavailable.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np

from fog.encoding.dims import DIMS

_NON_ID = re.compile(r"[^a-z0-9]")


def observations_by_turn(battle: Any) -> dict[int, Any]:
    """Return per-turn protocol events for native, engine, and ordinary poke-env battles.

    Some poke-env releases retain only ``_replay_data`` and expose no ``observations`` mapping.
    Build an incremental compatibility view from that already-censored player log. The cache is
    attached to the battle and consumes only newly appended lines, so calling this every decision
    is linear in new protocol traffic rather than replay length.
    """
    native = getattr(battle, "observations", None)
    if native:
        return native
    replay = getattr(battle, "_replay_data", None)
    if replay is None:
        return native or {}

    cache = getattr(battle, "_fog_observation_cache", None)
    if cache is None:
        cache = {}
        battle._fog_observation_cache = cache
        battle._fog_replay_cursor = 0
        battle._fog_replay_turn = 0
    cursor = int(getattr(battle, "_fog_replay_cursor", 0))
    current = int(getattr(battle, "_fog_replay_turn", 0))
    if cursor > len(replay):  # defensive reset if an upstream parser replaces its replay buffer
        cache.clear()
        cursor = 0
        current = 0
    for event in replay[cursor:]:
        if len(event) > 2 and event[1] == "turn":
            try:
                current = int(event[2])
            except (TypeError, ValueError):
                continue
            cache.setdefault(current, SimpleNamespace(events=[]))
        else:
            cache.setdefault(current, SimpleNamespace(events=[])).events.append(event)
    battle._fog_replay_cursor = len(replay)
    battle._fog_replay_turn = current
    return cache


def _my_movesets(battle: Any) -> dict[str, dict[str, int]]:
    """{mon_key: {move_id: slot 0-3}} for retro my-action inference (events.parse_turn_events)."""
    out: dict[str, dict[str, int]] = {}
    for key, mon in (getattr(battle, "team", None) or {}).items():
        moves = {}
        for slot, move_id in enumerate(list(getattr(mon, "moves", None) or {})[:4]):
            moves[_NON_ID.sub("", str(move_id).lower())] = slot
        out[str(key)] = moves  # battle.team keys are already 'p1: Name'
    return out


def _species_by_key(battle: Any, tables) -> dict[str, int]:
    from fog.encoding.events import submitted_species_id

    out: dict[str, int] = {}
    for group in ("team", "opponent_team"):
        for key, mon in (getattr(battle, group, None) or {}).items():
            species = getattr(mon, "species", None)
            if species:
                out[str(key)] = tables.species_index(submitted_species_id(species))
    return out


def encode_history(
    battle: Any, tables=None, *, perspective: str = "p1", tracker: Any = None
) -> tuple[np.ndarray, int]:
    """Return (history[T, D_turn] float32, history_len) for completed turns, most recent LAST,
    truncated to the last `DIMS.history_cap` turns (front-truncation, §2.3). battle=None or no
    observations -> single zero row (shape-test path).

    `tracker` (BattleTracker, optional) supplies the per-turn start-of-turn snapshots, my recorded
    action indices, and hp seeds — the retro-fidelity fields that raw event parsing can't recover."""
    obs_map = observations_by_turn(battle)
    current = int(getattr(battle, "turn", 0) or 0)
    turns = sorted(t for t in obs_map if 1 <= t < current)
    if battle is None or tables is None or not turns:
        return np.zeros((1, DIMS.turn_dim), dtype=np.float32), 0

    from fog.encoding.events import parse_turn_events

    role = getattr(battle, "player_role", None) or perspective
    movesets = _my_movesets(battle)
    turns = turns[-DIMS.history_cap :]
    records = np.stack(
        [
            parse_turn_events(
                getattr(obs_map[t], "events", None) or [],
                role,
                tables,
                my_movesets=movesets,
                my_action_idx=(tracker.my_actions.get(t) if tracker is not None else None),
                snapshot=(tracker.snapshots.get(t) if tracker is not None else None),
                hp_seed=(tracker.hp_at_turn_start.get(t) if tracker is not None else None),
            )
            for t in turns
        ]
    ).astype(np.float32)
    return records, len(turns)


def encode_history_sidecars(
    battle: Any, tables=None, *, perspective: str = "p1", tracker: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return categorical ids and per-stat start state aligned one-for-one with ``encode_history``.

    The sidecars are optional model inputs so old replay shards and checkpoints remain usable.
    Real rows follow completed-turn protocol order; the zero row is used only for a length-zero
    history and is masked by ``history_len``.
    """
    from fog.encoding.events import N_HISTORY_ID_FIELDS, parse_turn_event_ids

    obs_map = observations_by_turn(battle)
    current = int(getattr(battle, "turn", 0) or 0)
    turns = sorted(t for t in obs_map if 1 <= t < current)[-DIMS.history_cap :]
    if battle is None or tables is None or not turns:
        return (
            np.zeros((1, N_HISTORY_ID_FIELDS), dtype=np.int64),
            np.zeros((1, DIMS.hist_state), dtype=np.float32),
        )

    role = getattr(battle, "player_role", None) or perspective
    species = _species_by_key(battle, tables)
    ids = []
    states = []
    for turn in turns:
        snapshot_ids = tracker.snapshot_ids.get(turn) if tracker is not None else None
        if snapshot_ids is None and tracker is not None and turn in tracker.snapshots:
            # Compatibility with trackers recorded before categorical sidecars existed.
            legacy = np.asarray(tracker.snapshots[turn])
            recovered = np.rint(legacy[:2] * 1000.0).astype(np.int64)
            recovered[(recovered < 0) | (recovered >= tables.n_species)] = tables.n_species
            snapshot_ids = recovered
        ids.append(
            parse_turn_event_ids(
                getattr(obs_map[turn], "events", None) or [],
                role,
                tables,
                snapshot_ids=snapshot_ids,
                species_by_key=species,
            )
        )
        state = tracker.snapshot_state.get(turn) if tracker is not None else None
        states.append(
            np.asarray(state, dtype=np.float32)
            if state is not None
            else np.zeros(DIMS.hist_state, dtype=np.float32)
        )
    return np.stack(ids).astype(np.int64), np.stack(states).astype(np.float32)


def pad_or_truncate_front(records: np.ndarray, cap: int = DIMS.history_cap) -> np.ndarray:
    """Keep the most recent `cap` turns, front-padding with zeros. Real and used by tests."""
    t, d = records.shape
    if t == cap:
        return records
    if t > cap:
        return records[t - cap :]
    out = np.zeros((cap, d), dtype=records.dtype)
    out[cap - t :] = records
    return out
