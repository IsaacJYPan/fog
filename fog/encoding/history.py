"""History records [T, D_turn].

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


# ------------------------------------------------------------------------------ per-turn cache
# Completed turns never change, so their parsed records are memoised on the per-battle tracker.
# Every cache entry stores a fingerprint of *all* inputs the parser reads (the turn's event list
# content, observer role, table identity, and the tracker/battle-derived arguments). A lookup only
# hits when the current inputs compare equal to the stored fingerprint, so a turn whose events are
# still growing, a later-revealed opponent species, or an overwritten action recording simply
# recomputes. Cached records are read-only; callers always receive freshly stacked arrays.
_CACHE_ATTR = "_history_record_cache"


def _array_fp(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    return (arr.dtype.str, arr.shape, arr.tobytes())


def _event_matches(event: Any, stored: list) -> bool:
    return (event if event.__class__ is list else list(event)) == stored


def _events_match(events: list, stored: list) -> bool:
    return len(events) == len(stored) and all(map(_event_matches, events, stored))


def _cache_for(tracker: Any) -> dict | None:
    if tracker is None:
        return None
    cache = getattr(tracker, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(tracker, _CACHE_ATTR, cache)
        except AttributeError:
            return None
    return cache


def _cached_record(cache, key, events, tables, inputs, compute, seen=None) -> np.ndarray:
    """Return the cached record for ``key`` when every input matches, else compute and store."""
    if cache is None:
        return compute()
    hit = cache.get(key)
    if (
        hit is not None
        and hit[1] is tables
        and hit[2] == inputs
        # A turn's event list only ever grows by appending, so the same list object at the same
        # length holds the same events; anything else is compared event by event.
        and ((hit[4] is events and hit[5] == len(events)) or _events_match(events, hit[0]))
    ):
        if seen is not None and hit[6] is None:
            cache[key] = (*hit[:6], seen)
        return hit[3]
    record = compute()
    frozen = np.array(record, copy=True)
    frozen.setflags(write=False)
    cache[key] = ([list(ev) for ev in events], tables, inputs, frozen, events, len(events), seen)
    return frozen


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
    battle: Any,
    tables=None,
    *,
    perspective: str = "p1",
    tracker: Any = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, int]:
    """Return (history[T, D_turn] float32, history_len) for completed turns, most recent LAST,
    truncated to the last `DIMS.history_cap` turns (front-truncation). battle=None or no
    observations -> single zero row (shape-test path).

    `tracker` (BattleTracker, optional) supplies the per-turn start-of-turn snapshots, my recorded
    action indices, and hp seeds — the retro-fidelity fields that raw event parsing can't recover.
    It also hosts the per-turn record cache (``use_cache=False`` bypasses it)."""
    obs_map = observations_by_turn(battle)
    current = int(getattr(battle, "turn", 0) or 0)
    turns = sorted(t for t in obs_map if 1 <= t < current)
    if battle is None or tables is None or not turns:
        return np.zeros((1, DIMS.turn_dim), dtype=np.float32), 0

    from fog.encoding.events import parse_turn_events

    role = getattr(battle, "player_role", None) or perspective
    movesets = _my_movesets(battle)
    turns = turns[-DIMS.history_cap :]
    cache = _cache_for(tracker) if use_cache else None
    rows = []
    for t in turns:
        events = getattr(obs_map[t], "events", None) or []
        action = tracker.my_actions.get(t) if tracker is not None else None
        snapshot = tracker.snapshots.get(t) if tracker is not None else None
        hp_seed = tracker.hp_at_turn_start.get(t) if tracker is not None else None

        def compute(events=events, action=action, snapshot=snapshot, hp_seed=hp_seed):
            return parse_turn_events(
                events,
                role,
                tables,
                my_movesets=movesets,
                my_action_idx=action,
                snapshot=snapshot,
                hp_seed=hp_seed,
            )

        if cache is None:
            rows.append(compute())
            continue
        key = ("rec", role, t)
        if action is not None:
            # The tracker replaces a turn's snapshot and hit-point seed with new objects whenever
            # it records that turn again and never edits them in place, so the same objects with
            # the same action and event count are the inputs this record was computed from.
            hit = cache.get(key)
            if hit is not None and hit[1] is tables:
                seen = hit[6]
                if (
                    seen is not None
                    and seen[0] is events
                    and seen[1] == len(events)
                    and seen[2] == action
                    and seen[3] is snapshot
                    and seen[4] is hp_seed
                ):
                    rows.append(hit[3])
                    continue
        inputs = (
            action,
            # movesets are read only for retro inference when no action was recorded
            movesets if action is None else None,
            _array_fp(snapshot),
            dict(hp_seed) if hp_seed is not None else None,
        )
        rows.append(_cached_record(
            cache, key, events, tables, inputs, compute,
            seen=(events, len(events), action, snapshot, hp_seed) if action is not None else None,
        ))
    records = np.stack(rows).astype(np.float32)
    return records, len(turns)


def encode_history_sidecars(
    battle: Any,
    tables=None,
    *,
    perspective: str = "p1",
    tracker: Any = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return categorical ids and per-stat start state aligned one-for-one with ``encode_history``.

    The sidecars are optional model inputs so old replay shards and checkpoints remain usable.
    Real rows follow completed-turn protocol order; the zero row is used only for a length-zero
    history and is masked by ``history_len``. Id rows are memoised on ``tracker`` unless
    ``use_cache=False``; the per-stat state rows are cheap copies and are never cached.
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
    cache = _cache_for(tracker) if use_cache else None
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
        events = getattr(obs_map[turn], "events", None) or []

        def compute(events=events, snapshot_ids=snapshot_ids):
            return parse_turn_event_ids(
                events,
                role,
                tables,
                snapshot_ids=snapshot_ids,
                species_by_key=species,
            )

        if cache is None:
            ids.append(compute())
        else:
            key = ("ids", role, turn)
            hit = cache.get(key)
            seen = hit[6] if hit is not None and hit[1] is tables else None
            if (
                seen is not None
                and seen[0] is events
                and seen[1] == len(events)
                and seen[2] is snapshot_ids
                and hit[2][1] == species
            ):
                ids.append(hit[3])
            else:
                inputs = (_array_fp(snapshot_ids), species)
                ids.append(_cached_record(
                    cache, key, events, tables, inputs, compute,
                    seen=(events, len(events), snapshot_ids) if snapshot_ids is not None else None,
                ))
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
