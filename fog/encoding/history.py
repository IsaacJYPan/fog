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
Retro fidelity note (v1): the start-of-turn snapshot block and my switch-action slots need
incremental live tracking to fill retroactively; the action/resolution signals (which move, who
moved first, faints, reveals, damage where derivable) are real. See KNOWN_ISSUES.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from fog.encoding.dims import DIMS

_NON_ID = re.compile(r"[^a-z0-9]")


def _my_movesets(battle: Any) -> dict[str, dict[str, int]]:
    """{mon_key: {move_id: slot 0-3}} for retro my-action inference (events.parse_turn_events)."""
    out: dict[str, dict[str, int]] = {}
    role = getattr(battle, "player_role", None) or "p1"
    for key, mon in (getattr(battle, "team", None) or {}).items():
        moves = {}
        for slot, move_id in enumerate(list(getattr(mon, "moves", None) or {})[:4]):
            moves[_NON_ID.sub("", str(move_id).lower())] = slot
        # battle.team keys are already 'p1: Name'; keep as-is, fall back to role-prefixed species
        out[str(key)] = moves
    return out


def encode_history(
    battle: Any, tables=None, *, perspective: str = "p1", tracker: Any = None
) -> tuple[np.ndarray, int]:
    """Return (history[T, D_turn] float32, history_len) for completed turns, most recent LAST,
    truncated to the last `DIMS.history_cap` turns (front-truncation, §2.3). battle=None or no
    observations -> single zero row (shape-test path).

    `tracker` (BattleTracker, optional) supplies the per-turn start-of-turn snapshots, my recorded
    action indices, and hp seeds — the retro-fidelity fields that raw event parsing can't recover."""
    obs_map = getattr(battle, "observations", None) or {}
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
