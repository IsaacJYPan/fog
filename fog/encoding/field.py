"""Field record [D_field=51] (spec §2.2 + 2026-07 encoding audit).

Layout (DIMS order): weather one-hot(5)+elapsed | terrain one-hot(5)+elapsed | per-side hazards(2x6)
| per-side screens(2x5) | trick room(2) | turn/50 | forced-switch flags(2) | delayed effects(10)
| gravity(2).

Counter semantics (v1, documented): poke-env stores the START turn of timed conditions, not turns
left; we encode ELAPSED turns /8 (the model can learn duration priors — 5 vs 8-turn weather is
item-dependent and technically hidden info anyway). Future Sight / Wish (field_delayed) come from
the BattleTracker's event tracking (poke-env doesn't track them); zeros when no tracker is passed.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from fog.constants.schema import Terrain, Weather
from fog.encoding.dims import DIMS

_NON_ID = re.compile(r"[^a-z0-9]")


def _to_id(name: object) -> str:
    return _NON_ID.sub("", str(name).lower())


def _elapsed(start_turn: object, now: int) -> float:
    try:
        return max(0.0, float(now) - float(start_turn)) / 8.0
    except (TypeError, ValueError):
        return 0.0


# side-condition id -> (offset within a 6-wide hazard block, normalizer)
_HAZARDS = {"stealthrock": (0, 1.0), "spikes": (1, 3.0), "toxicspikes": (2, 2.0), "stickyweb": (3, 1.0)}
# side-condition id -> offset within a 5-wide screen block (slot 4 = shared elapsed counter)
_SCREENS = {"reflect": 0, "lightscreen": 1, "auroraveil": 2, "tailwind": 3}


def encode_field(battle: Any, *, perspective: str = "p1", tracker: Any = None) -> np.ndarray:
    field = np.zeros(DIMS.field_dim, dtype=np.float32)
    if battle is None:
        return field
    turn = int(getattr(battle, "turn", 0) or 0)

    o = 0
    # --- weather one-hot(5) + elapsed ---------------------------------------------------------------
    for w, start in (getattr(battle, "weather", None) or {}).items():
        name = _to_id(getattr(w, "name", w)).upper()
        for member in Weather:
            if member is not Weather.NONE and member.name in name:
                field[o + member.value] = 1.0
                field[o + 5] = _elapsed(start, turn)
    if not field[o : o + 5].any():
        field[o + Weather.NONE.value] = 1.0
    o += DIMS.field_weather

    # --- terrain one-hot(5) + elapsed; trick room + gravity live in battle.fields too ---------------
    trick_room_elapsed = gravity_elapsed = None
    for f, start in (getattr(battle, "fields", None) or {}).items():
        name = _to_id(getattr(f, "name", f)).upper()
        for member in Terrain:
            if member is not Terrain.NONE and name == f"{member.name}TERRAIN":
                field[o + member.value] = 1.0
                field[o + 5] = _elapsed(start, turn)
        if name == "TRICKROOM":
            trick_room_elapsed = _elapsed(start, turn)
        if name == "GRAVITY":
            gravity_elapsed = _elapsed(start, turn)
    if not field[o : o + 5].any():
        field[o + Terrain.NONE.value] = 1.0
    o += DIMS.field_terrain

    # --- per-side hazards (mine then opponent), 6 wide each ------------------------------------------
    sides = (
        getattr(battle, "side_conditions", None) or {},
        getattr(battle, "opponent_side_conditions", None) or {},
    )
    for side in sides:
        for cond, value in side.items():
            name = _to_id(getattr(cond, "name", cond))
            if name in _HAZARDS:
                off, norm = _HAZARDS[name]
                # poke-env: stackable conditions store the stack COUNT, timed ones the start turn
                field[o + off] = min(float(value or 1), norm) / norm if norm > 1 else 1.0
        o += 6

    # --- per-side screens/tailwind, 5 wide each ------------------------------------------------------
    for side in sides:
        for cond, value in side.items():
            name = _to_id(getattr(cond, "name", cond))
            if name in _SCREENS:
                field[o + _SCREENS[name]] = 1.0
                field[o + 4] = max(field[o + 4], _elapsed(value, turn))
        o += 5

    # --- trick room, turn, forced-switch ------------------------------------------------------------
    if trick_room_elapsed is not None:
        field[o] = 1.0
        field[o + 1] = trick_room_elapsed
    o += DIMS.field_trickroom
    field[o] = min(turn, 50) / 50.0
    o += DIMS.field_turn
    force = getattr(battle, "force_switch", False)
    field[o] = 1.0 if (force if isinstance(force, bool) else any(force or [])) else 0.0
    # opponent forced-switch not observable live in v1 -> 0
    o += DIMS.field_forced_switch

    # --- delayed effects (Future Sight / Wish): filled from the BattleTracker when available --------
    if tracker is not None:
        field[o : o + DIMS.field_delayed] = tracker.delayed_block(turn)
    o += DIMS.field_delayed
    if gravity_elapsed is not None:
        field[o] = 1.0
        field[o + 1] = gravity_elapsed
    o += DIMS.field_gravity
    assert o == DIMS.field_dim
    return field
