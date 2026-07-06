"""Action-space helpers shared by live players, encoders, and tests.

The project action space is fixed at 14 entries:
0-3 moves, 4-7 tera moves, 8-12 switches in team-order-minus-active order, 13 forced/default.
These helpers deliberately use duck typing so unit tests can cover the mapping without importing
poke-env, and runtime code can still pass real poke-env battle/player objects.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from fog.constants.schema import ACTIONS, ActionType, FORCED_SLOT, N_ACTIONS

_NON_ID = re.compile(r"[^a-z0-9]")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_id(value: Any) -> str:
    return _NON_ID.sub("", str(value).lower())


def move_key(move: Any) -> str:
    """Stable comparison key for a poke-env Move object or a lightweight test double."""
    for attr in ("id", "move_id", "name"):
        val = getattr(move, attr, None)
        if val:
            return _normalize_id(val)
    return _normalize_id(move)


def pokemon_key(mon: Any) -> str:
    """Stable comparison key for a poke-env Pokemon object or a lightweight test double."""
    for attr in ("identifier", "ident", "id", "species", "species_id", "name"):
        val = getattr(mon, attr, None)
        if val:
            return _normalize_id(val)
    return _normalize_id(mon)


def same_pokemon(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    ka, kb = pokemon_key(a), pokemon_key(b)
    return bool(ka and kb and ka == kb)


def active_move_slots(battle: Any) -> list[Any]:
    active = getattr(battle, "active_pokemon", None)
    return _as_list(getattr(active, "moves", None))[:4]


def available_move_by_slot(battle: Any) -> dict[int, Any]:
    """Map the active mon's move-slot index to the corresponding currently legal move object."""
    available = {move_key(m): m for m in _as_list(getattr(battle, "available_moves", None))}
    out: dict[int, Any] = {}
    for slot, move in enumerate(active_move_slots(battle)):
        key = move_key(move)
        if key in available:
            out[slot] = available[key]
    return out


def team_order(battle: Any) -> list[Any]:
    return _as_list(getattr(battle, "team", None))


def opponent_team_order(battle: Any) -> list[Any]:
    return _as_list(getattr(battle, "opponent_team", None))


def team_order_minus_active(battle: Any) -> list[Any]:
    active = getattr(battle, "active_pokemon", None)
    return [mon for mon in team_order(battle) if not same_pokemon(mon, active)]


def available_switch_by_slot(battle: Any) -> dict[int, Any]:
    """Map project switch slots 0-4 to currently legal switch targets."""
    available = _as_list(getattr(battle, "available_switches", None))
    out: dict[int, Any] = {}
    for slot, target in enumerate(team_order_minus_active(battle)[:5]):
        for offered in available:
            if same_pokemon(offered, target):
                out[slot] = offered
                break
    return out


def bool_attr(obj: Any, name: str, default: bool = False) -> bool:
    val = getattr(obj, name, default)
    if callable(val):
        val = val()
    return bool(val)


def action_legality_mask(battle: Any) -> list[bool]:
    """Compute the 14-way legality mask from a live battle/request-like object."""
    mask = [False] * N_ACTIONS

    for slot in available_move_by_slot(battle):
        if slot < 4:
            mask[slot] = True
            if bool_attr(battle, "can_tera", False):
                mask[4 + slot] = True

    for slot in available_switch_by_slot(battle):
        if slot < 5:
            mask[8 + slot] = True

    if not any(mask[:FORCED_SLOT]):
        mask[FORCED_SLOT] = True
    return mask


def active_entity_indices(battle: Any) -> tuple[int, int]:
    """Return entity-token indices for active mine/opponent, defaulting to slots 0 and 6."""
    active = getattr(battle, "active_pokemon", None)
    opp_active = getattr(battle, "opponent_active_pokemon", None)

    mine_idx = 0
    for i, mon in enumerate(team_order(battle)[:6]):
        if same_pokemon(mon, active):
            mine_idx = i
            break

    opp_idx = 6
    for i, mon in enumerate(opponent_team_order(battle)[:6]):
        if same_pokemon(mon, opp_active):
            opp_idx = 6 + i
            break

    return mine_idx, opp_idx


def _create_order(player: Any, target: Any, *, terastallize: bool = False) -> Any:
    if not terastallize:
        return player.create_order(target)
    return player.create_order(target, terastallize=True)


def map_action_to_order(battle: Any, action_idx: int, player: Any) -> Any:
    """Convert a project action index into a poke-env BattleOrder.

    If this raises AssertionError during live play, the legality mask and the battle request disagree.
    That is intentionally loud because a silent fallback would corrupt training labels.
    """
    if not 0 <= action_idx < N_ACTIONS:
        raise ValueError(f"action index out of range: {action_idx}")

    kind, sub = ACTIONS[action_idx]
    if kind is ActionType.FORCED:
        return player.choose_default_move()

    if kind in (ActionType.MOVE, ActionType.MOVE_TERA):
        move = available_move_by_slot(battle).get(sub)
        assert move is not None, f"move slot {sub} is not currently available"
        return _create_order(player, move, terastallize=(kind is ActionType.MOVE_TERA))

    if kind is ActionType.SWITCH:
        target = available_switch_by_slot(battle).get(sub)
        assert target is not None, f"switch slot {sub} is not currently available"
        return _create_order(player, target)

    raise AssertionError(f"unhandled action kind: {kind!r}")
