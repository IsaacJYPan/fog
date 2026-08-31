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

from fog.constants.schema import ACTIONS, FORCED_SLOT, N_ACTIONS, ActionType

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

    # Showdown can offer the synthetic Struggle move while voluntary switches remain legal.
    # Struggle has no native moveset slot, so represent it with the project's default/forced slot
    # instead of silently making a legal teacher choice impossible to label.
    if any(move_key(move) == "struggle" for move in _as_list(getattr(battle, "available_moves", None))):
        mask[FORCED_SLOT] = True
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


def _request_switch_position(battle: Any, target: Any) -> int | None:
    """Return Showdown's current one-based party position for ``target``.

    Poke-env normally formats switches by nickname. Showdown first tries to parse that text as an
    integer, so a valid ladder nickname such as ``"2 girls 1 cup"`` is interpreted as party slot
    two instead of as a name. Names can also collide with another party member's species. The
    private choice request is authoritative and already contains the exact current party order;
    match its ``ident`` back to the same Pokemon object and send the unambiguous numeric slot.
    """
    request = getattr(battle, "last_request", None)
    if callable(request):
        request = request()
    if not isinstance(request, dict):
        return None
    side = request.get("side")
    if not isinstance(side, dict):
        return None
    requested_team = side.get("pokemon")
    team = getattr(battle, "team", None)
    if not isinstance(requested_team, list) or not isinstance(team, dict):
        return None

    # Every poke-env Pokemon updated from this request retains its exact request row. Prefer that
    # object-local ident: during Illusion reconciliation the battle.team alias for an ident can
    # temporarily point at a different object even though available_switches contains the correct
    # target. Species/name matching cannot disambiguate that case (and a numeric nickname must
    # never be allowed to fall through to Showdown's string switch parser).
    target_request = getattr(target, "_last_request", None)
    target_ident = target_request.get("ident") if isinstance(target_request, dict) else None
    if target_ident is not None:
        for position, requested in enumerate(requested_team, start=1):
            if isinstance(requested, dict) and requested.get("ident") == target_ident:
                return position

    # was_illusioned() can temporarily clear Pokemon._last_request. The Pokemon's current name is
    # still the request nickname, so match it directly against the authoritative ident suffix.
    target_name = getattr(target, "name", None)
    if target_name:
        normalized_name = _normalize_id(target_name)
        name_matches = []
        for position, requested in enumerate(requested_team, start=1):
            if not isinstance(requested, dict):
                continue
            ident = str(requested.get("ident", ""))
            request_name = ident.split(":", 1)[-1].strip()
            if normalized_name and _normalize_id(request_name) == normalized_name:
                name_matches.append(position)
        if len(name_matches) == 1:
            return name_matches[0]

    for position, requested in enumerate(requested_team, start=1):
        if not isinstance(requested, dict):
            continue
        ident = requested.get("ident")
        # poke-env can rebuild ``available_switches`` from a newer request, so the offered
        # target is not guaranteed to be the same Python object stored in ``battle.team``.
        # Match semantically against the authoritative ident instead of falling back to a
        # nickname (numeric-leading nicknames are parsed by Showdown as slot numbers).
        if ident is not None and same_pokemon(team.get(ident), target):
            return position

    # The request and battle.team are both initialized in authoritative party order. This remains
    # stable even when Illusion temporarily changes which object an ident aliases.
    for position, mon in enumerate(team.values(), start=1):
        if mon is target and position <= len(requested_team):
            return position
    return None


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
        request_position = _request_switch_position(battle, target)
        if request_position is not None:
            return _create_order(player, f"/choose switch {request_position}")
        target_name = str(getattr(target, "name", "") or "")
        if target_name and target_name.lstrip().startswith(tuple("0123456789")):
            raise AssertionError(
                "numeric-leading switch nickname could not be mapped to an authoritative "
                f"request slot: {target_name!r}"
            )
        return _create_order(player, target)

    raise AssertionError(f"unhandled action kind: {kind!r}")


def map_order_to_action(battle: Any, order: Any) -> int:
    """Convert a poke-env ``BattleOrder`` into the matching project action index.

    This is the supervised-data inverse of :func:`map_action_to_order`.  It is intentionally
    strict: an order that cannot be matched to the current authoritative request must not become
    a mislabeled policy row.
    """
    target = getattr(order, "order", None)
    terastallize = bool(getattr(order, "terastallize", False))
    if move_key(target) == "struggle" and action_legality_mask(battle)[FORCED_SLOT]:
        return FORCED_SLOT

    for slot, move in available_move_by_slot(battle).items():
        if target is move or move_key(target) == move_key(move):
            return slot + (4 if terastallize else 0)

    for slot, mon in available_switch_by_slot(battle).items():
        if target is mon or same_pokemon(target, mon):
            return 8 + slot

    legal = [index for index, allowed in enumerate(action_legality_mask(battle)) if allowed]
    if legal == [FORCED_SLOT]:
        return FORCED_SLOT
    if len(legal) == 1 and target is None:
        return legal[0]
    raise ValueError(f"scripted order does not match the current request: {order!r}")
