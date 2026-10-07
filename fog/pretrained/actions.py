"""Semantic native-action to authoritative Showdown request mapping."""

from __future__ import annotations

import re


def identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def request_choice(choice: str, request: dict) -> str:
    """Canonicalize external player move names to the request's numeric slots."""
    choice = choice.removeprefix("/choose ").removeprefix("/").strip()
    parts = choice.split()
    if parts and parts[0] == "move" and len(parts) >= 2 and not parts[1].isdigit():
        offered = request.get("active", [{}])[0].get("moves", [])
        matches = [
            slot
            for slot, move in enumerate(offered)
            if identifier(move.get("id") or move["move"]) == identifier(parts[1])
        ]
        if len(matches) != 1:
            raise ValueError("External move name does not match exactly one request slot")
        parts[1] = str(matches[0] + 1)
    return " ".join(parts)


def native_choices(state, request: dict) -> dict[int, str]:
    """Kakuna's 13 alphabetic actions, including Tera, switches and Revival Blessing.

    Use the complete active moveset for native ordering, and the authoritative
    request for physical slots and legality. Never renumber disabled moves.
    Names alone cannot distinguish duplicate switch targets; ambiguity is fatal.
    """
    if request.get("wait") or request.get("teamPreview"):
        return {}
    forced = any(request.get("forceSwitch", [False]))
    choices = {}
    active = (request.get("active") or [{}])[0]
    offered = active.get("moves", [])
    move_order = sorted(state.player_active_pokemon.moves, key=lambda m: identifier(m.name))
    offered_ids = [identifier(m.get("id") or m["move"]) for m in offered]
    if not forced:
        if offered_ids == ["recharge"] or offered_ids == ["struggle"]:
            # Native replay data represents these automatic choices at index zero.
            choices[0] = "move 1"
        else:
            for native_index, move in enumerate(move_order[:4]):
                matches = [i for i, name in enumerate(offered_ids) if name == identifier(move.name)]
                if len(matches) > 1:
                    raise ValueError("Ambiguous move in authoritative request")
                if not matches:
                    continue
                slot = matches[0]
                if offered[slot].get("disabled") or offered[slot].get("pp", 1) <= 0:
                    continue
                choices[native_index] = f"move {slot + 1}"
                if active.get("canTerastallize"):
                    choices[native_index + 9] = f"move {slot + 1} terastallize"
    party = request.get("side", {}).get("pokemon", [])
    reviving = any(p.get("reviving") for p in party)
    if active.get("trapped") and not forced:
        return choices
    for native_index, pokemon in enumerate(
        sorted(state.available_switches, key=lambda p: identifier(p.name))[:5]
    ):
        candidates = []
        for slot, member in enumerate(party):
            fainted = member.get("condition", "").endswith("fnt")
            if member.get("active") or (not fainted if reviving else fainted):
                continue
            species = identifier(member["details"].split(",")[0])
            # Exact native current/base species plus the player's private nickname.
            names = {identifier(pokemon.name), identifier(pokemon.base_species)}
            if species in names:
                candidates.append(slot)
        if len(candidates) != 1:
            raise ValueError(
                f"Native switch target {pokemon.name!r} maps to {len(candidates)} request slots"
            )
        choices[native_index + 4] = f"switch {candidates[0] + 1}"
    if not choices:
        raise ValueError("No validated native action for this choice request")
    return choices
