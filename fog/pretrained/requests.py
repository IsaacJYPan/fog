"""Adapt native battle identity and own reserves to authoritative Gen9OU requests."""

from copy import deepcopy
import hashlib

from fog.pretrained.actions import identifier


def gen9ou_battle_tag(identity: str) -> str:
    """Keep transport identities out of Metamon's format token.

    UniversalState.from_Battle reads battle_tag.split('-')[1]. Local collection
    and evaluation identities need not follow Showdown's battle-gen9ou-id form.
    Keep the caller's identity for cache keys and randomness, and supply a
    canonical internal tag to the Gen9OU parser.
    """
    if not isinstance(identity, str) or not identity:
        raise ValueError("Require a nonempty battle identity")
    return "battle-gen9ou-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def reconcile_own_roster(state, request, convert):
    """Remove parser ghosts after Illusion without revealing opponent information.

    Native replay parsing can leave a disguised or fainted own Pokemon in the
    reserve list. Request slots, identities and conditions are authoritative.
    Preserve parsed temporal features and PP where the identity still matches.
    """
    party = request.get("side", {}).get("pokemon", [])
    active = [member for member in party if member.get("active")]
    if len(active) != 1:
        raise ValueError("An own choice request must identify exactly one active Pokemon")
    previous = [state.player_active_pokemon, *state.available_switches]

    def pokemon(member):
        fresh = convert(member)
        matches = [p for p in previous if identifier(p.name) == identifier(fresh.name)]
        if len(matches) > 1:
            raise ValueError("Native own roster has duplicate species identities")
        if not matches:
            return fresh
        result = deepcopy(matches[0])
        for field in ("hp_pct", "status", "item", "ability"):
            setattr(result, field, getattr(fresh, field))
        return result

    if identifier(state.player_active_pokemon.name) != identifier(
        active[0]["details"].split(",")[0]
    ):
        actual = pokemon(active[0])
        # The active public effects belong to the illusionist's physical slot.
        for field, value in vars(state.player_active_pokemon).items():
            if field.endswith("_boost") or field == "effect":
                setattr(actual, field, value)
        state.player_active_pokemon = actual
    reviving = any(member.get("reviving") for member in party)
    state.available_switches = [
        pokemon(member)
        for member in party
        if not member.get("active")
        and (
            member.get("condition", "").endswith("fnt")
            if reviving
            else not member.get("condition", "").endswith("fnt")
        )
    ]
    return state
