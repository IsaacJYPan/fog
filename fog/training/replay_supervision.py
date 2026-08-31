"""Build conservative submitted-set labels from a censored public battle.

The belief model predicts the opponent's *submitted* set.  Poke-env's terminal Pokemon objects
instead describe current state: a consumed item is ``<NO_ITEM>``, and Trick can replace an item or
ability.  This module reconstructs only facts the public protocol actually exposed and never turns
an unrevealed field into a negative label.

For synthetic validation, ``oracle_censored_truth`` uses the simulator's source paste only to choose
the value of a fact whose public reveal was independently observed.  It does not expose pool/team
identity or any field that remained hidden, matching the supervision available from real replays.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fog.encoding.events import event_mon_key, event_role, to_id
from fog.encoding.history import observations_by_turn
from fog.encoding.reveals import opponent_slots

_MOVE_COLS = (3, 4, 5, 6)
_TRANSFER_MARKERS = (
    "trick",
    "switcheroo",
    "skillswap",
    "roleplay",
    "entrainment",
    "trace",
    "receiver",
    "powerofalchemy",
    "mummy",
    "lingeringaroma",
    "wanderingspirit",
    "pickpocket",
    "magician",
    "symbiosis",
    "bestow",
    "covet",
    "thief",
)


def _preview_species_id(details: str) -> str:
    """Normalize a protocol switch/replace detail string to its team-preview species id.

    Showdown represents a terastallized form in switch details as (for example)
    ``Ogerpon-Cornerstone-Tera`` while team preview remains ``Ogerpon-Cornerstone``.  Tera is a
    battle-state form, not a different submitted Pokemon, so it must not break the stable preview
    slot mapping used by censored set supervision.
    """
    species = str(details).split(",")[0].strip()
    if species.lower().endswith("-tera"):
        species = species[:-5]
    return to_id(species)


def _is_transfer(extra: list[str]) -> bool:
    compact = " ".join(to_id(part) for part in extra)
    return any(marker in compact for marker in _TRANSFER_MARKERS)


def _effect_source(args: list[str], prefix: str) -> tuple[str | None, str | None]:
    """Return (source value, owner token) for ``[from] item:/ability:`` protocol evidence."""
    marker = f"[from] {prefix}:"
    value = next(
        (
            part[len(marker) :].strip()
            for part in args
            if part.lower().startswith(marker)
        ),
        None,
    )
    if value is None:
        return None, None
    owner = next(
        (
            part[len("[of] ") :].strip()
            for part in args
            if part.lower().startswith("[of] ")
        ),
        args[0] if args else None,
    )
    return value, owner


def _effect_value(args: list[str], prefix: str) -> str | None:
    marker = f"[from] {prefix}:"
    return next(
        (
            part[len(marker) :].strip()
            for part in args
            if part.lower().startswith(marker)
        ),
        None,
    )


def public_replay_truth(battle: Any, tables) -> np.ndarray:
    """Return ``[6,15]`` original-set labels recoverable from this public perspective.

    Species follow team-preview slot order. Moves are packed in public reveal order. Nature and all
    EV fields remain ``-1`` because ordinary Showdown replays do not reveal the exact submitted set.
    Item/ability changes caused by transfer mechanics are handled conservatively.
    """
    slots = opponent_slots(battle)
    from fog.constants.schema import SET_N_ATTRS, SET_TERA_COL

    labels = np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
    species_to_slot: dict[int, int] = {}
    for slot, (species_id, _) in enumerate(slots):
        if not species_id:
            continue
        species = tables.species_index(species_id)
        labels[slot, 0] = species
        species_to_slot.setdefault(species, slot)

    key_to_slot: dict[str, int] = {}
    for key, mon in (getattr(battle, "opponent_team", None) or {}).items():
        species = getattr(mon, "species", None)
        if species:
            slot = species_to_slot.get(tables.species_index(to_id(species)))
            if slot is not None:
                key_to_slot[str(key)] = slot

    # A Zoroark can enter under an Illusion: the public move lines use the disguise's name until a
    # later ``|replace|...|Zoroark...|`` reveals the true species.  Slot-keyed accumulation would
    # permanently assign those earlier moves to the disguise (for example Bitter Malice on
    # Arcanine-Hisui), creating labels that violate the hard learnset mask.  Keep move evidence on
    # a distinct active *incarnation* and point that incarnation at the actual preview slot once
    # Illusion resolves.  A genuine Pokemon that later switches in under the same nickname gets a
    # new incarnation, so its already-observed moves cannot be stolen by the illusion.
    active_incarnation: int | None = None
    active_key: str | None = None
    next_incarnation = 0
    incarnation_slot: dict[int, int] = {}
    incarnation_moves: dict[int, list[int]] = {}
    incarnation_tainted: set[int] = set()
    item_tainted = [False] * 6
    ability_tainted = [False] * 6
    role = getattr(battle, "player_role", None) or "p1"
    opponent_role = "p2" if role == "p1" else "p1"

    def ability_is_legal(slot: int, ability: int) -> bool:
        species = int(labels[slot, 0])
        return (
            species >= 0
            and ability >= 0
            and ability < tables.n_abilities
            and bool(tables.legal_abilities_mask(species)[ability])
        )

    def unique_move_slot(move: int, current_slot: int) -> int | None:
        """Return the preview slot uniquely compatible with an ordinary revealed move.

        This closes the remaining Illusion case: the protocol need not emit ``|replace|`` merely
        because a disguise used a move it cannot learn.  Team preview already contains Zoroark's
        true species, so an otherwise-illegal move that only one preview member can learn is public
        proof of that member's identity.  Ambiguous cases deliberately remain censored.
        """
        if bool(tables.learnset_mask(int(labels[current_slot, 0]))[move]):
            return current_slot
        candidates = [
            slot
            for slot in range(6)
            if labels[slot, 0] >= 0
            and bool(tables.learnset_mask(int(labels[slot, 0]))[move])
        ]
        return candidates[0] if len(candidates) == 1 else None

    def record_ability(slot: int | None, ability_name: str, *, taint_after: bool = False) -> bool:
        if slot is None:
            return False
        ability = tables.ability_index(to_id(ability_name), default=-1)
        recorded = False
        if (
            labels[slot, 2] < 0
            and not ability_tainted[slot]
            and ability_is_legal(slot, ability)
        ):
            labels[slot, 2] = ability
            recorded = True
        if taint_after:
            ability_tainted[slot] = True
        return recorded

    for turn in sorted(observations_by_turn(battle)):
        events = getattr(observations_by_turn(battle)[turn], "events", None) or []
        for event in events:
            if len(event) < 3:
                continue
            kind, args = event[1], event[2:]
            key = event_mon_key(args[0])
            primary_is_opponent = event_role(args[0]) == opponent_role

            if primary_is_opponent and kind in ("switch", "drag", "replace") and len(args) >= 2:
                species = tables.species_index(_preview_species_id(args[1]))
                slot = species_to_slot.get(species)
                # A switch can use a display-only form that is not in the generated species
                # vocabulary.  The nickname was already mapped on a prior genuine switch, so it
                # remains a sound fallback (and avoids assigning the new incarnation to the
                # previous active slot).
                if slot is None:
                    slot = key_to_slot.get(key)
                if slot is not None:
                    key_to_slot[key] = slot
                    if kind in ("switch", "drag"):
                        active_incarnation = next_incarnation
                        next_incarnation += 1
                        active_key = key
                        incarnation_slot[active_incarnation] = slot
                        incarnation_moves[active_incarnation] = []
                    elif active_incarnation is not None:
                        # ``replace`` is the protocol event that resolves Illusion.  The key in
                        # the event is already the true identity, while prior move events used
                        # active_key (the disguise); both names must now resolve to the same slot.
                        incarnation_slot[active_incarnation] = slot
                        if active_key is not None:
                            key_to_slot[active_key] = slot
            slot = key_to_slot.get(key) if primary_is_opponent else None

            if slot is not None and kind == "move" and len(args) >= 2:
                move = tables.move_index(to_id(args[1]), default=None)
                called = any(part.lower().startswith("[from]") for part in args[2:])
                move_slot = (
                    unique_move_slot(move, incarnation_slot[active_incarnation])
                    if (
                        active_incarnation is not None
                        and active_incarnation not in incarnation_tainted
                        and move is not None
                        and not called
                    )
                    else None
                )
                if move_slot is not None and active_incarnation is not None:
                    incarnation_slot[active_incarnation] = move_slot
                    if active_key is not None:
                        key_to_slot[active_key] = move_slot
                if (
                    active_incarnation is not None
                    and active_incarnation not in incarnation_tainted
                    and not called
                    and move is not None
                    and move_slot is not None
                    and move not in incarnation_moves[active_incarnation]
                    and len(incarnation_moves[active_incarnation]) < 4
                ):
                    incarnation_moves[active_incarnation].append(move)
                if to_id(args[1]) == "transform" and active_incarnation is not None:
                    incarnation_tainted.add(active_incarnation)
            elif (
                slot is not None
                and kind == "-transform"
                and active_incarnation is not None
            ):
                # Imposter emits a separate |-transform| event, then later move lines name the
                # copied moves.  Those moves are public battle facts but not Ditto's submitted
                # moveset and must not become belief labels.
                incarnation_tainted.add(active_incarnation)
            elif slot is not None and kind == "-enditem" and len(args) >= 2:
                # Knock Off/consumption names the item that was held.  During Trick the lost item is
                # still the original only if this mon had not already received a replacement.
                item = tables.item_index(to_id(args[1]))
                if labels[slot, 1] < 0 and not item_tainted[slot]:
                    labels[slot, 1] = item
                if _is_transfer(args[2:]):
                    item_tainted[slot] = True
            elif slot is not None and kind == "-item" and len(args) >= 2:
                if _is_transfer(args[2:]):
                    item_tainted[slot] = True
                elif labels[slot, 1] < 0 and not item_tainted[slot]:
                    labels[slot, 1] = tables.item_index(to_id(args[1]))
            elif slot is not None and kind == "-ability" and len(args) >= 2:
                changed = _is_transfer(args[2:])
                # Trace is the original submitted ability named in the source marker; args[1] is
                # the copied current ability. Other transfer effects are retained in history but
                # are not safe original-set labels.
                original = _effect_value(args[2:], "ability")
                if original is not None:
                    record_ability(slot, original)
                if changed:
                    ability_tainted[slot] = True
                else:
                    record_ability(slot, args[1])
            elif slot is not None and kind == "-terastallize" and len(args) >= 2:
                from fog.constants.schema import TYPE_INDEX

                labels[slot, SET_TERA_COL] = TYPE_INDEX.get(to_id(args[1]), 18)
                # Terastallization can itself replace an ability (Ogerpon, Terapagos). Later
                # ability messages are current-form facts unless the original was already known.
                ability_tainted[slot] = True

            # Skill Swap names both pre-swap abilities in a single public event:
            #   actor | Skill Swap | target-original | actor-original | [of] target
            # Record those submitted facts before tainting both current abilities.
            if kind == "-activate" and len(args) >= 5 and to_id(args[1]) == "skillswap":
                other_token = next(
                    (
                        part[len("[of] ") :].strip()
                        for part in args[4:]
                        if part.lower().startswith("[of] ")
                    ),
                    None,
                )
                actor_slot = key_to_slot.get(key) if primary_is_opponent else None
                other_slot = (
                    key_to_slot.get(event_mon_key(other_token))
                    if other_token and event_role(other_token) == opponent_role
                    else None
                )
                record_ability(actor_slot, args[3], taint_after=True)
                record_ability(other_slot, args[2], taint_after=True)

            item_name, item_owner = (
                _effect_source(args, "item") if kind != "-item" else (None, None)
            )
            if item_name and item_owner and event_role(item_owner) == opponent_role:
                item_slot = key_to_slot.get(event_mon_key(item_owner))
                if (
                    item_slot is not None
                    and labels[item_slot, 1] < 0
                    and not item_tainted[item_slot]
                ):
                    labels[item_slot, 1] = tables.item_index(to_id(item_name))

            ability_name = _effect_value(args, "ability") if kind != "-ability" else None
            if ability_name:
                ability = tables.ability_index(to_id(ability_name), default=-1)
                candidates: set[int] = set()
                for owner_token in (
                    [args[0]]
                    + [
                        part[len("[of] ") :].strip()
                        for part in args
                        if part.lower().startswith("[of] ")
                    ]
                ):
                    if event_role(owner_token) != opponent_role:
                        continue
                    candidate = key_to_slot.get(event_mon_key(owner_token))
                    if candidate is not None and ability_is_legal(candidate, ability):
                        candidates.add(candidate)
                # `[of]` is not uniformly the ability owner: it is Rough Skin's holder on a
                # damage event, but the attacker whose move triggered Water Absorb on a heal
                # event. A unique legality-compatible participant is sound; ambiguous events
                # stay censored instead of manufacturing a submitted ability.
                if len(candidates) == 1:
                    record_ability(candidates.pop(), ability_name)

    moves: list[list[int]] = [[] for _ in range(6)]
    for incarnation, values in incarnation_moves.items():
        slot = incarnation_slot.get(incarnation)
        if slot is None:
            continue
        for move in values:
            if move not in moves[slot] and len(moves[slot]) < 4:
                moves[slot].append(move)
    for slot in range(6):
        labels[slot, 3 : 3 + len(moves[slot])] = moves[slot]
    return labels


def oracle_censored_truth(
    battle: Any,
    full_truth: np.ndarray,
    tables,
) -> np.ndarray:
    """Censor known simulator truth by independently observed public replay reveals.

    The returned values are suitable for validating a replay-first learner on generated battles.
    Full truth resolves original-vs-current semantics but cannot make any unrevealed mask entry
    valid. Inconsistent observed values are dropped rather than forcing an oracle label.
    """
    public = public_replay_truth(battle, tables)
    truth = np.asarray(full_truth, dtype=np.int64)
    out = np.full_like(truth, -1)
    for slot in range(6):
        if public[slot, 0] == truth[slot, 0]:
            out[slot, 0] = truth[slot, 0]
        for column in (1, 2, 8):
            if public[slot, column] >= 0 and public[slot, column] == truth[slot, column]:
                out[slot, column] = truth[slot, column]
        truth_moves = {int(move) for move in truth[slot, _MOVE_COLS] if move >= 0}
        observed_moves = [
            int(move) for move in public[slot, _MOVE_COLS] if int(move) in truth_moves
        ]
        out[slot, 3 : 3 + len(observed_moves)] = observed_moves[:4]
    return out


__all__ = ["oracle_censored_truth", "public_replay_truth"]
