"""Truthful public-evidence checks shared by real-play and simulated-leaf belief updates.

The input is a public [6, 15] reveal grid and a collection of complete candidate team sheets.
The output says which candidate sheets can still be true. Keeping this calculation in one module
prevents the real-play table and the simulated-leaf table from applying different rules to the
same public fact.

Only facts that identify a team-sheet attribute belong here. Current HP, fainting, and the active
slot are already part of the public board. Damage magnitude and speed-order likelihoods require
separate mechanical calculations and are deliberately not inferred from the reveal grid.
"""
from __future__ import annotations

import numpy as np

# Public exact fields in SET_SLOT_ATTRS order: species, item, ability, move0..3, tera.
_MOVE_COLS = slice(3, 7)
_EXACT_COLS = (0, 1, 2, 14)


def keep_mask_from_reveals(
    revealed_tokens: np.ndarray,
    candidates: np.ndarray,
    *,
    no_item_token: int | None = None,
) -> np.ndarray:
    """Return one boolean per candidate team: can it produce all truthful public reveals?

    Non-move attributes match exactly, with two conservative exceptions. ``-1`` in a candidate
    means that the sheet did not specify the attribute and cannot disprove a public value. A
    public current item of ``<NO_ITEM>`` does not disprove the sheet item, because an item may
    have been consumed, removed, or moved after team selection.

    Revealed moves use set membership rather than move-slot order. Callers remain responsible for
    excluding facts made untruthful by Illusion, Transform, or item movement.
    """
    revealed = np.asarray(revealed_tokens, dtype=np.int64)
    cands = np.asarray(candidates, dtype=np.int64)
    keep = np.ones(len(cands), dtype=bool)
    for candidate_index, candidate in enumerate(cands):
        possible = True
        for slot in range(revealed.shape[0]):
            for column in _EXACT_COLS:
                value = int(revealed[slot, column])
                candidate_value = int(candidate[slot, column])
                if value < 0 or candidate_value < 0:
                    continue
                if column == 1 and no_item_token is not None and value == no_item_token:
                    continue
                if candidate_value != value:
                    possible = False
                    break
            if not possible:
                break
            shown = revealed[slot, _MOVE_COLS]
            candidate_moves = {
                int(move) for move in candidate[slot, _MOVE_COLS] if int(move) >= 0
            }
            if any(int(move) not in candidate_moves for move in shown if int(move) >= 0):
                possible = False
                break
        keep[candidate_index] = possible
    return keep


def keep_mask_from_initial_drive_activation(
    battle,
    role: str,
    candidates: np.ndarray,
) -> np.ndarray:
    """Filter candidate sheets by the lead's deterministic Protosynthesis/Quark Drive signal.

    A reveal grid records facts that appeared, but cannot express that an automatic public event
    *did not* appear. At the untouched turn-one root, that absence is decisive. For example, a
    Great Tusk with Booster Energy must consume it and announce Protosynthesis on entry when sun
    is absent. If the public root contains no such activation, that candidate is impossible even
    though no positive item token exists to contradict it.

    This check is deliberately restricted to the untouched initial root. Later in a battle, an
    item may have been removed or moved before an entry, so absence cannot be interpreted safely
    without the complete item-event history. Weather/terrain already present on the public board
    is included because it independently activates the corresponding ability without consuming
    Booster Energy. The exact boosted-stat suffix is compared as well; it is a deterministic
    consequence of the candidate's nature/EV stats and is public in the engine protocol.
    """
    cands = np.asarray(candidates, dtype=np.int64)
    keep = np.ones(len(cands), dtype=bool)
    if int(getattr(battle, "turn", 0)) != 1 or getattr(battle, "observations", None):
        return keep

    from fog.env.engine_battle import _decode_world_slot, build_engine_pokemon, to_id

    tables = battle.tables
    active = battle.active(role)
    species_token = tables.species_index(to_id(active.id))
    side = battle.side(role)
    observed = {to_id(value) for value in (side.volatile_statuses or ())}
    observed_drive = {
        value for value in observed
        if value.startswith(("protosynthesis", "quarkdrive"))
    }
    weather = to_id(battle.state.weather)
    terrain = to_id(battle.state.terrain)

    for candidate_index, candidate in enumerate(cands):
        row_index = next(
            (index for index, row in enumerate(candidate) if int(row[0]) == species_token),
            None,
        )
        if row_index is None:
            keep[candidate_index] = False
            continue
        row = candidate[row_index]
        ability = (
            tables.abilities["ids"][int(row[2])]
            if 0 <= int(row[2]) < tables.n_abilities
            else ""
        )
        item = (
            tables.items["ids"][int(row[1])]
            if 0 <= int(row[1]) < tables.n_items
            else ""
        )
        ability = to_id(ability)
        item = to_id(item)
        field_active = (
            ability == "protosynthesis" and weather in {"sun", "harshsun"}
        ) or (
            ability == "quarkdrive" and terrain == "electricterrain"
        )
        booster_active = item == "boosterenergy" and ability in {
            "protosynthesis", "quarkdrive",
        }
        if not field_active and not booster_active:
            if observed_drive:
                keep[candidate_index] = False
            continue

        mon = build_engine_pokemon(_decode_world_slot(row, int(active.level), tables), tables)
        stats = {
            "atk": int(mon.attack),
            "def": int(mon.defense),
            "spa": int(mon.special_attack),
            "spd": int(mon.special_defense),
            "spe": int(mon.speed),
        }
        boosted = max(stats, key=stats.get)
        if f"{ability}{boosted}" not in observed_drive:
            keep[candidate_index] = False
    return keep
