"""Entity records [12, D_ent], entity ids [12, N_ID_FIELDS], active indices, and the legality mask.

Slots 0-5 = my team in team order (battle.team insertion order — stable, it comes from my own
request); 6-11 = opponent in preview order (`reveals.opponent_slots` — stable across the battle).
For my own six all epistemic flags read "known" and true values fill the ids; for the opponent only
publicly-derivable info fills, with the epistemic flags derived from the C1 reveal tokens so the
entity record and the belief-net input can never disagree about what is known.

Live extraction:
  * booster_consumed / fallen_count / last_used_move / volatile counters are event-derived: filled
    from the per-battle BattleTracker when one is passed, 0/UNK otherwise.
  * native nature + six EV-unit ids: my own slots recover the effective private spread from the
    request stats; opponent slots read C1 reveal tokens (these fields are normally never public).
  * paradox_boost is best-effort from effect names containing PROTOSYNTHESIS/QUARKDRIVE + stat.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from fog.encoding.pokemon_state import is_terastallized

from fog.constants.schema import (
    EV_STATS,
    N_ACTIONS,
    N_EV_UNITS,
    N_NATURES,
    N_TERA_TYPES,
    NATURE_INDEX,
    NATURES,
    PARADOX_BOOST_STATS,
    SET_EV_COLS,
    SET_N_ATTRS,
    SET_NATURE_COL,
    SET_TERA_COL,
    TYPE_INDEX,
    VOLATILE_COUNTERS,
    VOLATILES,
    Status,
)
from fog.encoding.dims import DIMS, N_ENTITIES
from fog.encoding.events import submitted_species_id
from fog.env.actions import action_legality_mask, team_order

_NON_ID = re.compile(r"[^a-z0-9]")

# Raw (pre-embedding) width of one entity record. Embedded fields contribute their id to entity_ids
# and are excluded from this float block; the model owns the embedding tables.
ENTITY_RAW_DIM = (
    DIMS.base_stats
    + DIMS.typing
    + DIMS.weight_bucket
    + DIMS.hp_fraction
    + DIMS.fainted_flag
    + DIMS.status
    + DIMS.sleep_tox_counters
    + DIMS.boosts
    + DIMS.volatiles
    + DIMS.volatile_counters
    + DIMS.paradox_boost
    + DIMS.booster_consumed
    + DIMS.fallen_count
    + DIMS.move_pp
    + DIMS.tera_type
    + DIMS.tera_used_flag
    + DIMS.epistemic_flags
    + DIMS.is_mine
    + DIMS.is_active
    + DIMS.party_slot
)

# Integer id fields embedded inside the model, in this order:
#   species, item, ability, move0..3, last_used_move, nature, ev_hp..ev_spe
# last_used_move (encoding audit 2026-07) is public info driving Encore/choice-lock/Torment
# inference; it reuses the move embedding table, with <UNKNOWN> (= n_moves) when nothing has been
# used since entry.
NATURE_ID_FIELD = 8
EV_ID_FIELDS = (9, 10, 11, 12, 13, 14)
ORIGINAL_ITEM_ID_FIELD = 15
ORIGINAL_ABILITY_ID_FIELD = 16
BASE_ID_FIELDS = 15
N_ID_FIELDS = 17

_BOOST_ORDER = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_VOLATILE_IDX = {name: i for i, name in enumerate(VOLATILES)}
_COUNTER_IDX = {name: i for i, name in enumerate(VOLATILE_COUNTERS)}
_PARADOX_STAT = {s: i for i, s in enumerate(PARADOX_BOOST_STATS)}


def _to_id(name: object) -> str:
    return _NON_ID.sub("", str(name).lower())


def _own_nature_evs(mon: Any, species_id: str, tables) -> tuple[int, list[int]]:
    """Recover the effective private nature and EV units from a request-side Pokemon.

    Some battle objects retain ``nature``/``evs`` directly. poke-env's live Pokemon generally
    exposes only calculated unboosted stats, so the fallback enumerates the 25 natures and inverts
    each stat over the 64 effective EV units. Neutral natures are battle-equivalent and canonicalize
    to Serious when the request cannot distinguish their labels.
    """
    raw_evs = getattr(mon, "evs", None)
    raw_nature = _to_id(getattr(mon, "nature", None) or "")
    if raw_evs is not None and raw_nature in NATURE_INDEX:
        if isinstance(raw_evs, dict):
            values = [int(raw_evs.get(stat, 0) or 0) for stat in EV_STATS]
        else:
            values = [int(v or 0) for v in list(raw_evs)[:6]]
        if len(values) == 6:
            return NATURE_INDEX[raw_nature], [min(63, max(0, v // 4)) for v in values]

    stats = getattr(mon, "stats", None) or {}
    features = (tables.species.get("features", {}) or {}).get(species_id)
    try:
        actual = np.asarray([int(stats[key]) for key in EV_STATS], dtype=np.int64)
        base = np.asarray(features["base_stats"], dtype=np.int64)
    except (KeyError, TypeError, ValueError):
        return NATURE_INDEX["serious"], [0] * 6

    from fog.constants.build_spreads import nature_multipliers

    level = int(getattr(mon, "level", 100) or 100)
    units = np.arange(N_EV_UNITS, dtype=np.int64)
    best: tuple[int, int, list[int]] | None = None
    for nature in NATURES:
        mult = nature_multipliers(nature)
        chosen: list[int] = []
        error = 0
        for stat in range(6):
            core = (2 * int(base[stat]) + 31 + units) * level // 100
            if stat == 0:
                predicted = (
                    np.ones_like(core)
                    if int(base[0]) == 1
                    else core + level + 10
                )
            else:
                predicted = np.floor((core + 5) * mult[stat]).astype(np.int64)
            distance = np.abs(predicted - actual[stat])
            value = int(distance.argmin())
            chosen.append(value)
            error += int(distance[value])
        over = max(0, sum(chosen) - 127)
        # Exact/legal reconstructions always beat a wrong-nature approximation. Among equivalent
        # neutral labels, prefer Serious as the canonical representation.
        score = error + 1000 * over
        tie = 0 if nature == "serious" else 1
        candidate = (score, tie, chosen)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
            best_nature = nature
    return NATURE_INDEX[best_nature], best[2]  # type: ignore[index,has-type]


class _Layout:
    """Named running offsets into the raw float block, in ENTITY_RAW_DIM order."""

    def __init__(self) -> None:
        o = 0
        for name, width in (
            ("base_stats", DIMS.base_stats), ("typing", DIMS.typing),
            ("weight", DIMS.weight_bucket), ("hp", DIMS.hp_fraction),
            ("fainted", DIMS.fainted_flag), ("status", DIMS.status),
            ("counters", DIMS.sleep_tox_counters), ("boosts", DIMS.boosts),
            ("volatiles", DIMS.volatiles), ("vol_counters", DIMS.volatile_counters),
            ("paradox", DIMS.paradox_boost), ("booster", DIMS.booster_consumed),
            ("fallen", DIMS.fallen_count), ("move_pp", DIMS.move_pp),
            ("tera_type", DIMS.tera_type), ("tera_used", DIMS.tera_used_flag),
            ("epistemic", DIMS.epistemic_flags), ("is_mine", DIMS.is_mine),
            ("is_active", DIMS.is_active), ("party_slot", DIMS.party_slot),
        ):
            setattr(self, name, o)
            o += width
        assert o == ENTITY_RAW_DIM


L = _Layout()


def _fill_mon_record(row: np.ndarray, mon: Any, tables) -> None:
    """Publicly-visible condition/rulebook blocks shared by mine and revealed-opponent mons."""
    stats = getattr(mon, "base_stats", None) or {}
    for i, key in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
        row[L.base_stats + i] = float(stats.get(key, 0)) / 255.0
    for t in getattr(mon, "types", None) or []:
        idx = TYPE_INDEX.get(_to_id(getattr(t, "name", t)))
        if idx is not None:
            row[L.typing + idx] = 1.0
    weight = getattr(mon, "weight", None)
    if weight:
        from fog.constants.build import weight_bucket

        row[L.weight + min(weight_bucket(float(weight)), DIMS.weight_bucket - 1)] = 1.0
    hp = getattr(mon, "current_hp_fraction", None)
    row[L.hp] = float(hp) if hp is not None else 1.0
    row[L.fainted] = 1.0 if getattr(mon, "fainted", False) else 0.0

    status = getattr(mon, "status", None)
    status_name = _to_id(getattr(status, "name", status)).upper() if status else "NONE"
    try:
        member = Status[status_name]
    except KeyError:
        member = Status.NONE
    row[L.status + member.value] = 1.0
    counter = float(getattr(mon, "status_counter", 0) or 0)
    if member is Status.SLP:
        row[L.counters + 0] = counter / 8.0
    elif member is Status.TOX:
        row[L.counters + 1] = counter / 8.0

    boosts = getattr(mon, "boosts", None) or {}
    for i, key in enumerate(_BOOST_ORDER):
        row[L.boosts + i] = float(boosts.get(key, 0)) / 6.0

    for effect in getattr(mon, "effects", None) or {}:
        name = _to_id(getattr(effect, "name", effect))
        if name in _VOLATILE_IDX:
            row[L.volatiles + _VOLATILE_IDX[name]] = 1.0
        # The simulator/poke-env effect value may be an exact internal remaining duration that the
        # player cannot know (partial trapping is the important example).  Countdown features are
        # populated only by BattleTracker from public start/end events below.
        for stat, pi in _PARADOX_STAT.items():
            if ("protosynthesis" in name or "quarkdrive" in name) and name.endswith(stat):
                row[L.paradox + pi] = 1.0

    if is_terastallized(mon):
        row[L.tera_used] = 1.0
    tera = getattr(mon, "tera_type", None)
    if tera is not None and is_terastallized(mon):
        idx = TYPE_INDEX.get(_to_id(getattr(tera, "name", tera)), N_TERA_TYPES - 1)
        row[L.tera_type + idx] = 1.0


def encode_entities(
    battle: Any, tables=None, *, opp_tokens: np.ndarray | None = None, perspective: str = "p1",
    tracker: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (entities[12, ENTITY_RAW_DIM] f32, entity_ids[12,17] i64, active_idx[2] i64).

    `opp_tokens` = the C1 reveal tokens [6,15] from reveals.build_belief_input — the single source
    of truth for the opponent's epistemic flags and revealed ids. battle=None or tables=None keeps
    the zero-fill shape-test path. `tracker` (encoding.tracker.BattleTracker, optional) fills the
    event-derived audit fields: last_used_move, booster_consumed, fallen_count, volatile counters.
    """
    entities = np.zeros((N_ENTITIES, ENTITY_RAW_DIM), dtype=np.float32)
    entity_ids = np.zeros((N_ENTITIES, N_ID_FIELDS), dtype=np.int64)
    active_idx = np.array([0, 6], dtype=np.int64)
    if battle is None or tables is None:
        return entities, entity_ids, active_idx

    # mon -> protocol key ('p1: Garchomp') map for the tracker taps
    key_by_mon: dict[int, str] = {}
    for k, m in (getattr(battle, "team", None) or {}).items():
        key_by_mon[id(m)] = str(k)
    for k, m in (getattr(battle, "opponent_team", None) or {}).items():
        key_by_mon[id(m)] = str(k)
    my_role = getattr(battle, "player_role", None) or perspective
    opp_role = "p2" if my_role == "p1" else "p1"
    now = int(getattr(battle, "turn", 0) or 0)

    def _tracker_fill(row: np.ndarray, ids: np.ndarray, mon: Any, role: str) -> None:
        if tracker is None or mon is None:
            return
        key = key_by_mon.get(id(mon))
        if key is None:
            return
        last = tracker.last_used_move.get(key)
        if last is not None:
            idx = tables.move_index(last, default=None)
            if idx is not None:
                ids[7] = idx
        if tracker.booster_was_consumed(key, mon):
            row[L.booster] = 1.0
        row[L.fallen] = tracker.fallen_for(key, role)
        for ci, eff in enumerate(VOLATILE_COUNTERS):
            v = tracker.counter_remaining(key, eff, now)
            if v > 0:
                row[L.vol_counters + ci] = v

    unk = {
        "species": tables.n_species, "item": tables.n_items,
        "ability": tables.n_abilities, "move": tables.n_moves,
        "nature": N_NATURES, "ev": N_EV_UNITS,
    }

    # ---- my six (slots 0-5): full sheet, all epistemic flags known --------------------------------
    my_mons = team_order(battle)[:6]
    active = getattr(battle, "active_pokemon", None)
    for j, mon in enumerate(my_mons):
        row = entities[j]
        _fill_mon_record(row, mon, tables)
        row[L.is_mine] = 1.0
        row[L.party_slot + j] = 1.0
        row[L.epistemic : L.epistemic + DIMS.epistemic_flags] = 1.0  # I know my own team
        if mon is active or (active is not None and mon == active):
            row[L.is_active] = 1.0
            active_idx[0] = j

        ids = entity_ids[j]
        stable_species_id = submitted_species_id(getattr(mon, "species", ""))
        ids[0] = tables.species_index(stable_species_id)
        item = getattr(mon, "item", None)
        ids[1] = tables.item_index(_to_id(item)) if item else tables.item_index("<NO_ITEM>")
        ability = getattr(mon, "ability", None)
        ids[2] = (
            tables.ability_index(_to_id(ability), default=unk["ability"]) if ability else unk["ability"]
        )
        moves = list((getattr(mon, "moves", None) or {}).items())[:4]
        for k in range(4):
            if k < len(moves):
                move_id, move = moves[k]
                ids[3 + k] = tables.move_index(_to_id(move_id), default=unk["move"])
                max_pp = getattr(move, "max_pp", None)
                cur_pp = getattr(move, "current_pp", None)
                if max_pp:
                    entities[j, L.move_pp + k] = float(cur_pp if cur_pp is not None else max_pp) / max_pp
            else:
                ids[3 + k] = unk["move"]
        ids[7] = unk["move"]      # last_used_move: filled from the tracker below when available
        exact_truth = (
            getattr(tracker, "exact_private_truth", {}).get(int(ids[0]))
            if tracker is not None
            else None
        )
        if exact_truth is not None:
            nature = int(exact_truth[7])
            ev_units = [int(value) for value in exact_truth[8:14]]
        else:
            nature, ev_units = _own_nature_evs(
                mon, stable_species_id, tables
            )
        ids[NATURE_ID_FIELD] = nature
        ids[list(EV_ID_FIELDS)] = ev_units
        mon_key = key_by_mon.get(id(mon), "")
        original_item = tracker.original_item.get(mon_key) if tracker is not None else None
        original_ability = tracker.original_ability.get(mon_key) if tracker is not None else None
        if original_item:
            try:
                ids[ORIGINAL_ITEM_ID_FIELD] = tables.item_index(original_item)
            except KeyError:
                ids[ORIGINAL_ITEM_ID_FIELD] = unk["item"]
        else:
            ids[ORIGINAL_ITEM_ID_FIELD] = unk["item"]
        ids[ORIGINAL_ABILITY_ID_FIELD] = (
            tables.ability_index(original_ability, default=unk["ability"])
            if original_ability else unk["ability"]
        )
        # my tera type is MY sheet knowledge even before terastallizing
        tera = getattr(mon, "tera_type", None)
        if tera is not None and not is_terastallized(mon):
            idx = TYPE_INDEX.get(_to_id(getattr(tera, "name", tera)), N_TERA_TYPES - 1)
            entities[j, L.tera_type + idx] = 1.0
        _tracker_fill(row, ids, mon, my_role)

    # ---- opponent six (slots 6-11): reveal tokens drive ids + epistemic flags ---------------------
    from fog.encoding.reveals import opponent_slots

    opp_active = getattr(battle, "opponent_active_pokemon", None)
    for j, (sid, mon) in enumerate(opponent_slots(battle)):
        slot = 6 + j
        row = entities[slot]
        row[L.party_slot + j] = 1.0
        if mon is not None:
            _fill_mon_record(row, mon, tables)
            if opp_active is not None and (mon is opp_active or mon == opp_active):
                row[L.is_active] = 1.0
                active_idx[1] = slot
        elif sid is not None and sid in tables.species["features"]:
            # preview-only mon: rulebook block from the dex, full HP
            feats = tables.species["features"][sid]
            for i, v in enumerate(feats["base_stats"]):
                row[L.base_stats + i] = v / 255.0
            for t in feats["types"]:
                row[L.typing + TYPE_INDEX[t]] = 1.0
            row[L.weight + min(feats["weight_bucket"], DIMS.weight_bucket - 1)] = 1.0
            row[L.hp] = 1.0

        ids = entity_ids[slot]
        tok = (
            opp_tokens[j]
            if opp_tokens is not None
            else np.full(SET_N_ATTRS, -1, dtype=np.int64)
        )
        ids[0] = tok[0] if tok[0] >= 0 else unk["species"]
        ids[1] = tok[1] if tok[1] >= 0 else unk["item"]
        ids[2] = tok[2] if tok[2] >= 0 else unk["ability"]
        for k in range(4):
            ids[3 + k] = tok[3 + k] if tok[3 + k] >= 0 else unk["move"]
        ids[7] = unk["move"]
        ids[NATURE_ID_FIELD] = (
            tok[SET_NATURE_COL] if tok[SET_NATURE_COL] >= 0 else unk["nature"]
        )
        for target, col in zip(EV_ID_FIELDS, SET_EV_COLS):
            ids[target] = tok[col] if tok[col] >= 0 else unk["ev"]
        ids[ORIGINAL_ITEM_ID_FIELD] = unk["item"]
        ids[ORIGINAL_ABILITY_ID_FIELD] = unk["ability"]
        # epistemic flags mirror every non-species submitted-set field in the C1 grid.
        for bit, col in enumerate(
            (1, 2, 3, 4, 5, 6, SET_NATURE_COL, *SET_EV_COLS, SET_TERA_COL)
        ):
            row[L.epistemic + bit] = 1.0 if tok[col] >= 0 else 0.0
        _tracker_fill(row, ids, mon, opp_role)

    return entities, entity_ids, active_idx


def legality_mask(battle: Any, *, perspective: str = "p1") -> np.ndarray:
    """Compute the 14-way action legality mask from a live battle/request-like object."""
    mask = np.asarray(action_legality_mask(battle), dtype=bool)
    if mask.shape != (N_ACTIONS,):
        raise ValueError(f"action legality mask must have shape ({N_ACTIONS},); got {mask.shape}")
    return mask
