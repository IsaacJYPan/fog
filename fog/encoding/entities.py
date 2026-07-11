"""Entity records [12, D_ent], entity ids [12, 9], active indices, and the legality mask (spec §2.1).

Slots 0-5 = my team in team order (battle.team insertion order — stable, it comes from my own
request); 6-11 = opponent in preview order (`reveals.opponent_slots` — stable across the battle).
For my own six all epistemic flags read "known" and true values fill the ids; for the opponent only
publicly-derivable info fills, with the epistemic flags derived from the C1 reveal tokens so the
entity record and the belief-net input can never disagree about what is known.

Fidelity notes (v1, live extraction — see KNOWN_ISSUES):
  * booster_consumed / fallen_count / last_used_move / volatile counters are event-derived: filled
    from the per-battle BattleTracker when one is passed, 0/UNK otherwise.
  * spread cluster id: my own slots stay UNK (spread inference from my sheet is not wired — a known
    fidelity bound); opponent slots read the C1 reveal token (spreads are never directly revealed).
  * paradox_boost is best-effort from effect names containing PROTOSYNTHESIS/QUARKDRIVE + stat.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from fog.constants.schema import (
    N_ACTIONS,
    N_TERA_TYPES,
    PARADOX_BOOST_STATS,
    Status,
    TYPE_INDEX,
    VOLATILE_COUNTERS,
    VOLATILES,
)
from fog.encoding.dims import DIMS, N_ENTITIES
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
#   species, item, ability, move0..3, last_used_move, spread
# last_used_move (encoding audit 2026-07) is public info driving Encore/choice-lock/Torment
# inference; it reuses the move embedding table, with <UNKNOWN> (= n_moves) when nothing has been
# used since entry. NOTE: encoder.EntityEmbeddings reads spread at index [-1] — keep spread last.
N_ID_FIELDS = 1 + 1 + 1 + DIMS.n_move_slots + 1 + 1

_BOOST_ORDER = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_VOLATILE_IDX = {name: i for i, name in enumerate(VOLATILES)}
_COUNTER_IDX = {name: i for i, name in enumerate(VOLATILE_COUNTERS)}
_PARADOX_STAT = {s: i for i, s in enumerate(PARADOX_BOOST_STATS)}


def _to_id(name: object) -> str:
    return _NON_ID.sub("", str(name).lower())


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

    for effect, count in (getattr(mon, "effects", None) or {}).items():
        name = _to_id(getattr(effect, "name", effect))
        if name in _VOLATILE_IDX:
            row[L.volatiles + _VOLATILE_IDX[name]] = 1.0
        if name in _COUNTER_IDX:
            row[L.vol_counters + _COUNTER_IDX[name]] = float(count or 0) / 8.0
        for stat, pi in _PARADOX_STAT.items():
            if ("protosynthesis" in name or "quarkdrive" in name) and name.endswith(stat):
                row[L.paradox + pi] = 1.0

    if getattr(mon, "terastallized", False):
        row[L.tera_used] = 1.0
    tera = getattr(mon, "tera_type", None)
    if tera is not None and (getattr(mon, "terastallized", False)):
        idx = TYPE_INDEX.get(_to_id(getattr(tera, "name", tera)), N_TERA_TYPES - 1)
        row[L.tera_type + idx] = 1.0


def encode_entities(
    battle: Any, tables=None, *, opp_tokens: np.ndarray | None = None, perspective: str = "p1",
    tracker: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (entities[12, ENTITY_RAW_DIM] f32, entity_ids[12, 9] i64, active_idx[2] i64).

    `opp_tokens` = the C1 reveal tokens [6, 9] from reveals.build_belief_input — the single source
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
        if key in tracker.booster_consumed:
            row[L.booster] = 1.0
        row[L.fallen] = tracker.fallen_for(key, role)
        for ci, eff in enumerate(VOLATILE_COUNTERS):
            v = tracker.counter_remaining(key, eff, now)
            if v > 0:
                row[L.vol_counters + ci] = v

    from fog.constants.schema import MAX_SPREADS_PER_SPECIES

    unk = {
        "species": tables.n_species, "item": tables.n_items,
        "ability": tables.n_abilities, "move": tables.n_moves,
        "spread": MAX_SPREADS_PER_SPECIES + 1,  # clusters + <OTHER_SPREAD>; UNK row on top
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
        ids[0] = tables.species_index(_to_id(getattr(mon, "species", "")))
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
        ids[8] = unk["spread"]    # own spread cluster: not wired (module-docstring fidelity note)
        # my tera type is MY sheet knowledge even before terastallizing
        tera = getattr(mon, "tera_type", None)
        if tera is not None and not getattr(mon, "terastallized", False):
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
        tok = opp_tokens[j] if opp_tokens is not None else np.full(9, -1, dtype=np.int64)
        ids[0] = tok[0] if tok[0] >= 0 else unk["species"]
        ids[1] = tok[1] if tok[1] >= 0 else unk["item"]
        ids[2] = tok[2] if tok[2] >= 0 else unk["ability"]
        for k in range(4):
            ids[3 + k] = tok[3 + k] if tok[3 + k] >= 0 else unk["move"]
        ids[7] = unk["move"]
        ids[8] = tok[7] if tok[7] >= 0 else unk["spread"]
        # epistemic flags [item, ability, move0-3, spread, tera] from the same tokens (C1 alignment)
        for bit, col in enumerate((1, 2, 3, 4, 5, 6, 7, 8)):
            row[L.epistemic + bit] = 1.0 if tok[col] >= 0 else 0.0
        _tracker_fill(row, ids, mon, opp_role)

    return entities, entity_ids, active_idx


def legality_mask(battle: Any, *, perspective: str = "p1") -> np.ndarray:
    """Compute the 14-way action legality mask from a live battle/request-like object."""
    try:
        mask = np.asarray(action_legality_mask(battle), dtype=bool)
    except Exception:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[13] = True
    return mask
