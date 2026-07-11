"""Per-turn protocol-event parsing -> history turn records + reveal-ledger feeding (spec §2.3, §4.3).

This is the mini-parser shared by the live player and (later) the replay parser. Input is poke-env's
per-turn event lists (`battle.observations[turn].events`): each event is one protocol line already
split on '|', e.g. ['', 'move', 'p1a: Garchomp', 'Earthquake', 'p2a: Corviknight'].

Turn-record layout (offsets into DIMS.turn_dim = 108, see dims.py):
  0..29   snapshot     (start-of-turn actives/field; supplied by the caller — the live player builds
                        it at decision time via `build_snapshot`, the replay parser via its tracker;
                        None -> zeros)
  30..43  my action    14-way one-hot (caller passes the chosen index; retro mode infers move slots
                        from my known movesets where possible, leaves switches at zero)
  44..83  opp action   [moved, move_idx/1000, switched, switch_species_idx/1000, tera, <UNSEEN>]
  84..107 resolution   [dmg_dealt, dmg_taken, crit_mine, crit_opp, eff_mine(5), eff_opp(5),
                        miss_mine, miss_opp, opp_item_revealed, opp_ability_revealed,
                        faint_mine, faint_opp, i_moved_first, exact_dmg_frac, spare(2)]
                        effectiveness one-hot order: [immune, resisted, neutral, super, unknown]

Damage fractions are derived from the hp values inside `|-damage|`/`|-heal|` conditions, seeded with
start-of-turn hp for the actives (`hp_seed`); deltas whose prior hp is unknown are skipped rather
than guessed.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from fog.constants.schema import TYPE_INDEX
from fog.encoding.dims import DIMS

_NON_ID = re.compile(r"[^a-z0-9]")

# --- offsets ---------------------------------------------------------------------------------------
_OFF_SNAP = 0
_OFF_MYACT = DIMS.hist_snapshot                                   # 30
_OFF_OPP = _OFF_MYACT + DIMS.hist_my_action                       # 44
_OFF_RES = _OFF_OPP + DIMS.hist_opp_action                        # 84
# resolution sub-offsets
_R_DMG_DEALT, _R_DMG_TAKEN, _R_CRIT_MINE, _R_CRIT_OPP = 0, 1, 2, 3
_R_EFF_MINE, _R_EFF_OPP = 4, 9
_R_MISS_MINE, _R_MISS_OPP, _R_ITEM_REV, _R_ABIL_REV = 14, 15, 16, 17
_R_FAINT_MINE, _R_FAINT_OPP, _R_FIRST, _R_EXACT = 18, 19, 20, 21
_EFF = {"immune": 0, "resisted": 1, "neutral": 2, "super": 3, "unknown": 4}


def to_id(name: object) -> str:
    return _NON_ID.sub("", str(name).lower())


def event_role(token: str) -> str | None:
    """'p1a: Garchomp' -> 'p1'. Returns None for non-positional tokens."""
    m = re.match(r"(p[12])[ab]?:", token)
    return m.group(1) if m else None


def event_mon_key(token: str) -> str:
    """'p1a: Garchomp' -> 'p1: Garchomp' — matches poke-env's battle.team dict keys."""
    m = re.match(r"(p[12])[ab]?:\s*(.*)", token)
    return f"{m.group(1)}: {m.group(2)}" if m else token


def hp_fraction(condition: str) -> float | None:
    """'163/323', '57/100 brn', '0 fnt' -> fraction in [0,1]; None if unparseable."""
    condition = condition.strip()
    if condition.startswith("0"):
        return 0.0
    m = re.match(r"(\d+)/(\d+)", condition)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    return num / den if den else None


def build_snapshot(battle: Any, tables) -> np.ndarray:
    """Start-of-turn snapshot [30]: both actives' species/1000, hp fracs, status one-hots(7+7),
    boost sums /12, weather one-hot(5), terrain one-hot(5). Duck-typed off a live battle."""
    snap = np.zeros(DIMS.hist_snapshot, dtype=np.float32)
    mine = getattr(battle, "active_pokemon", None)
    opp = getattr(battle, "opponent_active_pokemon", None)
    for base, mon in ((0, mine), (1, opp)):
        if mon is None:
            continue
        species = getattr(mon, "species", None)
        if species:
            snap[base] = tables.species_index(to_id(species)) / 1000.0
        hp = getattr(mon, "current_hp_fraction", None)
        snap[2 + base] = float(hp) if hp is not None else 1.0
        status = getattr(mon, "status", None)
        if status is not None:
            from fog.constants.schema import Status

            name = to_id(getattr(status, "name", status)).upper()
            try:
                snap[4 + 7 * base + Status[name].value] = 1.0
            except KeyError:
                pass
        boosts = getattr(mon, "boosts", None) or {}
        snap[18 + base] = float(sum(boosts.values())) / 12.0
    # weather(5) at 20..24, terrain(5) at 25..29 — presence one-hots off the live dicts
    from fog.constants.schema import Terrain, Weather

    for w in list(getattr(battle, "weather", None) or {}):
        name = to_id(getattr(w, "name", w)).upper()
        for member in Weather:
            if member.name in name and member is not Weather.NONE:
                snap[20 + member.value] = 1.0
    for f in list(getattr(battle, "fields", None) or {}):
        name = to_id(getattr(f, "name", f)).upper()
        for member in Terrain:
            if member is not Terrain.NONE and f"{member.name}TERRAIN" in name:
                snap[25 + member.value] = 1.0
    return snap


def parse_turn_events(
    events: list[list[str]],
    my_role: str,
    tables,
    *,
    my_action_idx: int | None = None,
    my_movesets: dict[str, dict[str, int]] | None = None,
    snapshot: np.ndarray | None = None,
    hp_seed: dict[str, float] | None = None,
) -> np.ndarray:
    """One completed turn's events -> a [DIMS.turn_dim] record. See module docstring for layout.

    `my_movesets`: {mon_key: {move_id: action slot 0-3}} for retro my-action inference (only used
    when `my_action_idx` is None). `hp_seed`: {mon_key: hp fraction at turn start}.
    """
    rec = np.zeros(DIMS.turn_dim, dtype=np.float32)
    if snapshot is not None:
        rec[_OFF_SNAP : _OFF_SNAP + DIMS.hist_snapshot] = snapshot
    opp_role = "p2" if my_role == "p1" else "p1"

    hp = dict(hp_seed or {})
    dmg_to_opp = 0.0
    dmg_to_me = 0.0
    max_single_hit_on_opp = 0.0
    first_mover: str | None = None
    opp_acted = False
    eff_mine = eff_opp = _EFF["unknown"]

    def res(i: int, v: float = 1.0) -> None:
        rec[_OFF_RES + i] = v

    for ev in events:
        if len(ev) < 2:
            continue
        kind = ev[1]
        args = ev[2:]

        if kind == "move" and args:
            role = event_role(args[0])
            if first_mover is None and role in ("p1", "p2"):
                first_mover = role
            if role == opp_role:
                opp_acted = True
                rec[_OFF_OPP + 0] = 1.0  # moved
                if len(args) >= 2:
                    idx = tables.move_index(to_id(args[1]), default=None)
                    if idx is not None:
                        rec[_OFF_OPP + 1] = idx / 1000.0
            elif role == my_role and my_action_idx is None and len(args) >= 2 and my_movesets:
                slots = my_movesets.get(event_mon_key(args[0])) or {}
                slot = slots.get(to_id(args[1]))
                if slot is not None and 0 <= slot < 4:
                    rec[_OFF_MYACT + slot] = 1.0

        elif kind == "switch" and args:
            role = event_role(args[0])
            if role == opp_role:
                opp_acted = True
                rec[_OFF_OPP + 2] = 1.0  # switched
                if len(args) >= 2:
                    species = to_id(args[1].split(",")[0])
                    rec[_OFF_OPP + 3] = tables.species_index(species) / 1000.0
            # seed hp for whoever switched in (condition is args[2] when present)
            if len(args) >= 3:
                frac = hp_fraction(args[2])
                if frac is not None:
                    hp[event_mon_key(args[0])] = frac

        elif kind in ("-damage", "-heal") and len(args) >= 2:
            key = event_mon_key(args[0])
            role = event_role(args[0])
            new = hp_fraction(args[1])
            if new is not None:
                prev = hp.get(key)
                if prev is not None and kind == "-damage":
                    delta = max(0.0, prev - new)
                    if role == opp_role:
                        dmg_to_opp += delta
                        max_single_hit_on_opp = max(max_single_hit_on_opp, delta)
                    elif role == my_role:
                        dmg_to_me += delta
                hp[key] = new

        elif kind == "-crit" and args:
            if event_role(args[0]) == opp_role:
                res(_R_CRIT_MINE)  # crit against the opponent = my attack crit
            else:
                res(_R_CRIT_OPP)

        elif kind in ("-supereffective", "-resisted", "-immune") and args:
            bucket = {"-supereffective": "super", "-resisted": "resisted", "-immune": "immune"}[kind]
            if event_role(args[0]) == opp_role:
                eff_mine = _EFF[bucket]
            else:
                eff_opp = _EFF[bucket]

        elif kind == "-miss" and args:
            if event_role(args[0]) == my_role:
                res(_R_MISS_MINE)
            else:
                res(_R_MISS_OPP)

        elif kind in ("-item", "-enditem", "-ability") and args:
            if event_role(args[0]) == opp_role:
                res(_R_ITEM_REV if kind != "-ability" else _R_ABIL_REV)

        elif kind == "-terastallize" and args:
            if event_role(args[0]) == opp_role:
                rec[_OFF_OPP + 4] = 1.0

        elif kind == "faint" and args:
            if event_role(args[0]) == my_role:
                res(_R_FAINT_MINE)
            else:
                res(_R_FAINT_OPP)

    if my_action_idx is not None and 0 <= my_action_idx < DIMS.hist_my_action:
        rec[_OFF_MYACT + my_action_idx] = 1.0
    if not opp_acted:
        rec[_OFF_OPP + 5] = 1.0  # <UNSEEN>: pre-empted — itself informative
    if eff_mine == _EFF["unknown"] and dmg_to_opp > 0:
        eff_mine = _EFF["neutral"]
    if eff_opp == _EFF["unknown"] and dmg_to_me > 0:
        eff_opp = _EFF["neutral"]
    rec[_OFF_RES + _R_EFF_MINE + eff_mine] = 1.0
    rec[_OFF_RES + _R_EFF_OPP + eff_opp] = 1.0
    res(_R_DMG_DEALT, min(1.0, dmg_to_opp))
    res(_R_DMG_TAKEN, min(1.0, dmg_to_me))
    res(_R_EXACT, round(min(1.0, max_single_hit_on_opp) * 255) / 255.0)
    if first_mover is not None:
        res(_R_FIRST, 1.0 if first_mover == my_role else 0.0)
    return rec


# --------------------------------------------------------------------------------- ledger feeding
def feed_ledger_from_events(events: list[list[str]], my_role: str, ledger, tables) -> None:
    """Record MY public reveals (the opponent's growing view of me, §3.3) off one turn's events."""
    from fog.constants.schema import NO_ITEM

    for ev in events:
        if len(ev) < 3:
            continue
        kind, target = ev[1], ev[2]
        if event_role(target) != my_role:
            continue
        key = event_mon_key(target)
        if kind == "move" and len(ev) >= 4:
            idx = tables.move_index(to_id(ev[3]), default=None)
            if idx is not None:
                ledger.record_move(key, idx)
        elif kind == "-item" and len(ev) >= 4:
            ledger.record_item(key, tables.item_index(to_id(ev[3])))
        elif kind == "-enditem":
            ledger.record_item(key, tables.item_index(NO_ITEM))
        elif kind == "-ability" and len(ev) >= 4:
            idx = tables.ability_index(to_id(ev[3]), default=None)
            if idx is not None:
                ledger.record_ability(key, idx)
        elif kind == "-terastallize" and len(ev) >= 4:
            ledger.record_tera(key, TYPE_INDEX.get(to_id(ev[3]), 18))
