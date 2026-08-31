"""Reconstruct a poke-engine `State` from a live Showdown position + one belief world (search §8).

This is the piece that lets determinized search run on the SHOWDOWN backend (the gauntlet), not
just on engine self-play: poke-env gives the observable position, the belief chain samples the
opponent's hidden set, and this module fuses them into a full engine state the simulator can step.

Input contract (`view`): any object exposing the poke-env `Battle` surface the encoders already
read — `team` / `opponent_team` / `active_pokemon` / `opponent_active_pokemon` /
`teampreview_opponent_team` / `weather` / `fields` / `side_conditions` /
`opponent_side_conditions` / `turn`, with mons exposing `species / level / stats (own) /
current_hp_fraction / status / boosts / effects / moves / item / ability / terastallized /
tera_type`. `EngineView` (engine_battle) satisfies the SAME surface — that is what makes the
round-trip test possible (engine state -> view -> reconstruct -> compare).

Exactness:
  * Searcher's own side: EXACT — real computed stats (`mon.stats`), moves, ability, item, HP,
    status, boosts come straight from poke-env's own-team knowledge.
  * Opponent side: stats/EVs/nature/moves/item/ability/tera from the SAMPLED WORLD; only
    the public board (HP fraction, status, boosts, revealed tera, active slot) is overlaid.

Documented approximations (the fuzz report — scripts/fuzz_recon.py — quantifies these; none are
fatal, and a stepped ply that trips the engine is caught upstream and the sample skipped):
  * Screen/weather/terrain TURNS-REMAINING: poke-env exposes start turns, not remaining, and the
    source (item vs move vs ability) is hidden — we seed nominal remaining counts.
  * Volatiles: mapped to the engine's known set; substitute HP seeded at 1/4 max when present.
  * Opponent move `disabled` flags: only set from a matched revealed move that reports disabled.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import poke_engine as pe
    HAVE_ENGINE = True
except ImportError:  # pragma: no cover
    pe = None
    HAVE_ENGINE = False

from fog.env.engine_battle import (
    N_SLOTS,
    _decode_world_slot,
    build_engine_pokemon,
    to_id,
)

# showdown status code -> poke-engine status string (inverse of engine_battle._STATUS_MAP)
_SHOWDOWN_TO_ENGINE = {
    "brn": "burn", "par": "paralyze", "psn": "poison",
    "tox": "toxic", "slp": "sleep", "frz": "freeze",
}
# poke-env / EngineView weather + terrain names (id form) -> engine strings
_WEATHER_TO_ENGINE = {"sun": "sun", "harshsun": "sun", "desolateland": "sun",
                      "rain": "rain", "heavyrain": "rain", "primordialsea": "rain",
                      "sand": "sand", "sandstorm": "sand", "snow": "snow", "hail": "snow"}
# engine's known volatile-status vocabulary (engine_battle._active_effects + trapping set); we pass
# through only these so a stray poke-env effect id never derails generate_instructions
_ENGINE_VOLATILES = {
    "confusion", "encore", "taunt", "yawn", "slowstart", "lockedmove", "substitute",
    "partiallytrapped", "leechseed", "protect", "endure", "flinch", "destinybond",
    "attract", "curse", "nightmare", "torment", "disable", "saltcure",
}
_SCREEN_NOMINAL = {"reflect": 5, "light_screen": 5, "aurora_veil": 5, "tailwind": 4}


def _require_engine() -> None:
    if not HAVE_ENGINE:
        raise RuntimeError("poke-engine is not installed: pip install poke-engine")


def _engine_status(view_status: Any) -> str:
    """poke-env Status enum / EngineView showdown-code string -> engine status string ('none' if
    unknown/healthy)."""
    if not view_status:
        return "none"
    code = to_id(getattr(view_status, "name", view_status))
    return _SHOWDOWN_TO_ENGINE.get(code, "none")


def _types_for(mon: Any, feats: dict, tera_now: bool) -> tuple[str, str]:
    tera = to_id(getattr(getattr(mon, "tera_type", None), "name", getattr(mon, "tera_type", None)))
    if tera_now and tera:
        return (tera, "typeless")
    types = [to_id(getattr(t, "name", t)) for t in (getattr(mon, "types", None) or [])]
    types = [t for t in types if t and t != "typeless"] or [str(x) for x in feats["types"][:2]]
    while len(types) < 2:
        types.append("typeless")
    return (types[0], types[1])


def _hp_int(frac: float, maxhp: int, alive: bool) -> int:
    hp = round(float(frac) * maxhp)
    return (max(1, hp) if alive else 0)


def _engine_mon_from_self(mon: Any, tables) -> pe.Pokemon:
    """Searcher's own mon (full knowledge) -> pe.Pokemon with EXACT stats."""
    sid = to_id(getattr(mon, "species", ""))
    feats = tables.species["features"][sid]
    stats = getattr(mon, "stats", None) or {}
    base = np.asarray(feats["base_stats"], dtype=np.int64)
    level = int(getattr(mon, "level", 100) or 100)

    def stat(key: str, default_base_i: int) -> int:
        v = stats.get(key)
        if v:
            return int(v)
        # fallback (own stats missing): neutral-nature, 0-EV estimate — logged as a drift source
        b = int(base[default_base_i])
        if key == "hp":
            return int((2 * b) * level / 100 + level + 10)
        return int((2 * b) * level / 100 + 5)

    maxhp = stat("hp", 0)
    tera_now = bool(getattr(mon, "terastallized", False))
    base_types = tuple(str(t) for t in feats["types"][:2])
    while len(base_types) < 2:
        base_types = (*base_types, "typeless")
    moves = []
    for mid, mv in (getattr(mon, "moves", None) or {}).items():
        moves.append(pe.Move(
            id=to_id(mid), pp=int(getattr(mv, "current_pp", None) or getattr(mv, "max_pp", 16) or 16),
            disabled=bool(getattr(mv, "disabled", False)),
        ))
    tera_id = to_id(getattr(getattr(mon, "tera_type", None), "name",
                            getattr(mon, "tera_type", None)) or "normal")
    item = to_id(getattr(mon, "item", None) or "none")
    return pe.Pokemon(
        id=sid, level=level, types=_types_for(mon, feats, tera_now), base_types=base_types,
        weight_kg=float(feats.get("weightkg") or 50.0),
        hp=_hp_int(getattr(mon, "current_hp_fraction", 1.0), maxhp,
                   not bool(getattr(mon, "fainted", False))),
        maxhp=maxhp,
        attack=stat("atk", 1), defense=stat("def", 2), special_attack=stat("spa", 3),
        special_defense=stat("spd", 4), speed=stat("spe", 5),
        ability=to_id(getattr(mon, "ability", None) or ""),
        item="none" if item in ("", "none", "unknownitem") else item,
        moves=moves, nature="serious",
        status=_engine_status(getattr(mon, "status", None)),
        sleep_turns=int(getattr(mon, "status_counter", 0) or 0),
        terastallized=tera_now, tera_type=tera_id,
    )


def _engine_mon_from_world(row: np.ndarray, view_mon: Any | None, level: int, tables) -> pe.Pokemon:
    """Opponent mon: hidden set from the sampled `row`, public board overlaid from `view_mon`
    (None => never seen: full HP, no status, not terastallized)."""
    base = build_engine_pokemon(_decode_world_slot(row, level, tables), tables)
    feats = tables.species["features"][to_id(base.id)]
    if view_mon is None:
        hp, status, tera_now = base.maxhp, "none", False
        disabled: set[str] = set()
    else:
        tera_now = bool(getattr(view_mon, "terastallized", False))
        hp = _hp_int(getattr(view_mon, "current_hp_fraction", 1.0), base.maxhp,
                     not bool(getattr(view_mon, "fainted", False)))
        status = _engine_status(getattr(view_mon, "status", None))
        disabled = {to_id(mid) for mid, mv in (getattr(view_mon, "moves", None) or {}).items()
                    if bool(getattr(mv, "disabled", False))}
    moves = [pe.Move(id=str(m.id), pp=int(m.pp), disabled=to_id(m.id) in disabled)
             for m in base.moves]
    return pe.Pokemon(
        id=base.id, level=base.level,
        types=_types_for(view_mon, feats, tera_now) if view_mon is not None else tuple(base.types),
        base_types=tuple(str(t) for t in base.types), weight_kg=base.weight_kg,
        hp=hp, maxhp=base.maxhp,
        attack=base.attack, defense=base.defense, special_attack=base.special_attack,
        special_defense=base.special_defense, speed=base.speed,
        evs=tuple(base.evs), nature=base.nature, ability=base.ability, item=base.item,
        moves=moves, status=status,
        sleep_turns=int(getattr(view_mon, "status_counter", 0) or 0) if view_mon is not None else 0,
        terastallized=tera_now, tera_type=base.tera_type,
    )


def _side_conditions(cond_dict: dict, turn: int) -> pe.SideConditions:
    """poke-env side-condition dict (hazard=count, screen=start turn) -> pe.SideConditions."""
    kw: dict[str, int] = {}
    hazards = {"stealthrock": "stealth_rock", "spikes": "spikes",
               "toxicspikes": "toxic_spikes", "stickyweb": "sticky_web"}
    screens = {"reflect": "reflect", "lightscreen": "light_screen",
               "auroraveil": "aurora_veil", "tailwind": "tailwind"}
    for cond, value in (cond_dict or {}).items():
        name = to_id(getattr(cond, "name", cond))
        if name in hazards:
            kw[hazards[name]] = int(value or 1)
        elif name in screens:
            eng = screens[name]
            kw[eng] = max(1, _SCREEN_NOMINAL[eng] - max(0, turn - int(value or turn)))
    return pe.SideConditions(**kw)


def _active_volatiles(mon: Any) -> tuple[set[str], int]:
    """(engine volatile-status set, substitute health) from a view mon's effects dict."""
    vols = {to_id(getattr(e, "name", e)) for e in (getattr(mon, "effects", None) or {})}
    vols &= _ENGINE_VOLATILES
    sub_hp = 1 if "substitute" in vols else 0  # health >0 flags the sub; exact value is hidden
    return vols, sub_hp


def _build_side(mons: list, active_idx: int, active_view: Any | None, cond_dict: dict,
                turn: int) -> pe.Side:
    boosts = getattr(active_view, "boosts", None) or {}
    vols, sub_hp = _active_volatiles(active_view) if active_view is not None else (set(), 0)
    if sub_hp and 0 <= active_idx < len(mons):
        sub_hp = max(1, mons[active_idx].maxhp // 4)
    return pe.Side(
        pokemon=mons, active_index=str(active_idx),
        attack_boost=int(boosts.get("atk", 0)), defense_boost=int(boosts.get("def", 0)),
        special_attack_boost=int(boosts.get("spa", 0)),
        special_defense_boost=int(boosts.get("spd", 0)),
        speed_boost=int(boosts.get("spe", 0)), accuracy_boost=int(boosts.get("accuracy", 0)),
        evasion_boost=int(boosts.get("evasion", 0)),
        side_conditions=_side_conditions(cond_dict, turn),
        volatile_statuses=vols, substitute_health=int(sub_hp),
    )


def _field_kwargs(view: Any, turn: int) -> dict:
    kw: dict[str, Any] = {}
    for w in (getattr(view, "weather", None) or {}):
        eng = _WEATHER_TO_ENGINE.get(to_id(getattr(w, "name", w)))
        if eng:
            kw["weather"], kw["weather_turns_remaining"] = eng, 5
    for f in (getattr(view, "fields", None) or {}):
        name = to_id(getattr(f, "name", f))
        if name.endswith("terrain"):
            kw["terrain"], kw["terrain_turns_remaining"] = name, 5
        elif name == "trickroom":
            kw["trick_room"], kw["trick_room_turns_remaining"] = True, 5
    return kw


def _census_from_view(view: Any, opp_role: str, tables) -> dict:
    """The searcher's current opponent reveal census (id-string form used by EngineView censoring),
    derived from the canonical `opponent_views_from_battle` so leaf fog matches the rest of the
    pipeline exactly."""
    from fog.encoding.reveals import opponent_views_from_battle

    no_item = tables.item_index("<NO_ITEM>")
    census: dict[str, dict] = {}
    for v in opponent_views_from_battle(view, tables):
        if v.species is None or int(v.species) < 0:
            continue
        sid = tables.species["ids"][int(v.species)]
        entry: dict[str, Any] = {"moves": [tables.moves["ids"][int(m)] for m in v.moves]}
        if v.item is not None:
            entry["item"] = "" if int(v.item) == no_item else tables.items["ids"][int(v.item)]
        if v.ability is not None:
            entry["ability"] = tables.abilities["ids"][int(v.ability)]
        census[f"{opp_role}: {sid}"] = entry
    return {opp_role: census, ("p2" if opp_role == "p1" else "p1"): {}}


def reconstruct_battle(
    view: Any, world_tokens: np.ndarray, tables, tracker, *, role: str = "p1", seed: int = 0,
):
    """A steppable `EngineBattle` at the Showdown position `view`, with the opponent set filled in
    from `world_tokens` (one BeliefModel.sample_team world, preview/paste slot order). `tracker` is
    the searcher's live BattleTracker (cloned internally — never mutated). See module docstring."""
    _require_engine()
    from fog.env.engine_battle import EngineBattle

    world_tokens = np.asarray(world_tokens, dtype=np.int64)
    opp_role = "p2" if role == "p1" else "p1"
    turn = int(getattr(view, "turn", 1) or 1)

    # ---- my side (searcher, exact) --------------------------------------------------------------
    my_view_mons = list((getattr(view, "team", None) or {}).values())[:N_SLOTS]
    my_mons = [_engine_mon_from_self(m, tables) for m in my_view_mons]
    my_active_view = getattr(view, "active_pokemon", None)
    my_active = next((i for i, m in enumerate(my_view_mons) if m is my_active_view), 0)
    my_species = [to_id(getattr(m, "species", "")) for m in my_view_mons]

    # ---- opponent side (sampled world + public overlay) -----------------------------------------
    opp_by_species: dict[str, Any] = {}
    for m in (getattr(view, "opponent_team", None) or {}).values():
        opp_by_species[to_id(getattr(m, "species", ""))] = m
    opp_active_view = getattr(view, "opponent_active_pokemon", None)
    opp_active_species = to_id(getattr(opp_active_view, "species", "")) if opp_active_view else None

    opp_mons, opp_species = [], []
    opp_active = 0
    species_ids = tables.species["ids"]
    for j in range(N_SLOTS):
        sp_tok = int(world_tokens[j, 0])
        if not (0 <= sp_tok < len(species_ids)) or species_ids[sp_tok].startswith("<"):
            continue
        sid = species_ids[sp_tok]
        vm = opp_by_species.get(sid)
        level = int(getattr(vm, "level", 100) or 100) if vm is not None else 100
        if opp_active_species is not None and sid == opp_active_species:
            opp_active = len(opp_mons)
        opp_mons.append(_engine_mon_from_world(world_tokens[j], vm, level, tables))
        opp_species.append(sid)

    # ---- assemble sides + field (searcher's actual role decides side_one/side_two) --------------
    my_side = _build_side(my_mons, my_active, my_active_view,
                          getattr(view, "side_conditions", None) or {}, turn)
    opp_side = _build_side(opp_mons, opp_active, opp_active_view,
                           getattr(view, "opponent_side_conditions", None) or {}, turn)
    sides = {role: my_side, opp_role: opp_side}
    state = pe.State(side_one=sides["p1"], side_two=sides["p2"], **_field_kwargs(view, turn))

    tera_used = {
        role: any(bool(getattr(m, "terastallized", False)) for m in my_view_mons),
        opp_role: any(bool(getattr(m, "terastallized", False))
                      for m in (getattr(view, "opponent_team", None) or {}).values()),
    }
    return EngineBattle._from_reconstruction(
        state, role=role, tables=tables, tracker=tracker.clone(),
        revealed=_census_from_view(view, opp_role, tables),
        paste_species={role: my_species, opp_role: opp_species},
        turn=turn, tera_used=tera_used, field_start={}, seed=seed,
    )
