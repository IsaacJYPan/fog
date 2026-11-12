"""Reconstruct a poke-engine `State` from a live Showdown position + one belief world.

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

Sleep: the engine keeps two counters. ``rest_turns`` (3 right after Rest, counting down once per
move attempt, waking at 1) marks sleep from Rest; ``sleep_turns`` counts move attempts made in
ordinary sleep and must stay at or below the engine's Gen 9 maximum of 3 (above it the wake-up
chance divides by zero or goes negative and the engine panics). Rest sleep is recognised from the
public log (``-status ... slp|[from] move: Rest``); an EngineView mon exposes ``rest_turns``.

Terastallization: ``types`` keeps the pre-Tera typing; ``terastallized`` plus ``tera_type`` are what
the engine reads for the Tera same-type bonus, defensive typing and the one-Tera-per-side rule (a
side may Tera only while none of its Pokemon is terastallized, fainted or benched ones included).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fog.encoding.events import event_mon_key
from fog.encoding.pokemon_state import is_terastallized

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
# Keys are ids of poke-env `Weather` names (SUNNYDAY, RAINDANCE, SANDSTORM, SNOWSCAPE, ...) and of
# the engine view's own names (SUN, RAIN, SAND, SNOW).
_WEATHER_TO_ENGINE = {"sun": "sun", "sunnyday": "sun", "harshsun": "sun", "desolateland": "sun",
                      "raindance": "rain", "snowscape": "snow",
                      "rain": "rain", "heavyrain": "rain", "primordialsea": "rain",
                      "sand": "sand", "sandstorm": "sand", "snow": "snow", "hail": "snow"}
# engine's known volatile-status vocabulary (engine_battle._active_effects + trapping set); we pass
# through only these so a stray poke-env effect id never derails generate_instructions
_ENGINE_VOLATILES = {
    "confusion", "encore", "taunt", "yawn", "slowstart", "lockedmove", "substitute",
    "partiallytrapped", "leechseed", "protect", "endure", "flinch", "destinybond",
    "attract", "curse", "nightmare", "torment", "disable", "saltcure",
    # public, announced with `-start`: the boosted stat from Protosynthesis / Quark Drive (sun,
    # Electric Terrain or Booster Energy) and an activated Flash Fire
    "protosynthesisatk", "protosynthesisdef", "protosynthesisspa", "protosynthesisspd",
    "protosynthesisspe", "quarkdriveatk", "quarkdrivedef", "quarkdrivespa", "quarkdrivespd",
    "quarkdrivespe", "flashfire",
}
# poke-env names a binding move by the move itself; the engine has one volatile for all of them
_BINDING_EFFECTS = {"bind", "clamp", "firespin", "infestation", "magmastorm", "sandtomb",
                    "snaptrap", "thundercage", "whirlpool", "wrap"}
_HAZARD_MAX_LAYERS = {"stealthrock": 1, "stickyweb": 1, "spikes": 3, "toxicspikes": 2}
_SCREEN_NOMINAL = {"reflect": 5, "light_screen": 5, "aurora_veil": 5, "tailwind": 4}
# Engine sleep constants (Gen 9 build): Rest sleeps for three move attempts, ordinary sleep wakes
# for certain on the third counted attempt.
_FIELD_NOMINAL_TURNS = 5   # weather, terrain and Trick Room without an extending item
_FIELD_EXTENDED_TURNS = 8  # weather or terrain set by a holder of the matching item
_WEATHER_EXTENDERS = {"rain": "damprock", "sun": "heatrock", "sand": "smoothrock",
                      "snow": "icyrock"}
_REST_SLEEP_TURNS = 3
_MAX_SLEEP_TURNS = 3


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
    """Pre-Tera typing. The engine derives Tera typing from ``terastallized``/``tera_type``, so a
    terastallized mon keeps its species types here (poke-env reports the Tera type as ``types``)."""
    types = [] if tera_now else [to_id(getattr(t, "name", t))
                                 for t in (getattr(mon, "types", None) or [])]
    types = [t for t in types if t and t != "typeless"] or [str(x) for x in feats["types"][:2]]
    while len(types) < 2:
        types.append("typeless")
    return (types[0], types[1])


def _rest_attempts(view: Any) -> dict[str, int]:
    """Public-log scan: mon key ('p1: Name') -> move attempts made since it fell asleep from Rest,
    for every mon whose current sleep came from Rest."""
    from fog.encoding.history import observations_by_turn

    asleep: dict[str, int] = {}
    try:
        obs = observations_by_turn(view)
    except Exception:  # noqa: BLE001 - a view without a log simply has no Rest information
        return asleep
    for t in sorted(obs):
        for ev in getattr(obs[t], "events", None) or []:
            if len(ev) < 3:
                continue
            kind, key = ev[1], event_mon_key(ev[2])
            if kind == "-status":
                if (len(ev) > 3 and to_id(ev[3]) == "slp"
                        and any(to_id(a) == "frommoverest" for a in ev[4:])):
                    asleep[key] = 0
                else:
                    asleep.pop(key, None)
            elif kind in ("-curestatus", "faint"):
                asleep.pop(key, None)
            elif key in asleep and (
                (kind == "cant" and len(ev) > 3 and to_id(ev[3]) == "slp")
                or (kind == "move" and not any(str(a).startswith("[from]") for a in ev[4:]))
            ):
                asleep[key] += 1
    return asleep


def _tera_from_log(view: Any) -> dict[str, str]:
    """Public-log scan: mon key ('p1: Name') -> Tera type for every ``-terastallize`` event. This
    backs up the parser's per-mon flag so a side's used Tera is never lost from the rebuilt state
    (the engine offers Tera to a side while none of its Pokemon is terastallized)."""
    from fog.encoding.history import observations_by_turn

    out: dict[str, str] = {}
    try:
        obs = observations_by_turn(view)
    except Exception:  # noqa: BLE001 - a view without a log falls back to the parser flags
        return out
    for t in sorted(obs):
        for ev in getattr(obs[t], "events", None) or []:
            if len(ev) >= 4 and ev[1] == "-terastallize":
                out[event_mon_key(ev[2])] = to_id(ev[3])
    return out


def _pre_tera_species(sid: str, tables) -> str:
    """'ogerponwellspringtera' -> 'ogerponwellspring': sampled worlds and team preview use the
    pre-Tera forme, while the battle log switches the species on Terastallization."""
    if sid.endswith("tera") and sid[:-4] in tables.species["features"]:
        return sid[:-4]
    return sid


def _sleep_fields(mon: Any, status: str, rest_attempts: int | None) -> dict[str, int]:
    """Engine ``sleep_turns`` / ``rest_turns`` for a view mon with engine status `status`."""
    if status != "sleep":
        return {"sleep_turns": 0, "rest_turns": 0}
    rest = getattr(mon, "rest_turns", None)  # EngineView mons carry the engine counter directly
    if rest is None and rest_attempts is not None:
        rest = max(1, _REST_SLEEP_TURNS - int(rest_attempts))
    if rest:
        return {"sleep_turns": 0, "rest_turns": min(_REST_SLEEP_TURNS, max(1, int(rest)))}
    counter = int(getattr(mon, "status_counter", 0) or 0)
    return {"sleep_turns": min(_MAX_SLEEP_TURNS, max(0, counter)), "rest_turns": 0}


def _move_pp(view_move: Any, default: int) -> int:
    """Current PP of a view move: 0 stays 0; unknown falls back to max PP, then `default`."""
    pp = getattr(view_move, "current_pp", None)
    if pp is None:
        pp = getattr(view_move, "max_pp", None)
    return int(default if pp is None else pp)


def _own_tera_type(mon: Any) -> str:
    """Own mon's Tera type. poke-env only learns it on use, so read the request's ``teraType``."""
    tera = getattr(getattr(mon, "tera_type", None), "name", getattr(mon, "tera_type", None))
    if not tera:
        tera = (getattr(mon, "_last_request", None) or {}).get("teraType")
    return to_id(tera) or "normal"


def _hp_int(frac: float, maxhp: int, alive: bool) -> int:
    hp = round(float(frac) * maxhp)
    return (max(1, hp) if alive else 0)


def _engine_mon_from_self(mon: Any, tables, rest_attempts: int | None = None,
                          logged_tera: str | None = None) -> pe.Pokemon:
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
    tera_now = is_terastallized(mon) or bool(logged_tera)
    base_types = tuple(str(t) for t in feats["types"][:2])
    while len(base_types) < 2:
        base_types = (*base_types, "typeless")
    moves = []
    for mid, mv in (getattr(mon, "moves", None) or {}).items():
        moves.append(pe.Move(
            id=to_id(mid), pp=_move_pp(mv, 16), disabled=bool(getattr(mv, "disabled", False)),
        ))
    tera_id = logged_tera or _own_tera_type(mon)
    status = _engine_status(getattr(mon, "status", None))
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
        moves=moves, nature="serious", status=status,
        **_sleep_fields(mon, status, rest_attempts),
        terastallized=tera_now, tera_type=tera_id,
    )


def _engine_mon_from_world(row: np.ndarray, view_mon: Any | None, level: int, tables,
                           rest_attempts: int | None = None,
                           logged_tera: str | None = None) -> pe.Pokemon:
    """Opponent mon: hidden set from the sampled `row`, public board overlaid from `view_mon`
    (None => never seen: full HP, no status, not terastallized)."""
    base = build_engine_pokemon(_decode_world_slot(row, level, tables), tables)
    feats = tables.species["features"][to_id(base.id)]
    if view_mon is None:
        hp, status, tera_now = base.maxhp, "none", False
        disabled: set[str] = set()
        seen_moves: dict[str, Any] = {}
    else:
        tera_now = is_terastallized(view_mon) or bool(logged_tera)
        hp = _hp_int(getattr(view_mon, "current_hp_fraction", 1.0), base.maxhp,
                     not bool(getattr(view_mon, "fainted", False)))
        status = _engine_status(getattr(view_mon, "status", None))
        seen_moves = {to_id(mid): mv for mid, mv in (getattr(view_mon, "moves", None) or {}).items()}
        disabled = {mid for mid, mv in seen_moves.items() if bool(getattr(mv, "disabled", False))}
    # A revealed move's observed PP is public; only values below the sampled default matter (the
    # engine tracks PP from 10 down, and 0 PP removes the move from the options).
    moves = [pe.Move(id=str(m.id),
                     pp=min(int(m.pp), _move_pp(seen_moves[to_id(m.id)], int(m.pp)))
                     if to_id(m.id) in seen_moves else int(m.pp),
                     disabled=to_id(m.id) in disabled)
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
        **(_sleep_fields(view_mon, status, rest_attempts) if view_mon is not None else {}),
        # a used Tera type is public: it overrides the sampled one
        terastallized=tera_now, tera_type=(_revealed_tera(view_mon, logged_tera) if tera_now
                                           else None) or base.tera_type,
    )


def _revealed_tera(view_mon: Any, logged_tera: str | None) -> str | None:
    tera = getattr(view_mon, "tera_type", None)
    return logged_tera or to_id(getattr(tera, "name", tera)) or None


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
            # poke-env stores a layer count only for the stackable hazards; for Stealth Rock and
            # Sticky Web the value is the turn they were set. The engine applies those two only
            # when the field is exactly 1, so a turn number silently removed them from the search.
            layers = _HAZARD_MAX_LAYERS[name]
            kw[hazards[name]] = 1 if layers == 1 else max(1, min(layers, int(value or 1)))
        elif name in screens:
            eng = screens[name]
            kw[eng] = max(1, _SCREEN_NOMINAL[eng] - max(0, turn - int(value or turn)))
    return pe.SideConditions(**kw)


def _active_volatiles(mon: Any) -> tuple[set[str], int]:
    """(engine volatile-status set, substitute health) from a view mon's effects dict."""
    vols = {to_id(getattr(e, "name", e)) for e in (getattr(mon, "effects", None) or {})}
    if vols & _BINDING_EFFECTS:
        vols.add("partiallytrapped")
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


def _turns_left(start: Any, turn: int, nominal: int = _FIELD_NOMINAL_TURNS) -> int:
    """Turns remaining for a field effect that began on turn `start` (poke-env stores the start
    turn as the dict value). The duration is the nominal five turns: an extending item (Heat Rock,
    Terrain Extender, ...) is hidden information, so it is not assumed. Never below 1 while the
    effect is still on the field; an unknown start falls back to the full nominal duration."""
    try:
        elapsed = int(turn) - int(start)
    except (TypeError, ValueError):
        return nominal
    return max(1, nominal - max(0, elapsed))


def _field_clocks(view: Any) -> dict[str, tuple[int, str | None]] | None:
    """Public-log scan for the weather, terrain and Trick Room now on the field (keys ``weather``,
    ``terrain``, ``trickroom``): ``(upkeeps passed since it began, mon key of whoever set it)``.

    Showdown prints ``|upkeep|`` after each end-of-turn residual step, which is when these timers
    tick, so counting upkeeps is exact where the start turn alone is off by one (an effect set by a
    lead or by a replacement after a faint has not ticked yet). The setter ('p1: Pelipper') lets
    the caller apply an extending item. Returns None when the log has no upkeep markers to count
    (the engine backend's synthesized log), so the caller falls back to the start turn."""
    from fog.encoding.history import observations_by_turn

    try:
        obs = observations_by_turn(view)
    except Exception:  # noqa: BLE001 - a view without a log falls back to start turns
        return None
    clocks: dict[str, tuple[int, str | None]] = {}
    saw_upkeep = False
    last_turn = 0
    for t in sorted(obs):
        last_turn = max(last_turn, int(t))
        last_mover: str | None = None
        for ev in getattr(obs[t], "events", None) or []:
            if len(ev) < 2:
                continue
            kind = ev[1]
            if kind == "upkeep":
                saw_upkeep = True
                clocks = {key: (count + 1, who) for key, (count, who) in clocks.items()}
                continue
            if kind == "move" and len(ev) > 2:
                last_mover = event_mon_key(ev[2])
                continue
            name = to_id(ev[2]) if len(ev) > 2 else ""
            source = next((event_mon_key(str(a)[len("[of] "):]) for a in ev[3:]
                           if str(a).startswith("[of] ")), last_mover)
            if kind == "-weather":
                if name in ("none", ""):
                    clocks.pop("weather", None)
                elif not any(str(a).startswith("[upkeep]") for a in ev[3:]):
                    clocks["weather"] = (0, source)
            elif kind in ("-fieldstart", "-fieldend"):
                key = ("terrain" if name.endswith("terrain") else
                       "trickroom" if name.endswith("trickroom") else None)
                if key is None:
                    continue
                if kind == "-fieldstart":
                    clocks[key] = (0, source)
                else:
                    clocks.pop(key, None)
    return clocks if saw_upkeep or last_turn <= 1 else None


def _field_kwargs(view: Any, turn: int, items: dict[str, str] | None = None) -> dict:
    """Engine field arguments. `items` maps a mon key ('p1: Pelipper') to its item id in the state
    being built (the searcher's real item, or the sampled world's item for an opponent), so
    weather or terrain set by a holder of the matching extending item lasts eight turns."""
    kw: dict[str, Any] = {}
    clocks = _field_clocks(view)

    def left(key: str, start: Any, extender: str | None = None) -> int:
        if clocks is None or key not in clocks:
            return _turns_left(start, turn)
        upkeeps, setter = clocks[key]
        extended = extender is not None and to_id((items or {}).get(setter or "", "")) == extender
        nominal = _FIELD_EXTENDED_TURNS if extended else _FIELD_NOMINAL_TURNS
        return max(1, nominal - upkeeps)

    for w, start in (getattr(view, "weather", None) or {}).items():
        eng = _WEATHER_TO_ENGINE.get(to_id(getattr(w, "name", w)))
        if eng:
            kw["weather"] = eng
            kw["weather_turns_remaining"] = left("weather", start, _WEATHER_EXTENDERS.get(eng))
    for f, start in (getattr(view, "fields", None) or {}).items():
        name = to_id(getattr(f, "name", f))
        if name.endswith("terrain"):
            kw["terrain"] = name
            kw["terrain_turns_remaining"] = left("terrain", start, "terrainextender")
        elif name == "trickroom":
            kw["trick_room"] = True
            kw["trick_room_turns_remaining"] = left("trickroom", start)
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
    rest = _rest_attempts(view)
    tera_log = _tera_from_log(view)
    my_items = list((getattr(view, "team", None) or {}).items())[:N_SLOTS]
    my_view_mons = [m for _, m in my_items]
    my_mons = [_engine_mon_from_self(m, tables, rest.get(str(k)), tera_log.get(str(k)))
               for k, m in my_items]
    my_active_view = getattr(view, "active_pokemon", None)
    my_active = next((i for i, m in enumerate(my_view_mons) if m is my_active_view), 0)
    my_species = [to_id(getattr(m, "species", "")) for m in my_view_mons]

    # ---- opponent side (sampled world + public overlay) -----------------------------------------
    opp_by_species: dict[str, Any] = {}
    opp_rest: dict[str, int | None] = {}
    opp_tera: dict[str, str | None] = {}
    for k, m in (getattr(view, "opponent_team", None) or {}).items():
        sid = _pre_tera_species(to_id(getattr(m, "species", "")), tables)
        opp_by_species[sid] = m
        opp_rest[sid] = rest.get(str(k))
        opp_tera[sid] = tera_log.get(str(k))
    opp_active_view = getattr(view, "opponent_active_pokemon", None)
    opp_active_species = (_pre_tera_species(to_id(getattr(opp_active_view, "species", "")), tables)
                          if opp_active_view else None)

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
        opp_mons.append(_engine_mon_from_world(world_tokens[j], vm, level, tables,
                                               opp_rest.get(sid), opp_tera.get(sid)))
        opp_species.append(sid)

    # ---- assemble sides + field (searcher's actual role decides side_one/side_two) --------------
    my_side = _build_side(my_mons, my_active, my_active_view,
                          getattr(view, "side_conditions", None) or {}, turn)
    opp_side = _build_side(opp_mons, opp_active, opp_active_view,
                           getattr(view, "opponent_side_conditions", None) or {}, turn)
    sides = {role: my_side, opp_role: opp_side}
    # items as they stand in this rebuilt state, keyed like the public log ('p1: Pelipper')
    items = {str(k): str(mon.item) for (k, _), mon in zip(my_items, my_mons)}
    for k, m in (getattr(view, "opponent_team", None) or {}).items():
        sid = _pre_tera_species(to_id(getattr(m, "species", "")), tables)
        if sid in opp_species:
            items[str(k)] = str(opp_mons[opp_species.index(sid)].item)
    state = pe.State(side_one=sides["p1"], side_two=sides["p2"],
                     **_field_kwargs(view, turn, items))

    tera_used = {
        role: any(bool(m.terastallized) for m in my_mons)
        or any(is_terastallized(m) for m in my_view_mons),
        opp_role: any(bool(m.terastallized) for m in opp_mons)
        or any(is_terastallized(m) for m in (getattr(view, "opponent_team", None) or {}).values()),
    }
    return EngineBattle._from_reconstruction(
        state, role=role, tables=tables, tracker=tracker.clone(),
        revealed=_census_from_view(view, opp_role, tables),
        paste_species={role: my_species, opp_role: opp_species},
        turn=turn, tera_used=tera_used, field_start={}, seed=seed,
    )
