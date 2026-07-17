"""poke-engine battle wrapper: partial observability + event synthesis (ARCHITECTURE.md §3.2).

`EngineBattle` owns one poke-engine game and everything the perfect-information engine cannot:
what each side has *seen*. It synthesizes pseudo-protocol events (the same ``['', 'move', ...]``
shape poke-env exposes) from three sources — the chosen actions, the applied instruction reprs,
and pre/post state diffs — and feeds them to the UNCHANGED downstream stack: `BattleTracker`,
`RevealLedger` (via `feed_ledger_from_events`), and `parse_turn_events` through
`Observation.from_battle` on the censored views this module builds. Nothing downstream knows it
is not talking to Showdown.

Engine facts this module is built on (probed against poke-engine 0.0.47 — re-verify on upgrade,
see tests/test_engine_battle.py::test_engine_contract):
  * ``state.apply_instructions(chosen)`` RETURNS A NEW STATE — it does not mutate. (The original
    adapter and fuzz harness discarded the result; their games never advanced.)
  * ``pe.Pokemon`` computes NOTHING from id/evs/nature: stats, types and weight_kg must be passed
    explicitly or every mon plays with 100s and NORMAL/TYPELESS typing. `build_engine_pokemon`
    computes real stats via constants.build_spreads.nature_modified_stats (level + IVs honored).
  * ``PyInstruction`` exposes no attributes — only its repr ("Damage SideOne: 92"). The event
    synthesis parses these reprs; unknown kinds are counted in `unknown_instructions`, never fatal.
  * Choice lock / Encore / Disable arrive as DisableMove/EnableMove instructions and are readable
    as ``move.disabled`` — legality reads engine state directly.
  * Faint: no Faint instruction; detect hp<=0. The fainted side sends a bare pokemon id next call,
    the opponent sends "none". Pivot (U-turn): ToggleSide*ForceSwitch + the opponent's saved move
    fires after the replacement (engine-internal).
  * One tera per battle per side is NOT enforced by the engine — `EngineBattle` masks it.
  * PP never decrements (no Struggle) and switch-trapping abilities (Shadow Tag / Arena Trap) are
    not modeled: documented divergences the cross-sim fuzz must watch (KNOWN_ISSUES).
  * Lead abilities do not activate at State construction; `_lead_patches` seeds weather/terrain/
    Intimidate for the leads (slower side wins conflicting weather, as on cartridge) and records
    the ability reveals.

Reveal semantics are CONSERVATIVE by design (plan §3): when an instruction cannot be attributed,
we reveal less, never more — under-revealing keeps labels censored (a supported path everywhere);
over-revealing would leak private choices into the belief input. Known under-reveals: passive item
proof (Leftovers heals are unattributed), most ability activations (only weather/terrain setters
and Intimidate are inferred). Known over-reveal: a chosen move blocked by flinch/full-para still
emits its move event (no instruction evidence distinguishes it from a no-op status move).
"""
from __future__ import annotations

import random
import re
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import poke_engine as pe
    HAVE_ENGINE = True
except ImportError:  # pragma: no cover
    pe = None
    HAVE_ENGINE = False

N_SLOTS = 6
_NON_ID = re.compile(r"[^a-z0-9]")

# switch-in abilities the engine only fires on a real Switch instruction; seeded for leads
_LEAD_WEATHER = {"drought": "sun", "drizzle": "rain", "sandstream": "sand", "snowwarning": "snow"}
_LEAD_TERRAIN = {
    "electricsurge": "electricterrain", "grassysurge": "grassyterrain",
    "mistysurge": "mistyterrain", "psychicsurge": "psychicterrain",
}
# engine status strings -> Showdown codes (encoders key Status[name.upper()])
_STATUS_MAP = {
    "burn": "brn", "paralyze": "par", "paralyzed": "par", "poison": "psn",
    "toxic": "tox", "sleep": "slp", "freeze": "frz", "frozen": "frz", "none": None,
}
# volatiles that forbid voluntary switching (engine allows the switch; Showdown does not)
_NO_SWITCH_VOLATILES = {"partiallytrapped", "lockedmove"}

_INSTR = {
    "damage": re.compile(r"^Damage Side(One|Two): (-?\d+)$"),
    "heal": re.compile(r"^Heal Side(One|Two): (-?\d+)$"),
    "switch": re.compile(r"^Switch Side(One|Two): P(\d) -> P(\d)$"),
    "item": re.compile(r"^ChangeItem Side(One|Two): (\w+) -> (\w+)$"),
    "volatile+": re.compile(r"^ApplyVolatileStatus Side(One|Two): (\w+)$"),
    "volatile-": re.compile(r"^RemoveVolatileStatus Side(One|Two): (\w+)$"),
    "tera": re.compile(r"^ToggleTerastallized Side(One|Two)$"),
    "weather": re.compile(r"^ChangeWeather: \w+,-?\d+ -> (\w+),(-?\d+)$"),
    "terrain": re.compile(r"^ChangeTerrain: \w+,-?\d+ -> (\w+),(-?\d+)$"),
    "boost": re.compile(r"^Boost Side(One|Two) (\w+): (-?\d+)$"),
    "forceswitch": re.compile(r"^ToggleSide(One|Two)ForceSwitch$"),
}
# instruction kinds we deliberately ignore (state-side effects the views read directly)
_IGNORED = re.compile(
    r"^(ChangeStatus|ChangeSideCondition|SetLastUsedMove|ChangeVolatileStatusDuration|"
    r"DisableMove|EnableMove|SetWish|DecrementWish|DecrementTerrainTurnsRemaining|"
    r"DecrementWeatherTurnsRemaining|Side(One|Two)MoveSecondSwitchOutMove|SetSecondMove|"
    r"ChangeType|ChangeAbility|SetSubstituteHealth|DamageSubstitute|ChangeSubstituteHealth|"
    r"SetRestTurns|SetSleepTurns|DecrementRestTurns|ToggleBatonPassing|ToggleShedTailing|"
    r"ToggleTrickRoom|IncrementFutureSight|DecrementFutureSight|SetFutureSight)"
)


def to_id(x: object) -> str:
    return _NON_ID.sub("", str(x).lower())


def _require_engine() -> None:
    if not HAVE_ENGINE:
        raise RuntimeError("poke-engine is not installed: pip install poke-engine")


# --------------------------------------------------------------------------- paste -> engine mons
def parse_paste(paste: str) -> list:
    """Showdown paste -> poke-env TeambuilderPokemon list (<=6)."""
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self):  # pragma: no cover
            return ""

    return _T().parse_showdown_team(paste)[:N_SLOTS]


def build_engine_pokemon(mon: Any, tables) -> "pe.Pokemon":
    """TeambuilderPokemon -> pe.Pokemon with REAL computed stats/types/weight (module docstring)."""
    _require_engine()
    from fog.constants.build_spreads import nature_modified_stats

    sid = to_id(mon.species or mon.nickname)
    feats = tables.species["features"].get(sid)
    if feats is None:
        raise ValueError(
            f"species {sid!r} not in the generated tables — engine self-play requires table-backed "
            "teams (the belief/encoding stack needs them too; validate the pool first)."
        )
    base = np.asarray(feats["base_stats"], dtype=np.int64)
    evs = np.asarray([int(v or 0) for v in (mon.evs or [0] * 6)], dtype=np.int64)
    ivs = np.asarray([31 if v is None else int(v) for v in (mon.ivs or [31] * 6)], dtype=np.int64)
    level = int(mon.level or 100)
    stats = nature_modified_stats(base, evs, mon.nature or "serious", level=level, ivs=ivs)
    types = [str(t) for t in feats["types"]][:2]
    while len(types) < 2:
        types.append("typeless")
    return pe.Pokemon(
        id=sid,
        level=level,
        types=tuple(types),
        weight_kg=float(feats.get("weightkg") or 50.0),
        hp=int(stats[0]), maxhp=int(stats[0]),
        attack=int(stats[1]), defense=int(stats[2]),
        special_attack=int(stats[3]), special_defense=int(stats[4]), speed=int(stats[5]),
        evs=tuple(int(v) for v in evs),
        nature=to_id(mon.nature or "serious"),
        ability=to_id(mon.ability or ""),
        item=to_id(mon.item or "none"),
        moves=[pe.Move(id=to_id(m)) for m in (mon.moves or []) if m][:4],
        tera_type=to_id(getattr(mon, "tera_type", None) or "normal"),
    )


# ------------------------------------------------------------- belief-world tokens -> engine mons
def _species_id_from_token(idx: int, tables) -> str:
    from fog.constants.schema import RARE_SPECIES

    if not 0 <= int(idx) < tables.n_species:
        raise ValueError(f"species token {idx} out of range")
    sid = tables.species["ids"][int(idx)]
    if sid == RARE_SPECIES:
        raise ValueError("cannot instantiate <RARE_SPECIES> in an engine world")
    return sid


def _decode_world_slot(row: np.ndarray, level: int, tables) -> SimpleNamespace:
    """One [9] belief token row (SLOT_ATTRS order) -> a TeambuilderPokemon-shaped namespace for
    `build_engine_pokemon`. Spread cluster supplies nature+EVs; sentinel items map to no item."""
    from fog.constants.schema import NO_ITEM, OTHER_ITEM, TYPES, UNKNOWN_ITEM

    sid = _species_id_from_token(row[0], tables)

    item = None
    if 0 <= int(row[1]) < tables.n_items:
        iid = tables.items["ids"][int(row[1])]
        if iid not in (NO_ITEM, OTHER_ITEM, UNKNOWN_ITEM):
            item = iid

    ability = ""
    if 0 <= int(row[2]) < tables.n_abilities:
        ability = tables.abilities["ids"][int(row[2])]

    moves: list[str] = []
    for t in row[3:7]:
        if 0 <= int(t) < tables.n_moves:
            mid = tables.moves["ids"][int(t)]
            if mid not in moves:
                moves.append(mid)

    clusters = (tables.spreads.get("species", {}) or {}).get(sid, [])
    c = clusters[int(row[7])] if 0 <= int(row[7]) < len(clusters) else (
        clusters[0] if clusters else None
    )
    nature = (c or {}).get("nature", "serious")
    evs = list((c or {}).get("evs", [0] * 6))

    tera = TYPES[int(row[8])] if 0 <= int(row[8]) < len(TYPES) else "normal"
    return SimpleNamespace(
        species=sid, nickname=None, level=level, evs=evs, ivs=None,
        nature=nature, ability=ability, item=item, moves=moves, tera_type=tera,
    )


def _world_engine_mon(row: np.ndarray, src: Any, tables) -> "pe.Pokemon":
    """Rebuild one opponent mon from a sampled token row, overlaying `src`'s PUBLIC board state
    (HP fraction, status, sleep/rest turns, terastallized). Hidden attributes come exclusively
    from the sample — never from `src`.

    Per-move `disabled` flags ARE carried over from `src` (a move can only be choice-locked /
    Encored / Disabled after it has been used, i.e. revealed, so it is always present in the
    sampled set): this preserves choice-lock and Disable across the fork. PP is not carried —
    poke-engine never decrements PP (documented engine limitation), so root and fork agree at 16.
    """
    base = build_engine_pokemon(_decode_world_slot(row, int(src.level), tables), tables)
    disabled_ids = {to_id(m.id) for m in (src.moves or []) if bool(getattr(m, "disabled", False))}
    moves = [
        pe.Move(id=str(m.id), pp=int(m.pp), disabled=to_id(m.id) in disabled_ids)
        for m in base.moves
    ]
    frac = (src.hp / src.maxhp) if src.maxhp else 0.0
    hp = int(round(frac * base.maxhp))
    hp = max(1, hp) if src.hp > 0 else 0
    tera_now = bool(src.terastallized)
    base_types = tuple(str(t) for t in base.types)
    types = (to_id(base.tera_type), "typeless") if tera_now else base_types
    return pe.Pokemon(
        id=base.id, level=base.level, types=types, base_types=base_types,
        weight_kg=base.weight_kg,
        hp=hp, maxhp=base.maxhp,
        attack=base.attack, defense=base.defense, special_attack=base.special_attack,
        special_defense=base.special_defense, speed=base.speed,
        evs=tuple(base.evs), nature=base.nature,
        ability=base.ability, item=base.item, moves=moves,
        status=str(src.status).lower(), rest_turns=int(src.rest_turns),
        sleep_turns=int(src.sleep_turns),
        terastallized=tera_now, tera_type=base.tera_type,
    )


def _rebuild_side(side: Any, mons: list) -> "pe.Side":
    """New pe.Side with `mons` and every public side attribute copied from `side`."""
    return pe.Side(
        pokemon=mons,
        active_index=str(side.active_index),
        attack_boost=int(side.attack_boost), defense_boost=int(side.defense_boost),
        special_attack_boost=int(side.special_attack_boost),
        special_defense_boost=int(side.special_defense_boost),
        speed_boost=int(side.speed_boost),
        accuracy_boost=int(side.accuracy_boost), evasion_boost=int(side.evasion_boost),
        side_conditions=side.side_conditions,
        volatile_statuses={str(v) for v in (side.volatile_statuses or ())},
        volatile_status_durations=side.volatile_status_durations,
        substitute_health=int(side.substitute_health or 0),
        force_switch=bool(side.force_switch), force_trapped=bool(side.force_trapped),
        baton_passing=bool(side.baton_passing), shed_tailing=bool(side.shed_tailing),
        slow_uturn_move=bool(side.slow_uturn_move),
        wish=side.wish, future_sight=side.future_sight,
        last_used_move=str(side.last_used_move),
        switch_out_move_second_saved_move=str(side.switch_out_move_second_saved_move),
    )


# ------------------------------------------------------------------------------- censored views
class _ViewMove:
    def __init__(self, move_id: str, pp: int = 16, max_pp: int = 16):
        self.id = move_id
        self.current_pp = pp
        self.max_pp = max_pp


class _ViewMon:
    """Duck-typed mon satisfying fog.encoding.* / env.actions. `revealed` = per-attribute census
    (None => full self-knowledge; a dict => opponent view built from that side's reveals)."""

    def __init__(self, ep: Any, feats: dict, revealed: dict | None = None):
        rev = revealed  # None = my own mon
        self.species = to_id(ep.id)
        if rev is None:
            item = to_id(ep.item)
            self.item = None if item in ("", "none") else item
            self.ability = to_id(ep.ability) or None
            self.moves = {to_id(m.id): _ViewMove(to_id(m.id), int(m.pp), max(int(m.pp), 1))
                          for m in (ep.moves or [])}
            self.tera_type = to_id(ep.tera_type)
        else:
            # item: unknown_item = never revealed; "" = revealed-gone (<NO_ITEM>); else the id
            item = rev.get("item", "unknown_item")
            self.item = item if item != "" else ""
            self.ability = rev.get("ability")
            self.moves = {m: _ViewMove(m) for m in rev.get("moves", ())}
            self.tera_type = to_id(ep.tera_type) if bool(ep.terastallized) else None
        self.terastallized = bool(ep.terastallized)

        self.level = int(ep.level)
        # exact COMPUTED stats — own mons only (opp stats are hidden; leaving them None avoids any
        # leak). Read by state_recon for the searcher's own side; the encoders never touch it.
        self.stats = (
            {"hp": int(ep.maxhp), "atk": int(ep.attack), "def": int(ep.defense),
             "spa": int(ep.special_attack), "spd": int(ep.special_defense), "spe": int(ep.speed)}
            if rev is None else None
        )
        self.base_stats = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), feats["base_stats"]))
        self.weight = float(feats.get("weightkg") or 50.0)
        self.types = [str(t) for t in feats["types"]]
        self.current_hp_fraction = (ep.hp / ep.maxhp) if ep.maxhp else 0.0
        self.fainted = ep.hp <= 0
        status = to_id(getattr(ep, "status", "none") or "none")
        self.status = _STATUS_MAP.get(status, status) if status else None
        self.status_counter = int(getattr(ep, "sleep_turns", 0) or 0)
        self.boosts: dict[str, int] = {}
        self.effects: dict[str, int] = {}


def _side_boosts(side) -> dict:
    return {
        "atk": int(side.attack_boost), "def": int(side.defense_boost),
        "spa": int(side.special_attack_boost), "spd": int(side.special_defense_boost),
        "spe": int(side.speed_boost), "accuracy": int(side.accuracy_boost),
        "evasion": int(side.evasion_boost),
    }


def _active_effects(side) -> dict[str, int]:
    """Engine side volatiles -> the active mon's effects dict (names lowercased for VOLATILES)."""
    effects = {to_id(v): 0 for v in (side.volatile_statuses or ())}
    vsd = side.volatile_status_durations
    for name in ("confusion", "encore", "taunt", "yawn", "slowstart", "lockedmove"):
        dur = int(getattr(vsd, name, 0) or 0)
        if dur and name in effects:
            effects[name] = dur
    if int(side.substitute_health or 0) > 0:
        effects.setdefault("substitute", 0)
    return effects


class EngineView:
    """A poke-env-battle-shaped view of one EngineBattle from `role`'s perspective.

    Feed it to Observation.from_battle / env.actions.* — the exact code paths the Showdown player
    uses. The opponent's team is censored from THAT side's recorded reveals (never engine truth).
    """

    def __init__(self, battle: "EngineBattle", role: str):
        self.player_role = role
        self.battle_tag = battle.battle_tag
        self.turn = battle.turn
        self.observations = battle.observations
        state = battle.state
        my_side = state.side_one if role == "p1" else state.side_two
        opp_side = state.side_two if role == "p1" else state.side_one
        opp_role = "p2" if role == "p1" else "p1"
        tables = battle.tables

        def feats(ep):
            return tables.species["features"][to_id(ep.id)]

        # engine states round-trip with placeholder "none" mons padding short sides — skip them
        my_mons = [p for p in my_side.pokemon if to_id(p.id) not in ("", "none")]
        opp_mons = [p for p in opp_side.pokemon if to_id(p.id) not in ("", "none")]
        self.team = {f"{role}: {to_id(p.id)}": _ViewMon(p, feats(p)) for p in my_mons}
        opp_rev = battle.revealed[opp_role]
        self.opponent_team = {
            f"{opp_role}: {to_id(p.id)}": _ViewMon(
                p, feats(p), revealed=opp_rev.get(f"{opp_role}: {to_id(p.id)}", {})
            )
            for p in opp_mons
        }
        # preview slot order = PASTE order (matches truth_tokens_from_paste + belief slot layout)
        by_id = {
            to_id(p.id): self.opponent_team[f"{opp_role}: {to_id(p.id)}"] for p in opp_mons
        }
        self.teampreview_opponent_team = [
            by_id[s] for s in battle.paste_species[opp_role] if s in by_id
        ]

        mine = list(self.team.values())
        theirs = list(self.opponent_team.values())
        my_active = int(my_side.active_index)
        opp_active = int(opp_side.active_index)
        self.active_pokemon = mine[my_active] if mine else None
        self.opponent_active_pokemon = theirs[opp_active] if theirs else None
        if self.active_pokemon is not None:
            self.active_pokemon.boosts = _side_boosts(my_side)
            self.active_pokemon.effects = _active_effects(my_side)
        if self.opponent_active_pokemon is not None:
            self.opponent_active_pokemon.boosts = _side_boosts(opp_side)
            self.opponent_active_pokemon.effects = _active_effects(opp_side)

        # ---- legality inputs (env.actions reads these; the mask falls out of them) ----------------
        forced = battle.needs_replacement(role) or bool(my_side.force_switch)
        self.force_switch = forced
        trapped = bool(my_side.force_trapped)
        if not trapped and not forced:
            vols = {to_id(v) for v in (my_side.volatile_statuses or ())}
            if vols & _NO_SWITCH_VOLATILES:
                # Ghost types can always leave a partial trap (Gen 6+); locked move still traps
                is_ghost = self.active_pokemon is not None and "ghost" in self.active_pokemon.types
                trapped = not (is_ghost and "lockedmove" not in vols)
        self.available_moves = (
            []
            if forced or self.active_pokemon is None or self.active_pokemon.fainted
            else [
                self.active_pokemon.moves[to_id(m.id)]
                for m in my_side.pokemon[my_active].moves
                if not m.disabled and int(m.pp) > 0 and to_id(m.id) in self.active_pokemon.moves
            ]
        )
        self.available_switches = (
            []
            if (trapped and not forced)
            else [m for m in mine if m is not self.active_pokemon and not m.fainted]
        )
        self.can_tera = not battle.tera_used[role] and not forced

        # ---- field ---------------------------------------------------------------------------------
        weather = str(state.weather)
        self.weather = (
            {} if to_id(weather) in ("none", "") else {weather: battle.field_start.get("weather", 0)}
        )
        self.fields: dict[str, int] = {}
        terrain = str(state.terrain)
        if to_id(terrain) not in ("none", ""):
            self.fields[terrain] = battle.field_start.get("terrain", 0)
        if bool(state.trick_room):
            self.fields["trickroom"] = battle.field_start.get("trickroom", 0)
        my_sc, opp_sc = my_side.side_conditions, opp_side.side_conditions
        self.side_conditions = _side_conditions(my_sc, self.turn)
        self.opponent_side_conditions = _side_conditions(opp_sc, self.turn)

        won = battle.winner
        self.won = None if won is None or not battle.finished else (won == role)


_SC_HAZARDS = ("spikes", "toxic_spikes", "stealth_rock", "sticky_web")
_SC_SCREENS = {"reflect": 5, "light_screen": 5, "aurora_veil": 5, "tailwind": 4}


def _side_conditions(sc, now: int) -> dict:
    """Engine SideConditions -> poke-env-shaped dict. Hazards store the stack count; screens store
    a SYNTHETIC start turn (poke-env convention) derived from the engine's turns-remaining."""
    out: dict[str, int] = {}
    for name in _SC_HAZARDS:
        v = int(getattr(sc, name, 0) or 0)
        if v:
            out[name.replace("_", "")] = v
    for name, dur in _SC_SCREENS.items():
        v = int(getattr(sc, name, 0) or 0)
        if v:
            out[name.replace("_", "")] = max(0, now - (dur - v))
    return out


# ----------------------------------------------------------------------------------- the battle
class EngineBattle:
    """One poke-engine game with per-side information state. See module docstring.

    Drive it: `pending_sides()` -> for each side build `view(side)`, pick an action index, convert
    with `engine_action(side, idx)`, then `step({side: string, other: "none" when absent})`.
    `finished`/`winner` report the outcome; `observations` accumulate per-turn synthesized events.
    """

    MAX_TURNS = 300

    def __init__(self, paste_one: str, paste_two: str, *, seed: int = 0, tables=None,
                 battle_tag: str = "engine-battle", leads: tuple[int, int] = (0, 0)):
        _require_engine()
        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        self.tables = tables
        self.battle_tag = battle_tag
        self.rng = random.Random(seed)
        self._pastes = {"p1": paste_one, "p2": paste_two}
        self._mons = {"p1": parse_paste(paste_one), "p2": parse_paste(paste_two)}
        self.paste_species = {
            r: [to_id(m.species or m.nickname) for m in self._mons[r]] for r in ("p1", "p2")
        }

        self.turn = 1
        self.observations: dict[int, Any] = {}
        self.finished = False
        self.winner: str | None = None
        self.tera_used = {"p1": False, "p2": False}
        # per-side reveal census keyed "p1: speciesid" (drives view censoring; the RevealLedger
        # mirrors the same facts as table indices for the leakage path)
        self.revealed: dict[str, dict[str, dict]] = {"p1": {}, "p2": {}}
        self.field_start: dict[str, int] = {}
        self.unknown_instructions: dict[str, int] = {}
        self._pending_events: list[list[str]] = []
        self._fs_end: list[tuple[str, int]] = []   # (role it lands on, turn it resolves)

        from fog.encoding.reveals import RevealLedger
        from fog.encoding.tracker import BattleTracker

        self.trackers = {"p1": BattleTracker(my_role="p1"), "p2": BattleTracker(my_role="p2")}
        self.ledgers = {"p1": RevealLedger(), "p2": RevealLedger()}

        self._build_state(leads)

    # ------------------------------------------------------------------------------ construction
    def set_leads(self, lead_one: int, lead_two: int) -> None:
        """Reorder both teams lead-first (team preview). Only before the first step."""
        if self.turn != 1 or self._pending_events or self.observations:
            raise RuntimeError("set_leads is only valid before the battle starts")
        self._build_state((lead_one, lead_two))

    def _build_state(self, leads: tuple[int, int]) -> None:
        team_mons: dict[str, list] = {}
        for role, lead in zip(("p1", "p2"), leads):
            mons = self._mons[role]
            if not 0 <= lead < len(mons):
                lead = 0
            team_mons[role] = [mons[lead]] + [m for i, m in enumerate(mons) if i != lead]
        engine_mons = {
            role: [build_engine_pokemon(m, self.tables) for m in team_mons[role]]
            for role in ("p1", "p2")
        }

        # lead switch-in abilities (engine fires them only on Switch instructions): weather/terrain
        # in speed order — slower activates last and WINS; Intimidate hits the enemy side's boost.
        state_kw: dict[str, Any] = {}
        boosts = {"p1": 0, "p2": 0}
        order = sorted(("p1", "p2"), key=lambda r: engine_mons[r][0].speed, reverse=True)
        for role in order:
            lead = engine_mons[role][0]
            ability = to_id(lead.ability)
            other = "p2" if role == "p1" else "p1"
            if ability in _LEAD_WEATHER:
                state_kw["weather"] = _LEAD_WEATHER[ability]
                state_kw["weather_turns_remaining"] = -1
                self._record_reveal(role, lead.id, ability=str(lead.ability))
            elif ability in _LEAD_TERRAIN:
                state_kw["terrain"] = _LEAD_TERRAIN[ability]
                state_kw["terrain_turns_remaining"] = 5
                self._record_reveal(role, lead.id, ability=str(lead.ability))
            elif ability == "intimidate":
                boosts[other] -= 1
                self._record_reveal(role, lead.id, ability=str(lead.ability))
        if "weather" in state_kw:
            self.field_start["weather"] = 0
        if "terrain" in state_kw:
            self.field_start["terrain"] = 0

        self.state = pe.State(
            side_one=pe.Side(pokemon=engine_mons["p1"], attack_boost=boosts["p1"]),
            side_two=pe.Side(pokemon=engine_mons["p2"], attack_boost=boosts["p2"]),
            **state_kw,
        )

    # ------------------------------------------------------------------------------ reveal census
    def _record_item_change(self, role: str, mon_id: str, old: str, new: str) -> None:
        """Reveal census for a ChangeItem instruction. The census tracks SHEET knowledge (the
        belief is over ORIGINAL sets), so what an item change exposes is the DEPARTING item —
        on the first change that IS the sheet item (Trick/Knock Off announce it). Recording the
        swapped-IN item instead contradicts every pool paste and eliminated the TRUE team from
        exact own-state enumeration (crashed a live posterior_log run, 2026-07-11). Consumption
        keeps the "" (<NO_ITEM>) convention; an existing sheet reveal is never overwritten."""
        rev = self.revealed[role].get(f"{role}: {to_id(mon_id)}", {})
        if rev.get("item") not in (None, ""):
            return  # sheet item already known — later changes carry no sheet information
        if new in ("none", ""):
            self._record_reveal(role, mon_id, item="")  # consumed/knocked: <NO_ITEM> convention
        else:
            self._record_reveal(role, mon_id, item=(old if old not in ("none", "") else ""))

    def _record_reveal(self, role: str, mon_id: str, *, move: str | None = None,
                       item: str | None = None, ability: str | None = None) -> None:
        mon_id = to_id(mon_id)
        if item is not None and item != "":
            item = to_id(item)
        if ability is not None:
            ability = to_id(ability)
        rev = self.revealed[role].setdefault(f"{role}: {mon_id}", {"moves": []})
        if move is not None and move not in rev["moves"] and len(rev["moves"]) < 4:
            rev["moves"].append(move)
        if item is not None:
            rev["item"] = item          # "" == revealed-gone (<NO_ITEM>)
        if ability is not None:
            rev["ability"] = ability
        # ledger mirror (table indices) — feed_ledger_from_events handles the event-driven path;
        # this direct call only covers construction-time (lead ability) reveals.
        if ability is not None and not self.observations and not self._pending_events:
            idx = self.tables.ability_index(to_id(ability), default=None)
            if idx is not None:
                self.ledgers[role].record_ability(f"{role}: {mon_id}", idx)

    # ----------------------------------------------------------------------------------- queries
    def side(self, role: str):
        return self.state.side_one if role == "p1" else self.state.side_two

    def active(self, role: str):
        s = self.side(role)
        return s.pokemon[int(s.active_index)]

    def needs_replacement(self, role: str) -> bool:
        s = self.side(role)
        return self.active(role).hp <= 0 and any(p.hp > 0 for p in s.pokemon)

    def pending_sides(self) -> list[str]:
        """Sides that must submit an action for the next step ([] once finished)."""
        if self.finished:
            return []
        forced = [
            r for r in ("p1", "p2")
            if self.needs_replacement(r) or bool(self.side(r).force_switch)
            or bool(self.side(r).baton_passing) or bool(self.side(r).shed_tailing)
        ]
        return forced if forced else ["p1", "p2"]

    def view(self, role: str) -> EngineView:
        return EngineView(self, role)

    def action_mask(self, role: str) -> np.ndarray:
        """14-way legality mask — literally env.actions.action_legality_mask over the view, so the
        mask and the observation can never disagree."""
        from fog.env.actions import action_legality_mask

        return np.asarray(action_legality_mask(self.view(role)), dtype=bool)

    def engine_action(self, role: str, action_idx: int) -> str:
        """Project action index (0-13) -> engine move string, mirroring env/actions semantics."""
        from fog.constants.schema import ACTIONS, ActionType

        s = self.side(role)
        active = s.pokemon[int(s.active_index)]
        legal_moves = [m for m in active.moves if not m.disabled and int(m.pp) > 0]
        bench = [p for i, p in enumerate(s.pokemon)
                 if i != int(s.active_index) and to_id(p.id) not in ("", "none")]
        alive_bench = [p for p in bench if p.hp > 0]

        kind, sub = ACTIONS[action_idx]
        if kind is ActionType.FORCED:
            if self.needs_replacement(role) or s.force_switch:
                return str(alive_bench[0].id) if alive_bench else str(active.moves[0].id)
            return str(legal_moves[0].id) if legal_moves else (
                str(alive_bench[0].id) if alive_bench else str(active.moves[0].id)
            )
        if kind in (ActionType.MOVE, ActionType.MOVE_TERA):
            moves = list(active.moves)
            move = moves[sub] if sub < len(moves) else moves[0]
            return f"{move.id}-tera" if kind is ActionType.MOVE_TERA else str(move.id)
        target = bench[sub] if sub < len(bench) else (alive_bench[0] if alive_bench else bench[0])
        return str(target.id)

    def truth_tokens(self, role: str) -> np.ndarray:
        """[6,9] full-truth tokens for `role`'s team (paste order = view slot order)."""
        from fog.env.team_truth import truth_tokens_from_paste

        return truth_tokens_from_paste(self._pastes[role], self.tables)

    # ------------------------------------------------------------------------- search-world fork
    def fork_for_search(self, role: str, world_tokens: np.ndarray, *, seed: int = 0) -> "EngineBattle":
        """A disposable copy of this battle with the opponent's HIDDEN attributes replaced by one
        sampled belief world (search §8). `role` is the searcher; `world_tokens` is one [6,9] row-
        per-slot sample from BeliefModel.sample_team (paste/preview slot order).

        The sampled world supplies exactly what `role` cannot see (item, ability, moves, spread,
        tera); everything public stays real: HP fraction, status, boosts, volatiles, hazards,
        weather/terrain, active mons, and `role`'s own side (deep-cloned engine truth). Trackers
        and the reveal census are deep-copied so leaf observations keep real history and honest
        censoring — the searcher's leaf view NEVER sees engine truth, only root reveals plus
        whatever the simulated ply itself reveals.

        Never mutates `self`: the searcher's side object is REUSED (safe — the engine never
        mutates states, `apply_instructions` returns successors; pinned by test_engine_contract),
        the opponent side is rebuilt, and the fork gets its own rng (seeded by `seed`) so search
        never advances the real battle's stochastic stream.

        Known fork losses (acceptable for 1-ply leaf evaluation, documented for depth>1 use):
        `_pastes[opp]` still holds the TRUE paste — never call truth_tokens(opp) on a fork. Slot
        matching is by species (standard preview formats); hidden-species worlds fall back to
        token order. (Move `disabled` flags ARE preserved — see `_world_engine_mon`.)
        """
        opp_role = "p2" if role == "p1" else "p1"
        return self._fork_worlds({opp_role: world_tokens}, seed=seed)

    def fork_for_own_world(self, role: str, world_tokens: np.ndarray, *, seed: int = 0,
                           opp_world_tokens: np.ndarray | None = None) -> "EngineBattle":
        """The mirror of `fork_for_search` for joint search (SEARCH_PLAN §4.1): a disposable copy
        with `role`'s OWN hidden attributes replaced by one own-world sample — a team the OPPONENT
        considers possible for `role`. Public board state (HP fraction, status, boosts, tera state,
        disabled moves) stays real, exactly as in `fork_for_search`; only the sheet-hidden
        attributes come from the world. `opp_world_tokens`, when given, additionally swaps the
        opponent's hidden attributes (the searcher doesn't know THEIR team either — two-sided
        worlds, one fork).

        Own-world fidelity notes (uniform across worlds, so cross-world value comparisons share
        the bias): the world's spread cluster replaces `role`'s exact EVs even for the true-state
        world (nearest-cluster stats), and a consumed item is restored to the world's sheet item
        (same approximation `fork_for_search` makes for the opponent). `truth_tokens(role)` on the
        fork still returns the REAL paste — never call it on a fork.
        """
        replace = {role: world_tokens}
        if opp_world_tokens is not None:
            replace["p2" if role == "p1" else "p1"] = opp_world_tokens
        return self._fork_worlds(replace, seed=seed)

    def _fork_worlds(self, replace: dict[str, np.ndarray], *, seed: int) -> "EngineBattle":
        """Shared fork body: rebuild the sides named in `replace` from [6,9] world tokens, keep
        every other side's engine truth (safe to reuse — the engine never mutates states)."""
        _require_engine()
        if self.finished:
            raise RuntimeError("cannot fork a finished battle")
        worlds = {r: np.asarray(w, dtype=np.int64) for r, w in replace.items()}
        for r, w in worlds.items():
            if w.shape != (N_SLOTS, 9):
                raise ValueError(f"world_tokens for {r} must be [6,9], got {w.shape}")

        # NOT a to_string/from_string round-trip: serialization uppercases every id, which would
        # desync the fork from to_id()-lowercased tables/views. Attribute reads keep ids intact.
        sides = {"p1": self.state.side_one, "p2": self.state.side_two}
        for r, w in worlds.items():
            sides[r] = _rebuild_side(sides[r], self._world_side_mons(sides[r], w))
        state = self._rebuild_state(sides["p1"], sides["p2"])
        fork = self._clone_shell(state, seed=seed)
        for r, w in worlds.items():
            fork.paste_species[r] = self._world_species_list(w)
        return fork

    def _world_side_mons(self, side: Any, world_tokens: np.ndarray) -> list:
        """Match engine side positions -> world token rows by species (lead reordering vs paste
        order) and rebuild each mon from its row, overlaying the source mon's public state."""
        rows = list(range(N_SLOTS))
        used = [False] * N_SLOTS
        new_mons = []
        for p in side.pokemon:
            if to_id(p.id) in ("", "none"):
                new_mons.append(p)  # placeholder padding mon: keep as-is
                continue
            sp_idx = self.tables.species_index(to_id(p.id))
            row_j = next(
                (j for j in rows if not used[j] and int(world_tokens[j, 0]) == sp_idx),
                next((j for j in rows if not used[j]), None),
            )
            if row_j is None:
                new_mons.append(p)
                continue
            used[row_j] = True
            new_mons.append(_world_engine_mon(world_tokens[row_j], p, self.tables))
        return new_mons

    def _world_species_list(self, world_tokens: np.ndarray) -> list[str]:
        """Preview species list for a world ([6,9] tokens). Empty/sentinel slots (short teams,
        <RARE_SPECIES>) contribute no preview name."""
        species_ids = self.tables.species["ids"]
        return [
            species_ids[int(world_tokens[j, 0])]
            for j in range(N_SLOTS)
            if 0 <= int(world_tokens[j, 0]) < len(species_ids)
            and not species_ids[int(world_tokens[j, 0])].startswith("<")
        ]

    @classmethod
    def _from_reconstruction(
        cls, state, *, role: str, tables, tracker, revealed: dict, paste_species: dict,
        turn: int, tera_used: dict, field_start: dict, seed: int = 0,
    ) -> "EngineBattle":
        """Wrap an externally-reconstructed engine `state` (state_recon: a Showdown position + one
        belief world) as a steppable EngineBattle from `role`'s perspective. NO pastes are attached
        — `truth_tokens` must never be called (the opponent side is a sample, not truth). `tracker`
        supplies the searcher's real history for leaf encoding; `revealed` is the searcher's current
        opponent reveal census (drives leaf censoring)."""
        from fog.encoding.reveals import RevealLedger
        from fog.encoding.tracker import BattleTracker

        opp_role = "p2" if role == "p1" else "p1"
        self = object.__new__(cls)
        self.tables = tables
        self.battle_tag = "recon-battle"
        self.rng = random.Random(seed)
        self._pastes = {}
        self._mons = {}
        self.paste_species = paste_species
        self.turn = int(turn)
        self.observations = {}
        self.finished = False
        self.winner = None
        self.tera_used = dict(tera_used)
        self.revealed = revealed
        self.field_start = dict(field_start)
        self.unknown_instructions = {}
        self._pending_events = []
        self._fs_end = []
        self.trackers = {role: tracker, opp_role: BattleTracker(my_role=opp_role)}
        self.ledgers = {role: RevealLedger(), opp_role: RevealLedger()}
        self.state = state
        return self

    def clone_for_step(self, *, seed: int = 0) -> "EngineBattle":
        """A disposable copy that can be stepped one ply without touching `self` — same as
        `fork_for_search` but with the opponent side UNCHANGED (used when the world is already
        baked into `self`, e.g. a Showdown-reconstructed battle). See `_clone_shell`."""
        _require_engine()
        if self.finished:
            raise RuntimeError("cannot clone a finished battle")
        return self._clone_shell(
            self._rebuild_state(self.state.side_one, self.state.side_two), seed=seed
        )

    def _rebuild_state(self, side_one, side_two) -> "pe.State":
        return pe.State(
            side_one=side_one, side_two=side_two,
            weather=str(self.state.weather),
            weather_turns_remaining=int(self.state.weather_turns_remaining),
            terrain=str(self.state.terrain),
            terrain_turns_remaining=int(self.state.terrain_turns_remaining),
            trick_room=bool(self.state.trick_room),
            trick_room_turns_remaining=int(self.state.trick_room_turns_remaining),
        )

    def _clone_shell(self, state, *, seed: int) -> "EngineBattle":
        """Shared clone body for _fork_worlds / clone_for_step. Trackers use the cheap
        `BattleTracker.clone` (shares read-only history); ledgers are fresh-empty — a leaf
        observation reads censoring from `self.revealed`, never the ledger, so copying the ledger
        is wasted work. `self` is never mutated."""
        from fog.encoding.reveals import RevealLedger

        fork = object.__new__(EngineBattle)
        fork.tables = self.tables
        fork.battle_tag = f"{self.battle_tag}-world"
        fork.rng = random.Random(seed)
        fork._pastes = dict(self._pastes)
        fork._mons = self._mons  # only read by set_leads/_build_state, which forks never call
        fork.paste_species = {r: list(v) for r, v in self.paste_species.items()}
        fork.turn = self.turn
        fork.observations = dict(self.observations)
        fork.finished = False
        fork.winner = None
        fork.tera_used = dict(self.tera_used)
        fork.revealed = {r: {k: {**v, "moves": list(v.get("moves", []))}
                             for k, v in mons.items()}
                         for r, mons in self.revealed.items()}
        fork.field_start = dict(self.field_start)
        fork.unknown_instructions = {}
        fork._pending_events = [list(e) for e in self._pending_events]
        fork._fs_end = list(self._fs_end)
        fork.trackers = {r: t.clone() for r, t in self.trackers.items()}
        fork.ledgers = {r: RevealLedger() for r in self.ledgers}
        fork.state = state
        return fork

    # -------------------------------------------------------------------------------------- step
    def _resolve_moves(self, moves: dict[str, str]) -> tuple[str, str, set]:
        """Shared step header: validate and fill the per-side engine strings."""
        if self.finished:
            raise RuntimeError("battle already finished")
        pending = set(self.pending_sides())
        m1 = moves.get("p1", "none" if "p1" not in pending else None)
        m2 = moves.get("p2", "none" if "p2" not in pending else None)
        if m1 is None or m2 is None:
            raise ValueError(f"missing action for a pending side (pending={sorted(pending)})")
        return m1, m2, pending

    def _apply_chosen(self, pre, chosen, m1: str, m2: str, pending: set) -> None:
        """Advance THIS battle under one already-selected chance outcome (`chosen` is one element
        of the pe.generate_instructions list). Shared tail of step() / step_branches()."""
        self.state = pre.apply_instructions(chosen)

        self._pending_events.extend(
            self._synthesize(pre, chosen, {"p1": m1, "p2": m2}, pending)
        )
        for role in ("p1", "p2"):
            self.tera_used[role] = self.tera_used[role] or any(
                bool(p.terastallized) for p in self.side(role).pokemon
            )

        alive = {r: any(p.hp > 0 for p in self.side(r).pokemon) for r in ("p1", "p2")}
        if not (alive["p1"] and alive["p2"]) or self.turn > self.MAX_TURNS:
            self.finished = True
            self.winner = (
                "p1" if alive["p1"] and not alive["p2"]
                else "p2" if alive["p2"] and not alive["p1"]
                else None
            )
            self._finalize_turn()
            return
        if self.pending_sides() == ["p1", "p2"]:
            self._finalize_turn()
            self.turn += 1

    def step(self, moves: dict[str, str]) -> None:
        """Advance one engine decision round. `moves` maps role -> engine string; a side that is
        not pending passes "none" (filled in automatically when absent). Chance resolves by
        SAMPLING one outcome from the engine's enumerated branch list — the volume-play path.
        Search must use step_branches instead (single-sample cells are interim only,
        PROJECT_DESIGN_PROPOSAL.md §10.4 / §18)."""
        m1, m2, pending = self._resolve_moves(moves)
        pre = self.state
        sets = pe.generate_instructions(pre, m1, m2)
        if not sets:
            raise RuntimeError(f"engine produced no instructions for ({m1!r}, {m2!r})")
        r = self.rng.random() * 100.0
        acc, chosen = 0.0, sets[-1]
        for cand in sets:
            acc += float(cand.percentage)
            if r <= acc:
                chosen = cand
                break
        self._apply_chosen(pre, chosen, m1, m2, pending)

    def step_branches(
        self, moves: dict[str, str], *, min_prob: float = 0.0, max_branches: int | None = None,
    ) -> tuple[list[tuple[float, "EngineBattle"]], float]:
        """Every chance outcome of one decision round, each as its own advanced fork
        (PROJECT_DESIGN_PROPOSAL.md §10.4: enumerate and branch, never one frozen sample).

        Returns `(branches, pruned_mass)`. `branches` is a list of `(probability, fork)` pairs:
        the engine's enumerated outcomes of committing `moves` (hit/miss, critical or not,
        secondary effect or not, and their combinations — the engine's granularity; damage rolls
        are collapsed to the average roll, see the §10.4 backend note). Probabilities are
        renormalized over the kept set so they always sum to one; `pruned_mass` is the total
        probability dropped by `min_prob` / `max_branches` pruning (log it — §19.16). The most
        likely branch is always kept. Branch order follows the engine's enumeration order.

        `self` is NOT advanced (unlike step); it is mutated only through its random stream, which
        seeds the forks. Every fork draws a fresh, independent random future — never a shared one
        (settled decision 21.17) — so the whole call is deterministic given this battle's rng
        state, while sibling forks stay uncorrelated if they are stepped further.
        """
        m1, m2, pending = self._resolve_moves(moves)
        pre = self.state
        sets = pe.generate_instructions(pre, m1, m2)
        if not sets:
            raise RuntimeError(f"engine produced no instructions for ({m1!r}, {m2!r})")
        probs = [float(c.percentage) / 100.0 for c in sets]
        top = max(range(len(sets)), key=lambda i: probs[i])
        keep = [i for i in range(len(sets)) if probs[i] >= min_prob or i == top]
        if max_branches is not None and len(keep) > max_branches:
            keep = sorted(sorted(keep, key=lambda i: -probs[i])[:max_branches])
        kept_mass = sum(probs[i] for i in keep)
        pruned_mass = max(0.0, 1.0 - kept_mass)
        branches: list[tuple[float, EngineBattle]] = []
        for i in keep:
            fork = self.clone_for_step(seed=self.rng.randrange(2**31))
            fork._apply_chosen(pre, sets[i], m1, m2, pending)
            branches.append((probs[i] / kept_mass, fork))
        return branches, pruned_mass

    # ---------------------------------------------------------------------------- event synthesis
    def _synthesize(self, pre, chosen, moves: dict[str, str], pending: set[str]) -> list[list[str]]:
        """Applied instruction reprs + chosen actions -> ordered pseudo-protocol events."""
        sides = {"p1": pre.side_one, "p2": pre.side_two}
        active = {r: int(sides[r].active_index) for r in ("p1", "p2")}
        mons = {r: list(sides[r].pokemon) for r in ("p1", "p2")}
        hp = {(r, i): int(p.hp) for r in ("p1", "p2") for i, p in enumerate(mons[r])}
        maxhp = {(r, i): max(int(p.maxhp), 1) for r in ("p1", "p2") for i, p in enumerate(mons[r])}

        def name(r):  # current active's protocol token
            return f"{r}a: {to_id(mons[r][active[r]].id)}"

        def cond(r):
            i = active[r]
            return f"{max(hp[(r, i)], 0)}/{maxhp[(r, i)]}"

        # classify chosen actions
        acts: dict[str, tuple] = {}
        for r in ("p1", "p2"):
            m = moves[r]
            if r not in pending or m == "none":
                acts[r] = ("none",)
            elif m.endswith("-tera"):
                acts[r] = ("move", m[: -len("-tera")], True)
            elif any(to_id(mv.id) == to_id(m) for mv in mons[r][active[r]].moves):
                acts[r] = ("move", m, False)
            else:
                acts[r] = ("switch", m)

        reprs = [repr(i) for i in chosen.instruction_list]

        # ---- move events, in inferred order (prepended; position vs other events is irrelevant
        # to the parsers, only the RELATIVE order of the two move events matters) ------------------
        movers = [r for r in ("p1", "p2") if acts[r][0] == "move"]
        if len(movers) == 2:
            movers = self._order_movers(pre, acts, reprs)
        events: list[list[str]] = []
        for r in movers:
            move_id = to_id(acts[r][1])
            events.append(["", "move", name(r), move_id])
            if move_id in ("futuresight", "doomdesire"):
                events.append(["", "-start", name(r), "move: futuresight"])
                self._fs_end.append(("p2" if r == "p1" else "p1", self.turn + 2))
            if acts[r][2]:  # tera chosen with the move
                events.append(["", "-terastallize", name(r), str(mons[r][active[r]].tera_type)])

        # ---- instruction walk -----------------------------------------------------------------
        last_switch_in: dict[str, int] = {}  # role -> event index of its most recent switch-in
        for rep in reprs:
            if m := _INSTR["damage"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                i = active[r]
                was_alive = hp[(r, i)] > 0
                hp[(r, i)] -= int(m.group(2))
                events.append(["", "-damage", name(r), cond(r)])
                if was_alive and hp[(r, i)] <= 0:
                    events.append(["", "faint", name(r)])
            elif m := _INSTR["heal"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                hp[(r, active[r])] += int(m.group(2))
                events.append(["", "-heal", name(r), cond(r)])
            elif m := _INSTR["switch"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                active[r] = int(m.group(3))
                events.append(["", "switch", name(r), to_id(mons[r][active[r]].id), cond(r)])
                last_switch_in[r] = len(events) - 1
            elif m := _INSTR["item"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                old, new = to_id(m.group(2)), to_id(m.group(3))
                if new in ("none", ""):
                    events.append(["", "-enditem", name(r), old])
                else:
                    events.append(["", "-item", name(r), new])
                self._record_item_change(r, str(mons[r][active[r]].id), old, new)
            elif m := _INSTR["volatile+"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                events.append(["", "-start", name(r), to_id(m.group(2))])
            elif m := _INSTR["volatile-"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                events.append(["", "-end", name(r), to_id(m.group(2))])
            elif m := _INSTR["tera"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                if not any(e[1] == "-terastallize" and e[2] == name(r) for e in events):
                    events.append(
                        ["", "-terastallize", name(r), str(mons[r][active[r]].tera_type)]
                    )
            elif m := _INSTR["weather"].match(rep):
                self.field_start["weather"] = self.turn
                self._ability_reveal_from_field(events, last_switch_in, mons, active,
                                                _LEAD_WEATHER, to_id(m.group(1)))
            elif m := _INSTR["terrain"].match(rep):
                self.field_start["terrain"] = self.turn
                self._ability_reveal_from_field(events, last_switch_in, mons, active,
                                                _LEAD_TERRAIN, to_id(m.group(1)))
            elif m := _INSTR["boost"].match(rep):
                r = "p1" if m.group(1) == "One" else "p2"
                if m.group(2) == "Attack" and int(m.group(3)) < 0:
                    other = "p2" if r == "p1" else "p1"
                    inti = mons[other][active[other]]
                    if to_id(inti.ability) == "intimidate" and other in last_switch_in:
                        events.append(
                            ["", "-ability", f"{other}a: {to_id(inti.id)}", "intimidate"]
                        )
                        self._record_reveal(other, inti.id, ability="intimidate")
            elif _INSTR["forceswitch"].match(rep) or _IGNORED.match(rep):
                pass
            else:
                key = rep.split(":")[0].split(" Side")[0]
                self.unknown_instructions[key] = self.unknown_instructions.get(key, 0) + 1

        # ---- reveal census from move events (mirrors feed_ledger rules, string domain) ----------
        for ev in events:
            if ev[1] == "move":
                r = ev[2][:2]
                self._record_reveal(r, ev[2].split(": ", 1)[1], move=to_id(ev[3]))
        return events

    def _ability_reveal_from_field(self, events, last_switch_in, mons, active,
                                   ability_map: dict, new_value: str) -> None:
        """Weather/terrain changed: if a mon that just switched in (this round) owns the matching
        setter ability, that ability is now public."""
        for r, ev_idx in last_switch_in.items():
            mon = mons[r][active[r]]
            ability = to_id(mon.ability)
            if ability_map.get(ability) and to_id(ability_map[ability]) == new_value:
                events.append(["", "-ability", f"{r}a: {to_id(mon.id)}", ability])
                self._record_reveal(r, mon.id, ability=ability)

    def _order_movers(self, pre, acts: dict, reprs: list[str]) -> list[str]:
        """Who moved first when both sides used a move: damage evidence, else priority + speed."""
        # evidence: the first Damage instruction hits the side that did NOT act at that point —
        # only trustworthy when both moves deal damage (status moves emit no Damage)
        feats = self.tables.moves["features"]
        power = {
            r: (feats.get(to_id(acts[r][1]), {}) or {}).get("base_power", 0) for r in ("p1", "p2")
        }
        if power["p1"] and power["p2"]:
            for rep in reprs:
                if m := _INSTR["damage"].match(rep):
                    target = "p1" if m.group(1) == "One" else "p2"
                    return ["p2", "p1"] if target == "p1" else ["p1", "p2"]
        prio = {
            r: (feats.get(to_id(acts[r][1]), {}) or {}).get("priority", 0) for r in ("p1", "p2")
        }
        if prio["p1"] != prio["p2"]:
            return ["p1", "p2"] if prio["p1"] > prio["p2"] else ["p2", "p1"]

        def eff_speed(role: str) -> float:
            side = pre.side_one if role == "p1" else pre.side_two
            mon = side.pokemon[int(side.active_index)]
            boost = int(side.speed_boost)
            mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
            spe = float(mon.speed) * mult
            if to_id(getattr(mon, "status", "none") or "none") in ("par", "paralyze", "paralyzed"):
                spe *= 0.5
            if to_id(mon.item) == "choicescarf":
                spe *= 1.5
            if int(getattr(side.side_conditions, "tailwind", 0) or 0):
                spe *= 2.0
            return spe

        s1, s2 = eff_speed("p1"), eff_speed("p2")
        first = ("p1" if s1 >= s2 else "p2") if not bool(pre.trick_room) else ("p1" if s1 <= s2 else "p2")
        return [first, "p2" if first == "p1" else "p1"]

    # ------------------------------------------------------------------------------ turn finalize
    def _finalize_turn(self) -> None:
        events = self._pending_events
        self._pending_events = []
        # scheduled Future Sight resolutions (tracker expects an -end on the side it lands on)
        for role, t in list(self._fs_end):
            if self.turn >= t:
                s = self.side(role)
                events.append(["", "-end", f"{role}a: {s.pokemon[int(s.active_index)].id}",
                               "move: futuresight"])
                self._fs_end.remove((role, t))
        self.observations[self.turn] = SimpleNamespace(events=events)

        from fog.encoding.events import feed_ledger_from_events

        for role in ("p1", "p2"):
            self.trackers[role].feed_turn(self.turn, events)
            feed_ledger_from_events(events, role, self.ledgers[role], self.tables)
