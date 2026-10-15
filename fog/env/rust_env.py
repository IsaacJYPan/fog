"""poke-engine (Rust) adapter — groundwork for the M3/M4 volume track (plan §7).

STATUS (honest, post-audit 2026-07-05; updated 2026-07-08): this module is an ADAPTER AND SMOKE
TEST. The partial-observability engine TRAINING path it flagged as unbuilt now lives in
`env/engine_battle.py` (`EngineBattle` + reveal census + event synthesis) and `env/engine_runner.py`
(batched self-play into RolloutBuffer) — use those, not this proxy, for volume self-play.

A SECOND consumer now exists: `env/state_recon.py` reconstructs a full engine `State` from a live
Showdown position + one belief-sampled world, powering determinized search on the Showdown backend
(`search/engine_step.ShowdownWorldStepper`, spec §8). That reconstruction is validated by
`scripts/fuzz_recon.py` (round-trips random positions; ~0 divergence on the core fields).

This proxy itself remains a smoke test / shape check. What it still does NOT do (use EngineBattle):
  * partial-observability reveal tracking (RustBattleProxy exposes HIDDEN-only or, with
    `reveal_opponent=True`, perfect information — NEVER train on that);
  * legality nuance (choice lock, trapping, disable, forced switches) beyond the basic mask.

Verified 2026-07: poke-engine 0.0.48 must be built with its `gen9` Cargo feature and older
generation defaults disabled. The ordinary prebuilt wheel does not establish that requirement;
the locked uv configuration compiles it from source. The Gen 9 build exposes the
terastallization feature (`Pokemon.tera_type` / `.terastallized` exist) — the load-bearing coverage
question. The proxy feeds the SAME encoders as the Showdown path, but note "same encoder" does not
mean "same inputs": every proxy field below that approximates (see _ProxyMon) is a drift source the
cross-sim fuzz must catch.

Division of labor (see ARCHITECTURE.md simulator authority):
  * poke-engine  -> self-play rollout volume (synchronous, no websocket, orders of magnitude faster)
  * Showdown     -> ground truth: BC replays, gauntlet/Elo eval, fuzzing, ladder, final fine-tune

Divergence control (the real risk — PPO will farm sim bugs):
  * `fog/scripts/fuzz_sim.py` replays seeded action scripts through BOTH sims and diffs per-turn
    hp/status/boost trajectories; run >=1k games before trusting the engine for training.
  * cross-sim gauntlet every N checkpoints; HALT if the Rust-vs-Showdown win-rate gap exceeds 3%.
  * final fine-tune (last ~200k games before any headline number) always on Showdown.

Move strings (verified against the Gen 9 build of poke-engine 0.0.48): a move id ("earthquake"),
a bare pokemon id for a switch ("ironvaliant" — no "switch " prefix), and "<moveid>-tera" to
terastallize with the move.
`generate_instructions(state, s1_move, s2_move)` returns probability-weighted StateInstructions;
`state.apply_instructions(instructions[i])` advances (sample by `.percentage` for stochastic play).
Re-run `smoke_roundtrip()` after any poke-engine upgrade — the string formats have drifted between
releases before.
"""
from __future__ import annotations

try:
    import poke_engine
    HAVE_ENGINE = True
except ImportError:  # pragma: no cover
    HAVE_ENGINE = False


def _require_engine() -> None:
    if not HAVE_ENGINE:
        raise RuntimeError("poke-engine is not installed: pip install poke-engine")


# ---------------------------------------------------------------------------- paste -> engine side
def paste_to_engine_pokemon(mon) -> poke_engine.Pokemon:
    """poke-env TeambuilderPokemon (parsed paste mon) -> poke_engine.Pokemon.

    With built tables this computes REAL stats/types/weight (pe.Pokemon derives nothing from
    id/evs/nature — verified 0.0.48: omitting them plays every mon at 100s and NORMAL/TYPELESS,
    which silently voids all damage calc). The table-less fallback keeps the old approximate
    construction for shape tests only — never fuzz or train on it.
    """
    _require_engine()
    import re

    tables = _try_tables()
    if tables is not None:
        from fog.env.engine_battle import build_engine_pokemon

        return build_engine_pokemon(mon, tables)

    def _id(x: object) -> str:
        return re.sub(r"[^a-z0-9]", "", str(x).lower())

    moves = [poke_engine.Move(id=_id(m)) for m in (mon.moves or []) if m]
    return poke_engine.Pokemon(
        id=_id(mon.species or mon.nickname),
        level=int(mon.level or 100),
        evs=tuple(int(v or 0) for v in (mon.evs or [0] * 6)),
        nature=_id(mon.nature or "serious"),
        ability=_id(mon.ability or ""),
        item=_id(mon.item or "none"),
        moves=moves,
        tera_type=_id(getattr(mon, "tera_type", None) or "normal"),
    )


def paste_to_engine_side(paste: str) -> poke_engine.Side:
    """Showdown paste -> poke_engine.Side (6 mons, first active)."""
    _require_engine()
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self):  # pragma: no cover
            return ""

    from fog.env.teams import normalize_paste_newlines

    mons = _T().parse_showdown_team(normalize_paste_newlines(paste))[:6]
    return poke_engine.Side(pokemon=[paste_to_engine_pokemon(m) for m in mons])


# --------------------------------------------------------------- engine state -> encoder battle proxy
def _try_tables():
    """Generated tables when built (base stats / weight source); None keeps the fallback path."""
    from fog.constants.tables import TABLES, TablesNotBuilt

    try:
        _ = TABLES.species  # force lazy load; raises when tables aren't generated
        return TABLES
    except TablesNotBuilt:
        return None


class _ProxyMove:
    def __init__(self, move_id: str, pp: int = 16, max_pp: int = 16):
        self.id = move_id
        self.current_pp = pp
        self.max_pp = max_pp


class _ProxyMon:
    """Duck-typed mon exposing the attributes fog.encoding.* reads (see tests/golden_fixture.py).

    Fidelity notes (drift sources vs the Showdown path — the fuzz harness must cover these):
      * base_stats come from the generated species table (the Showdown-path source). The engine only
        exposes COMPUTED stats (HP 404 etc.), which are NOT base stats — encoding those under a /255
        scale put every entity off-distribution; the computed values remain only as a last-resort
        fallback for species missing from the table.
      * weight likewise from the table (engine builds may omit weight_kg -> Grass Knot bucket wrong).
      * status_counter covers sleep turns only; boosts/effects are filled by the proxy caller.
    """

    def __init__(self, ep, revealed: bool = True, tables=None):
        self.species = str(ep.id)
        self.item = str(ep.item) if revealed and str(ep.item) not in ("", "none") else None
        self.ability = str(ep.ability) if revealed else None
        self.moves = {str(m.id): _ProxyMove(str(m.id), getattr(m, "pp", 16))
                      for m in (ep.moves or []) if revealed}
        self.tera_type = str(ep.tera_type) if getattr(ep, "terastallized", False) or revealed else None
        self.terastallized = bool(getattr(ep, "terastallized", False))

        feats = None
        if tables is not None:
            try:
                feats = tables.species["features"].get(self.species)
            except (AttributeError, KeyError, TypeError):
                feats = None
        if feats is not None:
            self.base_stats = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"),
                                       feats["base_stats"]))
            self.weight = float(feats.get("weightkg", 50.0) or 50.0)
        else:  # last-resort fallback: computed stats, WRONG scale — logged as a drift source above
            stats = (ep.maxhp, ep.attack, ep.defense, ep.special_attack, ep.special_defense, ep.speed)
            self.base_stats = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), stats))
            self.weight = float(getattr(ep, "weight_kg", 50.0) or 50.0)

        self.types = [str(t) for t in (getattr(ep, "types", None) or ()) if str(t) != "typeless"]
        self.current_hp_fraction = (ep.hp / ep.maxhp) if ep.maxhp else 0.0
        self.fainted = ep.hp <= 0
        status = str(getattr(ep, "status", "none") or "none")
        self.status = None if status in ("none", "None") else status
        self.status_counter = int(getattr(ep, "sleep_turns", 0) or 0)
        self.boosts = {}
        self.effects = {}


class RustBattleProxy:
    """A poke-env-battle-shaped view of a poke_engine.State, from side_one's perspective.

    Feed it to `Observation.from_battle` — the same encoder the Showdown path uses (the golden-battle
    test pins that encoder; the _ProxyMon fidelity notes list where the INPUTS still differ).

    PARTIAL OBSERVABILITY: the engine state is perfect-information; the engine keeps no event log,
    so this proxy cannot know what has been revealed. Default `reveal_opponent=False` exposes the
    opponent as species-only (preview knowledge). A real training loop must track reveals itself
    (RevealLedger + BattleTracker fed from applied instructions) and rebuild the proxy's opponent
    view from that ledger — that wrapper is the unbuilt core of the volume track (module docstring).
    `reveal_opponent=True` is for debugging/fuzzing only: an agent trained on it plays
    perfect-information Pokemon and learns nothing about the fog.
    """

    player_role = "p1"

    def __init__(self, state: poke_engine.State, turn: int = 1, observations: dict | None = None,
                 reveal_opponent: bool = False):
        self._state = state
        self.battle_tag = "rust-battle"
        self.turn = turn
        self.observations = observations or {}
        tables = _try_tables()

        s1, s2 = state.side_one, state.side_two
        self.team = {f"p1: {p.id}": _ProxyMon(p, revealed=True, tables=tables) for p in s1.pokemon}
        self.opponent_team = {
            f"p2: {p.id}": _ProxyMon(p, revealed=reveal_opponent, tables=tables) for p in s2.pokemon
        }
        self.teampreview_opponent_team = list(self.opponent_team.values())
        mine = list(self.team.values())
        theirs = list(self.opponent_team.values())
        self.active_pokemon = mine[int(s1.active_index)] if mine else None
        self.opponent_active_pokemon = theirs[int(s2.active_index)] if theirs else None
        # boosts live on the SIDE in poke-engine; copy onto the active proxy
        if self.active_pokemon is not None:
            self.active_pokemon.boosts = _side_boosts(s1)
        if self.opponent_active_pokemon is not None:
            self.opponent_active_pokemon.boosts = _side_boosts(s2)

        self.available_moves = ([m for m in self.active_pokemon.moves.values()]
                                if self.active_pokemon and not self.active_pokemon.fainted else [])
        self.available_switches = [m for m in mine
                                   if m is not self.active_pokemon and not m.fainted]
        self.can_tera = not any(p.terastallized for p in mine)
        self.force_switch = bool(getattr(s1, "force_switch", False))
        # side_one's result once one side is out of mons; None while the battle is live
        s1_alive = any(p.hp > 0 for p in s1.pokemon)
        s2_alive = any(p.hp > 0 for p in s2.pokemon)
        self.won = None if (s1_alive and s2_alive) else bool(s1_alive)

        self.weather = {str(state.weather): 0} if str(state.weather) not in ("none", "None") else {}
        self.fields = {}
        if str(state.terrain) not in ("none", "None"):
            self.fields[f"{state.terrain}terrain"] = 0
        if getattr(state, "trick_room", False):
            self.fields["trickroom"] = 0
        self.side_conditions = _side_conditions(s1)
        self.opponent_side_conditions = _side_conditions(s2)


def _side_boosts(side) -> dict:
    return {
        "atk": int(side.attack_boost), "def": int(side.defense_boost),
        "spa": int(side.special_attack_boost), "spd": int(side.special_defense_boost),
        "spe": int(side.speed_boost), "accuracy": int(side.accuracy_boost),
        "evasion": int(side.evasion_boost),
    }


def _side_conditions(side) -> dict:
    out: dict[str, int] = {}
    sc = getattr(side, "side_conditions", None)
    if sc is None:
        return out
    for name in ("stealth_rock", "spikes", "toxic_spikes", "sticky_web", "reflect",
                 "light_screen", "aurora_veil", "tailwind"):
        v = int(getattr(sc, name, 0) or 0)
        if v:
            out[name.replace("_", "")] = v
    return out


# ------------------------------------------------------------------------------------ action strings
def action_to_engine_move(battle_proxy: RustBattleProxy, action_idx: int) -> str:
    """Project action index (0-13) -> poke-engine move string. Mirrors env/actions semantics."""
    from fog.constants.schema import ACTIONS, ActionType

    kind, sub = ACTIONS[action_idx]
    if kind is ActionType.FORCED:
        moves = list(battle_proxy.active_pokemon.moves) if battle_proxy.active_pokemon else []
        return moves[0] if moves else "splash"
    if kind in (ActionType.MOVE, ActionType.MOVE_TERA):
        moves = list(battle_proxy.active_pokemon.moves)
        move = moves[sub] if sub < len(moves) else moves[0]
        return f"{move}-tera" if kind is ActionType.MOVE_TERA else move
    # switch: team-order-minus-active; poke-engine takes the BARE pokemon id (no "switch " prefix)
    bench = [m for m in battle_proxy.team.values() if m is not battle_proxy.active_pokemon]
    target = bench[sub] if sub < len(bench) else bench[0]
    return target.species


def smoke_roundtrip(paste_a: str, paste_b: str) -> dict:
    """One engine turn end-to-end: build sides, encode, apply one instruction set. Run this against
    the installed poke-engine version BEFORE any training run (API strings drift across releases)."""
    _require_engine()
    from fog.encoding.observation import Observation

    state = poke_engine.State(
        side_one=paste_to_engine_side(paste_a), side_two=paste_to_engine_side(paste_b)
    )
    proxy = RustBattleProxy(state)
    obs = Observation.from_battle(proxy)
    obs.validate()
    m1 = action_to_engine_move(proxy, 0)
    m2 = list(state.side_two.pokemon)[1].id            # bare id = switch
    instructions = poke_engine.generate_instructions(state, m1, m2)
    applied = False
    if instructions:
        # apply_instructions RETURNS the successor state (verified 0.0.48) — it does not mutate
        new_state = state.apply_instructions(instructions[0])
        applied = new_state.to_string() != state.to_string()
    return {"obs_ok": True, "n_instruction_sets": len(instructions), "applied": applied,
            "moves": (m1, m2), "encoded_entities": obs.entities.shape}
