"""EngineMCTS: the engine's built-in tree search over several sampled opponent worlds.

No network. At each decision:

1. sample ``K`` opponent team sheets from usage priors (``fog.search.usage_worlds``);
2. build one engine state per world (``EngineBattle.fork_for_search`` on the engine backend;
   ``state_recon.reconstruct_battle`` for Showdown play);
3. run ``poke_engine.monte_carlo_tree_search`` on each world with ``budget_ms // K`` ms (or a fixed
   iteration count, for deterministic tests);
4. aggregate AT THE ROOT across worlds: ``p(a) = sum_w b(w) * visits_w(a) / visits_w``, then
   renormalise. Never pick per world and vote (strategy fusion F1, see the search design notes);
5. play the argmax of ``p`` mapped to a legal FOG action index.

Move-string mapping. The engine reports root options as strings: a move id (``stealthrock``), a
switch (``switch greattusk``) or a Tera move (``stealthrock-tera``). The controller
builds the reverse map from ``EngineBattle.engine_action`` over the legal indices of
``action_mask`` (so mask, mapping and the step contract can never disagree), normalising ids with
``to_id`` and prefixing switches with ``switch``. Search entries that do not map to a legal action
are dropped before aggregation and their visits counted in ``unmapped_visits``.

Terastallization. poke-engine 0.0.48 must be built with its ``terastallization`` feature (see
pyproject.toml); without it the search never lists a ``-tera`` root option. With it, the search
lists ``<move>-tera`` for each move while no mon on that side has terastallized, and stops once one
has. ``-tera`` entries are additionally dropped once the side has used Tera (``battle.tera_used``),
and any other entry that is illegal under ``action_mask`` is dropped by the mapping anyway.

The engine's per-option ``total_score / visits`` is in ``[0, 1]`` from the side's own perspective;
``last_value`` is its world-weighted mean over our side's kept visits.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fog.constants.schema import N_ACTIONS

DEFAULT_K = 8
DEFAULT_BUDGET_MS = 1600
_TERA_SUFFIX = "-tera"
# The engine reports internal errors as Rust panics, which Python sees as a ``BaseException``
# subclass. A panicking world is dropped like any other failed world; these are not swallowed.
_NOT_SWALLOWED = (KeyboardInterrupt, SystemExit, GeneratorExit)


# ----------------------------------------------------------------------------- string helpers
def _to_id(value: object) -> str:
    from fog.env.engine_battle import to_id

    return to_id(value)


def normalize_choice(choice: str) -> str:
    """Canonical key for an engine root option: ``switch <id>``, ``<move>`` or ``<move>-tera``."""
    text = str(choice).strip()
    lower = text.lower()
    if lower.startswith("switch "):
        return "switch " + _to_id(text[len("switch "):])
    if lower.endswith(_TERA_SUFFIX):
        return _to_id(text[: -len(_TERA_SUFFIX)]) + _TERA_SUFFIX
    return _to_id(text)


def is_tera_choice(choice: str) -> bool:
    return normalize_choice(choice).endswith(_TERA_SUFFIX)


# ------------------------------------------------------------------------------- core search
@dataclass
class WorldSearch:
    """One world's root statistics for our side (after dropping filtered entries)."""

    weight: float
    visits: dict[str, int]          # normalized choice -> visits (kept entries only)
    total_visits: int               # engine-reported total visits for the world
    kept_visits: int
    dropped_visits: int
    mean_score: float               # visit-weighted mean engine score over kept entries, [0, 1]
    elapsed_s: float = 0.0


@dataclass
class RootSearch:
    policy: dict[str, float]        # aggregated visit fraction per normalized choice (sums to 1)
    value: float                    # world-weighted mean of per-world mean scores
    worlds: list[WorldSearch] = field(default_factory=list)
    failed_worlds: int = 0          # worlds whose search raised or kept no visits


def aggregate_root(worlds: Sequence[WorldSearch]) -> RootSearch:
    """Root aggregation across worlds: ``sum_w weight_w * visits_w(a) / kept_visits_w``.

    Worlds with zero weight or zero kept visits are ignored; world weights are renormalised over
    the rest. Each world is normalised by its KEPT visits, so dropping illegal entries in one world
    does not shrink that world's influence. Raises ``ValueError`` when nothing survives.
    """
    live = [w for w in worlds if w.weight > 0 and w.kept_visits > 0]
    if not live:
        raise ValueError("no world with positive weight and kept visits")
    total_weight = float(sum(w.weight for w in live))
    policy: dict[str, float] = {}
    value = 0.0
    for world in live:
        share = world.weight / total_weight
        for choice, visits in world.visits.items():
            policy[choice] = policy.get(choice, 0.0) + share * visits / world.kept_visits
        value += share * world.mean_score
    mass = sum(policy.values())
    policy = {choice: p / mass for choice, p in policy.items()}
    return RootSearch(policy=policy, value=float(value), worlds=list(worlds))


def _side_entries(result: Any, side: str) -> list:
    return list(result.side_one if side == "p1" else result.side_two)


def search_world(
    state: Any,
    side: str,
    *,
    weight: float = 1.0,
    duration_ms: int = 0,
    iterations: int | None = None,
    threads: int = 1,
    keep: Callable[[str], bool] | None = None,
) -> WorldSearch:
    """Run the built-in search on one engine ``State`` and summarise ``side``'s root."""
    import poke_engine as pe

    if side not in ("p1", "p2"):
        raise ValueError("side must be 'p1' or 'p2'")
    start = time.perf_counter()
    if iterations:
        result = pe.monte_carlo_tree_search(state, iterations=int(iterations), threads=int(threads))
    else:
        result = pe.monte_carlo_tree_search(
            state, duration_ms=max(1, int(duration_ms)), threads=int(threads),
        )
    elapsed = time.perf_counter() - start
    visits: dict[str, int] = {}
    score = 0.0
    kept = dropped = 0
    for entry in _side_entries(result, side):
        choice = normalize_choice(entry.move_choice)
        n = int(entry.visits)
        if keep is not None and not keep(choice):
            dropped += n
            continue
        visits[choice] = visits.get(choice, 0) + n
        kept += n
        score += float(entry.total_score)
    return WorldSearch(
        weight=float(weight), visits=visits, total_visits=int(result.total_visits),
        kept_visits=kept, dropped_visits=dropped,
        mean_score=score / kept if kept else 0.5, elapsed_s=elapsed,
    )


def engine_root_search(
    worlds: Sequence[tuple[Any, float]],
    side: str,
    *,
    budget_ms: int = DEFAULT_BUDGET_MS,
    iterations: int | None = None,
    threads: int = 1,
    keep: Callable[[str], bool] | None = None,
    on_error: Callable[[BaseException, Any], None] | None = None,
) -> RootSearch:
    """Search each ``(State, weight)`` world and aggregate at the root.

    ``budget_ms`` is split evenly across worlds; a positive ``iterations`` overrides it with a
    fixed per-world iteration count. ``keep`` filters normalized choices (illegal / Tera) before
    aggregation. Worlds whose search raises or keeps no visits are counted in ``failed_worlds``.
    Raises ``ValueError`` if no world survives.
    """
    if not worlds:
        raise ValueError("no worlds to search")
    per_world_ms = max(1, int(budget_ms) // len(worlds))
    results: list[WorldSearch] = []
    failed = 0
    for state, weight in worlds:
        try:
            ws = search_world(
                state, side, weight=weight, duration_ms=per_world_ms, iterations=iterations,
                threads=threads, keep=keep,
            )
        except _NOT_SWALLOWED:
            raise
        except BaseException as exc:  # noqa: BLE001 - a failed world (or engine panic) is dropped
            failed += 1
            if on_error is not None:
                on_error(exc, state)
            continue
        if ws.kept_visits <= 0:
            failed += 1
        results.append(ws)
    root = aggregate_root(results)
    root.failed_worlds = failed
    return root


# ------------------------------------------------------------------------------ action mapping
def engine_choice_map(battle: Any, role: str, mask: np.ndarray | None = None) -> dict[str, int]:
    """Normalized engine choice -> legal FOG action index, built from ``engine_action``.

    The lowest index wins when two indices produce the same string (e.g. a padded move list).
    The forced slot (13) is registered under both its bare and ``switch`` keys, since the engine
    may report a forced replacement either way.
    """
    if mask is None:
        mask = battle.action_mask(role)
    out: dict[str, int] = {}
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        raw = str(battle.engine_action(role, idx))
        if 8 <= idx <= 12:
            out.setdefault("switch " + _to_id(raw), idx)
            continue
        out.setdefault(normalize_choice(raw), idx)
        if idx == 13:
            out.setdefault("switch " + _to_id(raw), idx)
    return out


def policy_to_actions(policy: dict[str, float], choice_map: dict[str, int]) -> np.ndarray:
    """[14] action distribution from a normalized-choice policy (unmapped choices dropped)."""
    dist = np.zeros(N_ACTIONS, dtype=np.float64)
    for choice, p in policy.items():
        idx = choice_map.get(choice)
        if idx is not None:
            dist[idx] += p
    total = dist.sum()
    return dist / total if total > 0 else dist


# ------------------------------------------------------------------------------ controllers
class _EngineMCTSBase:
    name = "EngineMCTS"

    def __init__(self, *, k: int = DEFAULT_K, budget_ms: int = DEFAULT_BUDGET_MS,
                 iterations: int | None = None, threads: int = 1, seed: int = 0,
                 tables=None) -> None:
        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.budget_ms = int(budget_ms)
        self.iterations = int(iterations) if iterations else None
        self.threads = int(threads)
        self.tables = tables
        self.rng = random.Random(seed)
        self.counters = {
            "decisions": 0, "searched": 0, "single_option": 0, "worlds_sampled": 0,
            "worlds_dropped": 0, "fallbacks": 0, "unmapped_visits": 0, "tera_entries_dropped": 0,
            "search_seconds": 0.0,
        }
        self.failure_reasons: dict[str, int] = {}
        self.unmapped_choices: dict[str, int] = {}    # engine root option -> worlds it was dropped
        self.failed_states: list[str] = []            # first few engine states whose search raised
        self.last_policy: np.ndarray | None = None   # [14] aggregated visit distribution
        self.last_value: float | None = None          # engine score for our side, [0, 1]
        self.last_root: RootSearch | None = None

    def _fail(self, exc: BaseException) -> None:
        reason = f"{type(exc).__name__}: {exc}"[:200]
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def _search_error(self, exc: BaseException, state: Any) -> None:
        self._fail(exc)
        if len(self.failed_states) < 5:
            try:
                self.failed_states.append(state.to_string())
            except Exception:  # noqa: BLE001 - diagnostics only
                pass

    def _decide(self, mask: np.ndarray, choice_map: dict[str, int], tera_used: bool,
                revealed: np.ndarray, build_state: Callable[[np.ndarray, int], Any],
                side: str, fallback: Callable[[], int]) -> int:
        """Shared decision body. Returns a legal action index."""
        from fog.search.usage_worlds import sample_usage_worlds

        self.counters["decisions"] += 1
        self.last_root = None
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            self.counters["single_option"] += 1
            self.last_policy = np.zeros(N_ACTIONS)
            self.last_policy[int(legal[0])] = 1.0
            self.last_value = None
            return int(legal[0])

        dropped = {"unmapped": 0, "tera": 0}

        def keep(choice: str) -> bool:
            if tera_used and choice.endswith(_TERA_SUFFIX):
                dropped["tera"] += 1
                return False
            if choice not in choice_map:
                dropped["unmapped"] += 1
                self.unmapped_choices[choice] = self.unmapped_choices.get(choice, 0) + 1
                return False
            return True

        states: list[tuple[Any, float]] = []
        try:
            tokens, weights = sample_usage_worlds(
                revealed, self.k, seed=self.rng.randrange(2**31), tables=self.tables,
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            tokens, weights = np.zeros((0, 6, 15), dtype=np.int64), np.zeros(0)
        self.counters["worlds_sampled"] += len(tokens)
        for world, weight in zip(tokens, weights):
            try:
                states.append((build_state(world, self.rng.randrange(2**31)), float(weight)))
            except _NOT_SWALLOWED:
                raise
            except BaseException as exc:  # noqa: BLE001 - failed worlds are dropped and counted
                self.counters["worlds_dropped"] += 1
                self._fail(exc)

        root = None
        if states:
            start = time.perf_counter()
            try:
                root = engine_root_search(
                    states, side, budget_ms=self.budget_ms, iterations=self.iterations,
                    threads=self.threads, keep=keep, on_error=self._search_error,
                )
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)
                self.counters["worlds_dropped"] += len(states)
            else:
                self.counters["worlds_dropped"] += root.failed_worlds
                self.counters["unmapped_visits"] += sum(
                    w.dropped_visits for w in root.worlds
                )
            self.counters["search_seconds"] += time.perf_counter() - start
            self.counters["tera_entries_dropped"] += dropped["tera"]

        if root is not None:
            dist = policy_to_actions(root.policy, choice_map) * mask
            if dist.sum() > 0:
                dist = dist / dist.sum()
                best = int(np.argmax(dist))
                if mask[best]:
                    self.counters["searched"] += 1
                    self.last_policy = dist
                    self.last_value = root.value
                    self.last_root = root
                    return best

        self.counters["fallbacks"] += 1
        self.last_policy = None
        self.last_value = None
        action = int(fallback())
        if not mask[action]:  # defensive: never return an illegal action
            action = int(legal[0])
        return action


class EngineMCTSPolicy(_EngineMCTSBase):
    """Engine-backend controller with the ``choose(battle, role) -> action index`` contract of
    ``fog.env.engine_scripted`` policies. Worlds come from ``battle.fork_for_search``.

    Counters: ``decisions``, ``searched``, ``single_option`` (one legal action, no search),
    ``worlds_sampled``, ``worlds_dropped``, ``fallbacks`` (MaxDamage), ``unmapped_visits``.
    ``last_policy`` / ``last_value`` hold the most recent aggregated root distribution over the
    14 action indices and the engine value for our side (``None`` when no search ran).
    """

    def __init__(self, *, k: int = DEFAULT_K, budget_ms: int = DEFAULT_BUDGET_MS,
                 iterations: int | None = None, threads: int = 1, seed: int = 0,
                 tables=None) -> None:
        super().__init__(k=k, budget_ms=budget_ms, iterations=iterations, threads=threads,
                         seed=seed, tables=tables)
        from fog.env.engine_scripted import EngineMaxDamagePolicy

        self.fallback_policy = EngineMaxDamagePolicy(tables=self.tables, seed=seed)

    def choose(self, battle: Any, role: str) -> int:
        from fog.search.usage_worlds import public_reveal_grid

        mask = battle.action_mask(role)
        choice_map = engine_choice_map(battle, role, mask)
        revealed = public_reveal_grid(battle, role, self.tables)

        def build_state(world: np.ndarray, seed: int):
            return battle.fork_for_search(role, world, seed=seed).state

        return self._decide(
            mask, choice_map, bool(battle.tera_used.get(role, False)), revealed, build_state,
            role, lambda: self.fallback_policy.choose(battle, role),
        )


def simpleheuristics_action(battle: Any, rng: random.Random | None = None) -> int:
    """SimpleHeuristics' choice on a poke-env battle, as a FOG action index.

    poke-env's baseline uses Python's module-global RNG in fallback cases; that call is isolated
    behind ``rng`` so a seeded game stays reproducible.
    """
    from poke_env.player import SimpleHeuristicsPlayer

    from fog.env.actions import map_order_to_action

    state = random.getstate()
    random.seed((rng or random.Random(0)).getrandbits(128))
    try:
        order = SimpleHeuristicsPlayer.choose_singles_move(battle)[0]
    finally:
        random.setstate(state)
    return int(map_order_to_action(battle, order))


def showdown_choice_map(battle: Any, mask: np.ndarray) -> dict[str, int]:
    """Normalized engine choice -> legal FOG action index on a poke-env battle.

    Built from poke-env's slot helpers, so indices use the same convention as
    ``action_legality_mask``; only indices legal under ``mask`` are registered.
    """
    from fog.env.actions import available_move_by_slot, available_switch_by_slot, move_key

    out: dict[str, int] = {}
    for slot, move in available_move_by_slot(battle).items():
        key = _to_id(move_key(move))
        if mask[slot]:
            out.setdefault(key, slot)
        if mask[4 + slot]:
            out.setdefault(key + _TERA_SUFFIX, 4 + slot)
    for slot, mon in available_switch_by_slot(battle).items():
        if mask[8 + slot]:
            out.setdefault("switch " + _to_id(getattr(mon, "species", mon)), 8 + slot)
    if mask[13]:
        # Struggle / recharge / a lone forced option: the engine reports these as a bare move id.
        for move in getattr(battle, "available_moves", None) or []:
            out.setdefault(_to_id(move_key(move)), 13)
        out.setdefault("none", 13)
    return out


class ShowdownEngineMCTS(_EngineMCTSBase):
    """EngineMCTS on a poke-env ``Battle`` (``showdown_direct`` / live Showdown play).

    ``choose(battle, tracker, role)`` takes the searcher's poke-env ``Battle`` and live
    ``BattleTracker``. Each world is built with ``fog.env.state_recon.reconstruct_battle`` (as
    ``ShowdownWorldStepper`` does). Engine choices map to FOG indices through poke-env's own slot
    helpers, so the returned index is legal under ``action_legality_mask``. The searcher's own
    request is imposed on each world (``apply_request_overlay``). Fallback when every
    world fails: SimpleHeuristics (counted in ``fallbacks``).
    """

    def choose(self, battle: Any, tracker: Any, role: str = "p1",
               fallback: Callable[[], int] | None = None) -> int:
        from fog.encoding.reveals import opponent_views_from_battle, tokens_from_views
        from fog.env.actions import action_legality_mask
        from fog.env.state_recon import reconstruct_battle

        mask = np.asarray(action_legality_mask(battle), dtype=bool)
        choice_map = showdown_choice_map(battle, mask)
        revealed = tokens_from_views(opponent_views_from_battle(battle, self.tables))
        tera_used = not bool(getattr(battle, "can_tera", False))

        def build_state(world: np.ndarray, seed: int):
            state = reconstruct_battle(battle, world, self.tables, tracker, role=role,
                                       seed=seed).state
            return apply_request_overlay(state, battle, role)

        if fallback is None:
            def fallback() -> int:
                return simpleheuristics_action(battle, self.rng)

        return self._decide(mask, choice_map, tera_used, revealed, build_state, role, fallback)


class ShowdownEngineMCTSController:
    """``fog.env.showdown_local_runner`` controller around :class:`ShowdownEngineMCTS`.

    ``act(battle)`` keeps one ``BattleTracker`` per battle (fed like ``FogPlayer`` feeds its own),
    searches, and returns the Showdown order for a legal action index. Per-game counters
    (``decisions``, ``worlds_dropped``, ``fallbacks``, ``rejected``, ``decision_seconds``, ...)
    are handed out once per game by ``take_game_counters``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.search = ShowdownEngineMCTS(**kwargs)
        self.name = "EngineMCTS"
        self._trackers: dict[str, tuple[Any, set[int]]] = {}
        self._game_start = self._snapshot()
        self.rejected = 0
        self.decision_seconds = 0.0
        self.order_errors = 0

    def _snapshot(self) -> dict[str, float]:
        return {**self.search.counters, "rejected": getattr(self, "rejected", 0),
                "decision_seconds": getattr(self, "decision_seconds", 0.0),
                "order_errors": getattr(self, "order_errors", 0)}

    def _tracker(self, battle: Any) -> Any:
        from fog.encoding.history import observations_by_turn
        from fog.encoding.tracker import BattleTracker

        tag = str(getattr(battle, "battle_tag", "battle"))
        if tag not in self._trackers:
            role = getattr(battle, "player_role", None) or "p1"
            self._trackers[tag] = (BattleTracker(my_role=role), set())
        tracker, fed = self._trackers[tag]
        tracker.record_private_team(battle)
        obs_map = observations_by_turn(battle)
        current = int(getattr(battle, "turn", 0) or 0)
        for t in sorted(obs_map):
            if t < current and t not in fed:
                tracker.feed_turn(t, getattr(obs_map[t], "events", None) or [])
                fed.add(t)
        current_obs = obs_map.get(current)
        if current_obs is not None:
            tracker.feed_turn(current, getattr(current_obs, "events", None) or [])
        return tracker

    def act(self, battle: Any) -> Any:
        from fog.env.actions import action_legality_mask, map_action_to_order
        from fog.env.showdown_local_runner import LocalOrderFactory

        start = time.perf_counter()
        role = getattr(battle, "player_role", None) or "p1"
        tracker = self._tracker(battle)
        action = self.search.choose(battle, tracker, role)
        if not action_legality_mask(battle)[action]:  # never send an illegal action
            self.order_errors += 1
            action = simpleheuristics_action(battle, self.search.rng)
        try:
            order = map_action_to_order(battle, action, LocalOrderFactory())
        except AssertionError:
            self.order_errors += 1
            order = map_action_to_order(
                battle, simpleheuristics_action(battle, self.search.rng), LocalOrderFactory()
            )
        self.decision_seconds += time.perf_counter() - start
        return order

    def discard_last_uncommitted_decision(self, battle: Any) -> None:
        """The runner calls this when Showdown rejected the last choice."""
        self.rejected += 1

    def _battle_finished(self, battle: Any) -> None:
        self._trackers.pop(str(getattr(battle, "battle_tag", "battle")), None)

    def take_game_counters(self) -> dict[str, Any]:
        """Counters accumulated since the previous call (one game), plus failure reasons."""
        now = self._snapshot()
        delta: dict[str, Any] = {key: now[key] - self._game_start.get(key, 0) for key in now}
        self._game_start = now
        self._trackers.clear()
        if self.search.failure_reasons:
            delta["failure_reasons"] = dict(self.search.failure_reasons)
            self.search.failure_reasons.clear()
        if self.search.failed_states:
            delta["failed_states"] = list(self.search.failed_states)
            self.search.failed_states.clear()
        return delta


def showdown_controller(**kwargs: Any) -> ShowdownEngineMCTSController:
    """Factory for ``showdown_direct_selection --controller LABEL=fog.search.engine_mcts:showdown_controller``.

    Keyword arguments go to :class:`ShowdownEngineMCTS` (``k``, ``budget_ms``, ``iterations``,
    ``threads``, ``seed``).
    """
    return ShowdownEngineMCTSController(**kwargs)


# ------------------------------------------------------------------ request overlay (Showdown)
def _rebuild(obj: Any, cls: Any, **overrides: Any) -> Any:
    """A copy of an immutable engine object with some constructor fields replaced."""
    import inspect

    kwargs = {name: getattr(obj, name) for name in inspect.signature(cls).parameters}
    kwargs.update(overrides)
    return cls(**kwargs)


def apply_request_overlay(state: Any, battle: Any, role: str) -> Any:
    """Impose the searcher's authoritative Showdown request on a reconstructed engine state.

    * a pending forced switch (faint or pivot) sets the side's ``force_switch`` flag, so the root
      offers only switches instead of a full turn for the leaving mon;
    * a request that offers no switch because the active mon is trapped (Magnet Pull, Shadow Tag,
      Arena Trap, binding moves ...) sets ``force_trapped``;
    * the active mon's moves that the request does not offer (0 PP, Disable, Taunt, Encore, choice
      lock, ...) are marked ``disabled``.
    Everything else is left as reconstructed.
    """
    import poke_engine as pe

    from fog.env.actions import move_key

    side_name = "side_one" if role == "p1" else "side_two"
    side = getattr(state, side_name)
    overrides: dict[str, Any] = {}
    if bool(getattr(battle, "force_switch", False)):
        overrides["force_switch"] = True
    else:
        if bool(getattr(battle, "trapped", False)) and not getattr(battle, "available_switches",
                                                                    None):
            overrides["force_trapped"] = True
        offered = {_to_id(move_key(m)) for m in (getattr(battle, "available_moves", None) or [])}
        if offered and "struggle" not in offered and "recharge" not in offered:
            pokemon = list(side.pokemon)
            active = int(side.active_index)
            mon = pokemon[active]
            moves = [
                _rebuild(m, pe.Move, disabled=bool(m.disabled) or _to_id(m.id) not in offered)
                if _to_id(m.id) not in ("none", "") else m
                for m in mon.moves
            ]
            if any(not m.disabled for m in moves):
                pokemon[active] = _rebuild(mon, pe.Pokemon, moves=moves)
                overrides["pokemon"] = pokemon
    if not overrides:
        return state
    return _rebuild(state, pe.State, **{side_name: _rebuild(side, pe.Side, **overrides)})
