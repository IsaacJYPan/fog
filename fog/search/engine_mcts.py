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
switch (``switch greattusk``) or, in principle, a Tera move (``stealthrock-tera``). The controller
builds the reverse map from ``EngineBattle.engine_action`` over the legal indices of
``action_mask`` (so mask, mapping and the step contract can never disagree), normalising ids with
``to_id`` and prefixing switches with ``switch``. Search entries that do not map to a legal action
are dropped before aggregation and their visits counted in ``unmapped_visits``.

Terastallization. Probed on poke-engine 0.0.48 (Gen 9 build): the built-in search never lists a
``-tera`` root option, before or after a side has terastallized, so EngineMCTS never
terastallizes. Should a later build list them, ``-tera`` entries are dropped once the side has used
Tera (``battle.tera_used``), and any other entry that is illegal under ``action_mask`` is dropped
by the mapping anyway.

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
        except Exception:  # noqa: BLE001 - a failed world is dropped and counted
            failed += 1
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
        self.last_policy: np.ndarray | None = None   # [14] aggregated visit distribution
        self.last_value: float | None = None          # engine score for our side, [0, 1]
        self.last_root: RootSearch | None = None

    def _fail(self, exc: BaseException) -> None:
        reason = f"{type(exc).__name__}: {exc}"[:200]
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

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
            except Exception as exc:  # noqa: BLE001 - failed worlds are dropped and counted
                self.counters["worlds_dropped"] += 1
                self._fail(exc)

        root = None
        if states:
            start = time.perf_counter()
            try:
                root = engine_root_search(
                    states, side, budget_ms=self.budget_ms, iterations=self.iterations,
                    threads=self.threads, keep=keep,
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


class ShowdownEngineMCTS(_EngineMCTSBase):
    """UNTESTED adapter for Showdown play (Showdown is not available where this was written).

    ``choose(battle, tracker, role)`` takes the searcher's poke-env ``Battle`` and live
    ``BattleTracker``. Each world is built with ``fog.env.state_recon.reconstruct_battle`` (as
    ``ShowdownWorldStepper`` does). Engine choices map to FOG indices through poke-env's own slot
    helpers (``available_move_by_slot`` / ``available_switch_by_slot``), so the returned index uses
    the same convention as ``action_legality_mask``. Fallback: the first legal non-Tera move (a
    SimpleHeuristics fallback needs a poke-env player and is left to the caller).
    """

    def choose(self, battle: Any, tracker: Any, role: str = "p1") -> int:
        from fog.encoding.reveals import opponent_views_from_battle, tokens_from_views
        from fog.env.actions import (
            action_legality_mask,
            available_move_by_slot,
            available_switch_by_slot,
        )
        from fog.env.state_recon import reconstruct_battle

        mask = np.asarray(action_legality_mask(battle), dtype=bool)
        choice_map: dict[str, int] = {}
        for slot, move in available_move_by_slot(battle).items():
            choice_map.setdefault(_to_id(getattr(move, "id", move)), slot)
            if mask[4 + slot]:
                choice_map.setdefault(_to_id(getattr(move, "id", move)) + _TERA_SUFFIX, 4 + slot)
        for slot, mon in available_switch_by_slot(battle).items():
            choice_map.setdefault("switch " + _to_id(getattr(mon, "species", mon)), 8 + slot)
        revealed = tokens_from_views(opponent_views_from_battle(battle, self.tables))
        tera_used = not bool(getattr(battle, "can_tera", False))

        def build_state(world: np.ndarray, seed: int):
            return reconstruct_battle(battle, world, self.tables, tracker, role=role,
                                      seed=seed).state

        def fallback() -> int:
            legal = np.flatnonzero(mask)
            moves = [int(i) for i in legal if i < 4]
            return moves[0] if moves else int(legal[0])

        return self._decide(mask, choice_map, tera_used, revealed, build_state, role, fallback)
