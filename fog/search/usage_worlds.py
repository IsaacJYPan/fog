"""Opponent team-sheet worlds sampled from usage statistics (no network).

Given the opponent's public reveal grid ``[6, 15]`` (SET_SLOT_ATTRS order, negative = unknown),
draw ``K`` complete team sheets. Every field the searcher has seen is clamped; everything else
comes from the pinned usage priors in ``fog/constants/generated``:

* known species: the per-species completion already used by the replay imputer,
  ``fog.training.replay_policy_data.complete_private_team`` (item / ability / move priors from
  ``priors.json``, nature + EV spreads from ``spreads.json`` via ``TABLES.spreads``, Tera priors);
* unknown species (rare in Gen 9 OU, where team preview shows all six): drawn one at a time from
  the teammate co-occurrence matrix in ``priors.json`` conditioned on the species already on the
  team, mixed with a usage marginal. ``priors.json`` stores no direct species-usage marginal, so
  the marginal is the co-occurrence row mass (proportional to a species' weighted team count).

A publicly observed ``<NO_ITEM>`` is a current-state fact, not the submitted item: it is treated
as hidden for completion (the original item is sampled), and ``keep_mask_from_reveals`` is applied
with ``no_item_token`` so it does not reject that world. ``fork_for_search`` re-applies the public
current item when the item reveal is known.

Draws are i.i.d. from the prior, so weights are empirical frequencies after deduplication.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from fog.constants.schema import SET_N_ATTRS

N_SLOTS = 6
# Mixture weight of the unconditional usage marginal in the teammate draw. Keeps species with no
# recorded co-occurrence reachable and protects against an all-zero conditional row.
MARGINAL_MIX = 0.1


def public_reveal_grid(battle, role: str, tables=None) -> np.ndarray:
    """The opponent's public reveal grid as seen by ``role`` (same extraction as the belief input).

    ``battle`` is an ``EngineBattle``; for a poke-env battle use
    ``tokens_from_views(opponent_views_from_battle(battle, tables))`` directly.
    """
    from fog.encoding.reveals import opponent_views_from_battle, tokens_from_views

    if tables is None:
        from fog.constants.tables import TABLES as tables
    return tokens_from_views(opponent_views_from_battle(battle.view(role), tables))


@lru_cache(maxsize=1)
def _species_prior() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(table species indices [S], marginal [S], row-normalised teammate conditional [S, S]).

    Restricted to species that have usage rows (``item_priors``) and a real table index.
    """
    from fog.constants.tables import TABLES
    from fog.training.replay_policy_data import replay_set_priors

    priors = replay_set_priors()
    order = list(priors["species_order"])
    cooc = np.clip(np.asarray(priors["teammate_cooccurrence"], dtype=np.float64), 0.0, None)
    used = set(priors.get("item_priors", {}))
    ids = TABLES.species["ids"]
    keep_rows, table_idx = [], []
    for row, sid in enumerate(order):
        if sid not in used:
            continue
        idx = TABLES.species_index(sid)
        if not 0 <= idx < len(ids) or ids[idx] != sid:
            continue
        keep_rows.append(row)
        table_idx.append(idx)
    keep = np.asarray(keep_rows, dtype=np.int64)
    sub = cooc[np.ix_(keep, keep)]
    marginal = sub.sum(axis=1)
    if marginal.sum() <= 0:
        marginal = np.ones(len(keep), dtype=np.float64)
    marginal = marginal / marginal.sum()
    row_mass = sub.sum(axis=1, keepdims=True)
    conditional = np.divide(sub, row_mass, out=np.zeros_like(sub), where=row_mass > 0)
    return np.asarray(table_idx, dtype=np.int64), marginal, conditional


def _sample_species(team: list[int], rng: np.random.Generator) -> int:
    """One usage-weighted species not already on ``team`` (table index)."""
    table_idx, marginal, conditional = _species_prior()
    position = {int(t): i for i, t in enumerate(table_idx)}
    known = [position[s] for s in team if s in position]
    if known:
        cond = conditional[known].mean(axis=0)
        probs = (1.0 - MARGINAL_MIX) * cond + MARGINAL_MIX * marginal
    else:
        probs = marginal.copy()
    on_team = set(team)
    for i, t in enumerate(table_idx):
        if int(t) in on_team:
            probs[i] = 0.0
    if probs.sum() <= 0:
        raise RuntimeError("no usage species left to fill an unrevealed team slot")
    probs = probs / probs.sum()
    return int(table_idx[int(rng.choice(len(table_idx), p=probs))])


def complete_world(revealed: np.ndarray, *, seed: int, tables=None) -> np.ndarray:
    """One complete ``[6, 15]`` sheet consistent with ``revealed`` (see module docstring)."""
    from fog.training.replay_policy_data import complete_private_team

    if tables is None:
        from fog.constants.tables import TABLES as tables
    grid = np.asarray(revealed, dtype=np.int64).copy()
    if grid.shape != (N_SLOTS, SET_N_ATTRS):
        raise ValueError(f"revealed grid must be [6,{SET_N_ATTRS}], got {grid.shape}")
    no_item = tables.item_index("<NO_ITEM>")
    grid[grid[:, 1] == no_item, 1] = -1

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EC1E5]))
    team = [int(s) for s in grid[:, 0] if s >= 0]
    for slot in range(N_SLOTS):
        if grid[slot, 0] < 0:
            species = _sample_species(team, rng)
            grid[slot, 0] = species
            team.append(species)
    completed, _ = complete_private_team(grid, seed=seed)
    return completed


def sample_usage_worlds(
    revealed: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    dedupe: bool = True,
    max_attempts_per_world: int = 4,
    tables=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``k`` opponent worlds. Returns ``(tokens [K', 6, 15] int64, weights [K'] float64)``.

    ``K' == k`` without ``dedupe``; with it, identical sheets merge and their weights add.
    Every returned world passes ``keep_mask_from_reveals(revealed, ...)``; a draw that does not
    (which clamping should make impossible) is redrawn, and the call raises if none survive.
    """
    from fog.search.evidence import keep_mask_from_reveals

    if tables is None:
        from fog.constants.tables import TABLES as tables
    if k < 1:
        raise ValueError("k must be positive")
    revealed = np.asarray(revealed, dtype=np.int64)
    no_item = tables.item_index("<NO_ITEM>")
    seeds = np.random.SeedSequence(int(seed)).generate_state(k * max_attempts_per_world, np.uint32)
    worlds: list[np.ndarray] = []
    for draw in range(k):
        for attempt in range(max_attempts_per_world):
            world = complete_world(
                revealed, seed=int(seeds[draw * max_attempts_per_world + attempt]), tables=tables,
            )
            if keep_mask_from_reveals(revealed, world[None], no_item_token=no_item)[0]:
                worlds.append(world)
                break
    if not worlds:
        raise RuntimeError("no sampled world is consistent with the public reveals")
    if not dedupe:
        tokens = np.stack(worlds)
        return tokens, np.full(len(tokens), 1.0 / len(tokens))
    index: dict[bytes, int] = {}
    unique: list[np.ndarray] = []
    counts: list[int] = []
    for world in worlds:
        key = world.tobytes()
        if key in index:
            counts[index[key]] += 1
        else:
            index[key] = len(unique)
            unique.append(world)
            counts.append(1)
    weights = np.asarray(counts, dtype=np.float64)
    return np.stack(unique), weights / weights.sum()
