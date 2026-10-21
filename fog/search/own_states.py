"""Candidate private teams from the opponent's point of view.

`EnumeratedPoolStates` filters a finite pool by the searcher's public reveals and
renormalizes the pool prior. `StablePoolOwnStates` keeps the full pool in fixed
order for a persistent belief table; the table handles reveal filtering.
`SampledOwnStates` samples the self-belief model on the opponent's observation,
deduplicates the draws, and weights them by frequency.

All sources include the true team. Enumeration raises if reveals exclude it;
sampling inserts it with one pseudo-count if absent. Sampled weights therefore
include this smoothing and are not exact model probabilities.

`states(battle, role)` returns tokens [K, 6, 15] in SET_SLOT_ATTRS order,
normalized weights [K], and the true team's index. Small finite pools are useful
for debugging but can be memorized; agreement there need not transfer off-pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OwnStates:
    tokens: np.ndarray    # [K, 6, 15] int64
    weights: np.ndarray   # [K] float64, sums to 1
    true_index: int

    def __post_init__(self) -> None:
        from fog.constants.schema import SET_N_ATTRS

        if self.tokens.ndim != 3 or self.tokens.shape[1:] != (6, SET_N_ATTRS):
            raise ValueError(f"tokens must be [K,6,{SET_N_ATTRS}], got {self.tokens.shape}")
        if len(self.weights) != len(self.tokens):
            raise ValueError("weights/tokens length mismatch")
        if not 0 <= self.true_index < len(self.tokens):
            raise ValueError("true_index out of range")

    @property
    def k(self) -> int:
        return int(self.tokens.shape[0])


def _my_revealed_views(battle, role: str, tables):
    """SlotViews of `role`'s team as the OPPONENT currently sees it: the opponent's view object's
    `opponent_team` IS role's censored side, so the standard reveal extraction applies directly."""
    from fog.encoding.reveals import opponent_views_from_battle

    opp_role = "p2" if role == "p1" else "p1"
    return opponent_views_from_battle(battle.view(opp_role), tables)


class EnumeratedPoolStates:
    """Exact own-state source over a finite team pool. `prior` defaults to uniform (the runner
    samples pool teams uniformly); pass explicit weights for usage-weighted pools."""

    def __init__(self, pool_pastes: list[str], tables=None, prior: np.ndarray | None = None):
        from fog.env.team_truth import PoolTruth

        if tables is None:
            from fog.constants.tables import TABLES as tables
        self.tables = tables
        self.pool = PoolTruth(pool_pastes, tables)
        n = len(pool_pastes)
        self.prior = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=np.float64)
        if len(self.prior) != n or not np.isclose(self.prior.sum(), 1.0):
            raise ValueError("prior must be a distribution over the pool")

    def states(self, battle, role: str) -> OwnStates:
        idxs = self.pool.consistent_indices(_my_revealed_views(battle, role, self.tables))
        if not idxs:
            raise RuntimeError(
                "no pool team is consistent with the searcher's reveal state — either the true "
                "team is not from this pool or the reveal census over-revealed (a bug: reveals "
                "are conservative by design)."
            )
        tokens = np.stack([self.pool.truth_tokens(i) for i in idxs])
        weights = self.prior[idxs]
        weights = weights / weights.sum()

        truth = battle.truth_tokens(role)
        true_pos = next(
            (j for j, i in enumerate(idxs) if np.array_equal(self.pool.truth_tokens(i), truth)),
            None,
        )
        if true_pos is None:
            raise RuntimeError(
                "true team not among the consistent pool states — the same census bug as above "
                "(the true state must never be filtered out by its own reveals)."
            )
        return OwnStates(tokens=tokens, weights=weights, true_index=true_pos)


class StablePoolOwnStates:
    """Keep all teams in pool order so persistent belief-table axes stay stable.

    The table handles reveal filtering. Search drops zero-mass candidates before
    forking, since a reveal-inconsistent team cannot produce a coherent world.
    """

    def __init__(self, pool_pastes: list[str], tables=None, prior: np.ndarray | None = None):
        from fog.env.team_truth import PoolTruth

        if tables is None:
            from fog.constants.tables import TABLES as tables
        self.tables = tables
        self.pool = PoolTruth(pool_pastes, tables)
        n = len(pool_pastes)
        self.tokens = np.stack([self.pool.truth_tokens(i) for i in range(n)])
        self.prior = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=np.float64)
        if len(self.prior) != n or not np.isclose(self.prior.sum(), 1.0):
            raise ValueError("prior must be a distribution over the pool")

    def states(self, battle, role: str) -> OwnStates:
        truth = battle.truth_tokens(role)
        true_pos = next(
            (i for i in range(len(self.tokens)) if np.array_equal(self.tokens[i], truth)), None)
        if true_pos is None:
            raise RuntimeError("the searcher's true team is not in this pool — a stable-pool "
                               "source only makes sense when the axes contain the truth")
        return OwnStates(tokens=self.tokens.copy(), weights=self.prior.copy(),
                         true_index=true_pos)


class SampledOwnStates:
    """Learned-sampler own-state source: SelfBeliefPredictor ancestral samples on the opponent's
    observation. `k` is the returned state count; `oversample` extra draws sharpen the empirical
    frequency weights before dedup/truncation."""

    def __init__(self, agent, tables=None, *, k: int = 16, oversample: int = 4,
                 temperature: float = 1.0, seed: int = 0):
        if tables is None:
            from fog.constants.tables import TABLES as tables
        self.agent = agent
        self.tables = tables
        self.k = int(k)
        self.oversample = max(1, int(oversample))
        self.temperature = float(temperature)
        self.seed = int(seed)
        self._calls = 0

    def states(self, battle, role: str) -> OwnStates:
        import torch

        from fog.encoding.observation import Observation
        from fog.env.engine_runner import _device, _stack_obs

        opp_role = "p2" if role == "p1" else "p1"
        obs = Observation.from_battle(battle.view(opp_role), tracker=battle.trackers[opp_role])
        batch = _stack_obs([obs], _device(self.agent))

        self._calls += 1
        gen = torch.Generator(device=_device(self.agent))  # sample_team's multinomial device
        gen.manual_seed(self.seed * 1_000_003 + self._calls)
        with torch.no_grad():
            contexts = self.agent.encode_contexts(batch)
            belief_context = self.agent.belief_context(contexts)
            draws = self.agent.self_belief.sample_team(
                belief_context, batch["opp_revealed"], k=self.k * self.oversample,
                temperature=self.temperature, generator=gen,
            )[0].cpu().numpy().astype(np.int64)                       # [k*oversample, 6, 15]

        counts: dict[bytes, int] = {}
        by_key: dict[bytes, np.ndarray] = {}
        for w in draws:
            key = w.tobytes()
            counts[key] = counts.get(key, 0) + 1
            by_key.setdefault(key, w)

        truth = np.asarray(battle.truth_tokens(role), dtype=np.int64)
        tkey = truth.tobytes()
        if tkey not in counts:  # add-one insertion: present, near its (low) model probability
            counts[tkey] = 1
            by_key[tkey] = truth

        # keep the k most frequent, truth always among them
        order = sorted(counts, key=lambda key: (key != tkey, -counts[key]))[: self.k]
        tokens = np.stack([by_key[key] for key in order])
        weights = np.array([counts[key] for key in order], dtype=np.float64)
        weights = weights / weights.sum()
        return OwnStates(tokens=tokens, weights=weights,
                         true_index=[key for key in order].index(tkey))
