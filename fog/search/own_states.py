"""Sources of MY possible private states, from the opponent's point of view (SEARCH_PLAN §4.1).

Joint search (fog/search/joint.py) reasons over K teams the opponent could believe the searcher
has. This module supplies them behind one interface, two implementations:

  * `EnumeratedPoolStates` — EXACT: on pool runs the opponent's belief over my team ranges over a
    finite set of known pastes (POOL-A: 8, pool_gen: 128). Filter to the pastes consistent with
    everything I have revealed so far and weight by the pool prior. No sampling error, no missed
    states — the debugging/validation rung. (An effect that only shows on the memorizable 8-team
    pool is an artifact; pool_gen is the claim-bearing pool — HANDOFF's standing caveat.)
  * `SampledOwnStates` — the scalability arm: ancestral samples from the agent's own
    SelfBeliefPredictor (it inherits BeliefModel.sample_team) run on the OPPONENT's observation of
    the battle, so hidden attributes are drawn from the learned model and revealed ones are
    clamped. Weights are the empirical sample frequencies after dedup (an unbiased proxy for the
    model's own probabilities that needs no extra scoring pass — v1, documented in SEARCH_PLAN §6).

Both ALWAYS include the true state (the doc rule: the system knows which state is real, but its
weight must stay whatever the opponent would assign it):
  * enumeration: the true team is a pool member, so it is present whenever it is consistent
    (inconsistency would be a reveal-census bug — raised, not papered over);
  * sampling: if absent from the draw it is inserted with one pseudo-count (add-one smoothing).

The `states(battle, role)` contract takes a live `EngineBattle` and the searcher's role and
returns an `OwnStates`: tokens [K,6,9] (SLOT_ATTRS order), normalized weights [K], and the index
of the true state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OwnStates:
    tokens: np.ndarray    # [K, 6, 9] int64
    weights: np.ndarray   # [K] float64, sums to 1
    true_index: int

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or self.tokens.shape[1:] != (6, 9):
            raise ValueError(f"tokens must be [K,6,9], got {self.tokens.shape}")
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
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed * 1_000_003 + self._calls)
        with torch.no_grad():
            s = self.agent._backbone_pass(batch)
            draws = self.agent.self_belief.sample_team(
                s, batch["opp_revealed"], k=self.k * self.oversample,
                temperature=self.temperature, generator=gen,
            )[0].cpu().numpy().astype(np.int64)                       # [k*oversample, 6, 9]

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
