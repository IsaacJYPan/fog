"""Battle-long joint belief over K own-team and J opponent-team candidates.

Candidate identities stay fixed; weights are stored in log space to avoid
underflow. The runner can update them in three ways:

* ``observation_step`` uses the public transcript's likelihood, marginalizing
  hidden actions and chance outcomes. Impossible observations get zero weight.
* ``behavior_step`` uses the strategies that actually acted. Legal actions get
  a probability floor; unavailable actions remain impossible. Pass None for a
  side with only one legal choice, since that action carries no behavioral evidence.
* ``mechanical_step`` removes candidates contradicted by hard evidence. Callers
  must account for Illusion, Transform, and item movement before eliminating a team.

An update that would empty the table keeps the previous weights and sets
``collapsed``. The caller must handle recovery; this module does not propose
replacement candidates. In an exact pool, collapse usually indicates bad evidence
or a bug in the likelihood calculation.
"""
from __future__ import annotations

import numpy as np

from fog.search.two_sided import floored_strategy

Array = np.ndarray


class JointBeliefTable:
    """Log-space rho over (own candidates K, opponent candidates J). See module docstring."""

    def __init__(self, log_prior: Array):
        log_prior = np.asarray(log_prior, dtype=np.float64)
        if log_prior.ndim != 2:
            raise ValueError(f"log_prior must be [K, J], got shape {log_prior.shape}")
        if not (log_prior > -np.inf).any():
            raise ValueError("prior has no support")
        self.log_rho = log_prior.copy()
        self.collapsed = False           # a refused all-zero step happened; rejuvenate
        self.behavior_steps = 0
        self.mechanical_steps = 0
        self.observation_steps = 0
        self.refused_steps = 0

    @classmethod
    def from_margins(cls, log_w_own: Array, log_w_opp: Array) -> JointBeliefTable:
        """Product prior for independently selected teams.

        For correlated team selection, construct the joint log-prior directly.
        """
        lo = np.asarray(log_w_own, dtype=np.float64)
        lp = np.asarray(log_w_opp, dtype=np.float64)
        return cls(lo[:, None] + lp[None, :])

    # ------------------------------------------------------------------------------- updates
    def _apply(self, log_delta: Array) -> bool:
        candidate = self.log_rho + log_delta
        if not (candidate > -np.inf).any():
            self.collapsed = True
            self.refused_steps += 1
            return False
        self.log_rho = candidate
        return True

    def observation_step(self, likelihood: Array) -> bool:
        """Joint likelihood of the observed public outcome, including latent action choices."""
        likelihood = np.asarray(likelihood, dtype=np.float64)
        if (likelihood.shape != self.log_rho.shape or not np.isfinite(likelihood).all()
                or np.any(likelihood < 0)):
            raise ValueError("observation likelihood must be finite, nonnegative and match the axes")
        with np.errstate(divide="ignore"):
            accepted = self._apply(np.log(likelihood))
        self.observation_steps += 1
        return accepted

    def behavior_step(
        self,
        sigma1: Array | None, avail1: Array | None, observed_col1: int | None,
        sigma2: Array | None, avail2: Array | None, observed_col2: int | None,
        *,
        floor: float = 0.02,
        kept1: Array | None = None,
        kept2: Array | None = None,
    ) -> bool:
        """Multiply in both sides' floored action likelihoods for one observed public pair.
        Each side is (strategy [N, cols], availability [N, cols], observed label column), or all
        None for a side with no behavioral content this step (forced single-choice actions).
        Returns False (and leaves the table unchanged) if the step would kill every cell.

        When solving over a subset, `kept1` / `kept2` map strategy rows back to
        this table's axes. Omitted candidates receive the plain `floor` likelihood;
        their legal labels were not enumerated, so they cannot be eliminated here.
        """
        with np.errstate(divide="ignore"):        # log(0) -> -inf is the intended elimination
            delta = np.zeros_like(self.log_rho)
            if observed_col1 is not None:
                like = floored_strategy(sigma1, avail1, floor)[:, observed_col1]
                if kept1 is not None:
                    full = np.full(self.log_rho.shape[0], float(floor))
                    full[np.asarray(kept1, dtype=np.int64)] = like
                    like = full
                delta = delta + np.log(like)[:, None]
            if observed_col2 is not None:
                like = floored_strategy(sigma2, avail2, floor)[:, observed_col2]
                if kept2 is not None:
                    full = np.full(self.log_rho.shape[1], float(floor))
                    full[np.asarray(kept2, dtype=np.int64)] = like
                    like = full
                delta = delta + np.log(like)[None, :]
        ok = self._apply(delta)
        self.behavior_steps += 1
        return ok

    def mechanical_step(self, keep_own: Array | None = None,
                        keep_opp: Array | None = None) -> bool:
        """Remove candidates contradicted by reliable evidence.

        Returns False and keeps the previous weights if no candidate survives.
        """
        delta = np.zeros_like(self.log_rho)
        if keep_own is not None:
            delta = delta + np.where(np.asarray(keep_own, dtype=bool), 0.0, -np.inf)[:, None]
        if keep_opp is not None:
            delta = delta + np.where(np.asarray(keep_opp, dtype=bool), 0.0, -np.inf)[None, :]
        ok = self._apply(delta)
        self.mechanical_steps += 1
        return ok

    # ------------------------------------------------------------------------------ read-outs
    def normalized(self) -> Array:
        """The probability table rho [K, J], summing to one."""
        m = self.log_rho.max()
        w = np.exp(self.log_rho - m)
        return w / w.sum()

    def margins(self) -> tuple[Array, Array]:
        rho = self.normalized()
        return rho.sum(axis=1), rho.sum(axis=0)

    def effective_sample_sizes(self) -> tuple[float, float]:
        """Return 1 / sum(p²) for each side's marginal distribution."""
        p_own, p_opp = self.margins()
        return float(1.0 / (p_own ** 2).sum()), float(1.0 / (p_opp ** 2).sum())

    def needs_rejuvenation(self, min_ess: float = 0.0) -> bool:
        """Check collapse and, optionally, low opponent-side effective sample size.

        The default checks only collapse: concentration is expected in exact pools.
        Sampled pools can set `min_ess` to detect lost candidate diversity as well.
        """
        if self.collapsed:
            return True
        if min_ess <= 0.0:
            return False
        _, ess_opp = self.effective_sample_sizes()
        return ess_opp < min_ess
