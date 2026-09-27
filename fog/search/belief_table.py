"""Persistent joint public belief across turns (ARCHITECTURE.md invariants S1–S3).

One `JointBeliefTable` lives for one battle. Its axes are STABLE candidate identities — K
candidate versions of our team and J candidate opponent teams, fixed at construction so that
sequential Bayes updates are well defined (persistent particles, §7.4). The cell (k, j) holds the
public probability that our private team is candidate k AND the opponent's is candidate j, given
every public observation so far. Weights are kept in log space: a battle multiplies in dozens of
likelihoods, and direct products underflow.

Update contract (who supplies what):

  * `observation_step(...)` — the persistent search runner supplies the joint likelihood of the
    public transcript, marginalized over hidden choices and chance outcomes. This is the primary
    update; mechanically incompatible worlds have zero likelihood. No arbitrary floor is applied.
  * `behavior_step(...)` — after a real public action pair, multiply in each side's modeled
    probability of the action it was OBSERVED to play. The caller must pass the distribution that
    ACTUALLY acted (the solver's averaged strategy during solver play, the raw policy during
    policy-only play — invariant 19.6), and the likelihoods are floored here via
    `two_sided.floored_strategy`, so a modeled "never" shrinks a candidate without destroying it
    (§7.2 safety rule 1). A candidate whose label space LACKS the observed action still gets an
    exact zero — that elimination is mechanical (the world cannot produce the observation), not
    behavioral. A forced action with a single legal choice carries no behavioral likelihood:
    pass that side's column as None (§14).
  * `mechanical_step(keep_own, keep_opp)` — hard evidence: boolean masks of candidates that
    remain possible. The CALLER owns the truthful-evidence carve-outs of §7.2 (Illusion,
    Transform, item movement): only pass keep=False for facts the rules guarantee are truthful
    about that candidate's own team. The current controlled pools contain none of the deception
    mechanics, so reveal-census contradictions are safe hard zeros there; open-format callers
    must filter first.

Never-empty guard (invariant 19.18): any step that would zero the whole table is REFUSED — the
table keeps its previous state, records `collapsed = True`, and `needs_rejuvenation()` turns
true. Rejuvenation itself (proposing fresh evidence-consistent particles and re-scoring them) is
a later phase; exact-pool play cannot collapse unless something upstream is wrong, which is
exactly why the refusal is loud in the stats rather than silent.
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
        """Independent prior rho_0(k, j) = P_own(k) * P_opp(j) (proposal §7: valid while the two
        team selections are sampled independently; a correlated matchmaking prior would construct
        the joint table directly)."""
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

        `kept1` / `kept2` (Phase G2 posterior-mass truncation): when a side's solve ran over a
        SUBSET of this table's candidates, pass the kept candidates' full-axis indices; that
        side's sigma/avail rows then correspond to the kept subset in order. Candidates OUTSIDE
        the subset were not modeled this turn, and unmodeled is not impossible (§7.2): they take
        the plain `floor` likelihood — shrunk relative to modeled candidates that liked the
        action, never zeroed, and label-based elimination applies only to modeled candidates
        (we never enumerated a truncated-out candidate's labels)."""
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
        """Zero candidates contradicted by hard, truthful evidence (see module docstring for the
        caller's carve-out duty). Returns False and refuses the step if it would empty the
        table — which on an exact pool means an upstream bug, and in an open space means
        rejuvenation is due."""
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
        """Per margin: 1 / sum(p^2) — the standard 'how many candidates still really count'
        number (equals the count when uniform, 1.0 when a point mass)."""
        p_own, p_opp = self.margins()
        return float(1.0 / (p_own ** 2).sum()), float(1.0 / (p_opp ** 2).sum())

    def needs_rejuvenation(self, min_ess: float = 0.0) -> bool:
        """True when the filter is dead (a refused all-zero step), or — if the caller opts into a
        positive `min_ess` — when the opponent-side effective sample size fell below it. The
        threshold is an OPEN-SPACE policy (§7.4 step 4: sampled particles losing diversity around
        a truth that may not be in the set); on an exact pool a concentrated posterior is the
        correct end state, not a failure, so the default checks collapse only."""
        if self.collapsed:
            return True
        if min_ess <= 0.0:
            return False
        _, ess_opp = self.effective_sample_sizes()
        return ess_opp < min_ess
