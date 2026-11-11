"""Statistics helpers for evaluation. Pokémon is high-variance — always report CIs.

Wilson intervals describe independent binary outcomes. Paired schedules need block-level
uncertainty; neither a fixed game count nor a point estimate establishes improvement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# z for common two-sided confidence levels
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a win-rate. `wins` of `n` games."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    z = _Z.get(confidence, 1.959963984540054)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, center - half, center + half)


def paired_block_interval(scores, *, alpha: float = 0.05, block_size: int = 4,
                          bounds: tuple[float, float] = (0.0, 1.0),
                          seed: int = 0, resamples: int = 20_000) -> Interval:
    """Percentile bootstrap over independent matched blocks, not correlated games.

    This is an approximate interval, not an anytime confidence sequence. Callers allocate alpha
    over scheduled comparisons. Constant samples use a bounded unseen-mass interval instead of
    a spuriously zero-width bootstrap interval.
    """
    import numpy as np

    values = np.asarray(scores, dtype=np.float64)
    if not (0 < alpha < 1) or block_size < 1 or len(values) < 2 * block_size:
        raise ValueError("interval needs valid alpha and at least two complete blocks")
    if (values.ndim != 1 or len(values) % block_size or not np.isfinite(values).all()
            or np.any(values < bounds[0]) or np.any(values > bounds[1])):
        raise ValueError("scores must be finite bounded values in complete matched blocks")
    means = values.reshape(-1, block_size).mean(axis=1)
    point = float(means.mean())
    if np.ptp(means) == 0:
        unseen = 1.0 - (alpha / 2.0) ** (1.0 / len(means))
        return Interval(point, point - (point - bounds[0]) * unseen,
                        point + (bounds[1] - point) * unseen)
    if resamples < 2 / alpha:
        raise ValueError("too few bootstrap resamples for the requested tail probability")
    rng = np.random.default_rng(seed)
    boot = np.concatenate([
        means[rng.integers(0, len(means), size=(min(512, resamples - i), len(means)))].mean(1)
        for i in range(0, resamples, 512)
    ])
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(low), float(high))


_VAR_FLOOR = 1e-6


@dataclass
class SequentialDecision:
    decision: str  # "accept_h1" | "accept_h0" | "continue" | "max_reached"
    blocks: int
    mean: float
    llr: float
    lower: float
    upper: float


class BlockSPRT:
    """Generalized SPRT on matched-block scores in [0, 1] (H0: mean=p0 vs H1: mean=p1).

    Uses a normal approximation, LLR = n (p1-p0)(2m-p0-p1) / (2 s^2), so error rates are
    approximate. "max_reached" means no decision was reached, not that the players are equal.
    """

    def __init__(self, p0: float = 0.50, p1: float = 0.55, alpha: float = 0.05,
                 beta: float = 0.05, min_blocks: int = 100, max_blocks: int = 1000):
        if not (0.0 < p0 < p1 < 1.0):
            raise ValueError("require 0 < p0 < p1 < 1")
        if not (0.0 < alpha < 0.5 and 0.0 < beta < 0.5):
            raise ValueError("alpha and beta must be in (0, 0.5)")
        if min_blocks < 2 or max_blocks < min_blocks:
            raise ValueError("require 2 <= min_blocks <= max_blocks")
        self.p0, self.p1 = p0, p1
        self.min_blocks, self.max_blocks = min_blocks, max_blocks
        self.lower = math.log(beta / (1.0 - alpha))
        self.upper = math.log((1.0 - beta) / alpha)
        self.n = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.finished: SequentialDecision | None = None

    def _llr(self, n: int, mean: float, var: float) -> float:
        var = max(var, _VAR_FLOOR)
        return n * (self.p1 - self.p0) * (2.0 * mean - self.p0 - self.p1) / (2.0 * var)

    def _decide(self, n: int, mean: float, llr: float) -> str:
        if n >= self.min_blocks:
            if llr >= self.upper:
                return "accept_h1"
            if llr <= self.lower:
                return "accept_h0"
            if n >= self.max_blocks:
                return "max_reached"
        return "continue"

    def update(self, block_score: float) -> SequentialDecision:
        if self.finished is not None:
            return self.finished
        s = float(block_score)
        if not (0.0 <= s <= 1.0):
            raise ValueError("block score must be in [0, 1]")
        self.n += 1
        self._sum += s
        self._sumsq += s * s
        n = self.n
        mean = self._sum / n
        var = (self._sumsq - n * mean * mean) / (n - 1) if n > 1 else 0.0
        llr = self._llr(n, mean, var)
        d = SequentialDecision(self._decide(n, mean, llr), n, mean, llr, self.lower, self.upper)
        if d.decision != "continue":
            self.finished = d
        return d

    @classmethod
    def evaluate(cls, scores, **kwargs) -> SequentialDecision:
        """Run the test over a sequence of block scores, stopping at the first decision."""
        test = cls(**kwargs)
        d = SequentialDecision("continue", 0, float("nan"), 0.0, test.lower, test.upper)
        for s in scores:
            d = test.update(s)
            if d.decision != "continue":
                break
        return d
