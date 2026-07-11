"""Statistics helpers for evaluation (spec §9). Pokémon is high-variance — always report CIs.

Wilson score interval for a binomial proportion (win-rate). Never conclude from <1000 games.
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
