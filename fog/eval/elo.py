"""Internal Elo among {BC, N-agents, L-agents×β, set oracle, action oracle} (spec §9).

Round-robin on the same team pool, ≥3000 games/pair, BayesElo or a simple logistic fit.

STUB: round-robin scheduling; the logistic-fit rating solver is sketched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ELO_SCALE = 400.0 / math.log(10.0)  # logistic scale so ratings read as Elo


@dataclass
class RoundRobin:
    players: list[str]
    games_per_pair: int = 3000
    results: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)  # (a,b) -> (a_wins, n)

    def schedule(self) -> list[tuple[str, str]]:
        return [(a, b) for i, a in enumerate(self.players) for b in self.players[i + 1 :]]

    def record(self, a: str, b: str, a_wins: int, n: int) -> None:
        self.results[(a, b)] = (a_wins, n)

    def run(self, play_fn) -> dict[str, float]:
        """play_fn(a, b, n) -> (a_wins, finished). Fills results then fits ratings."""
        for a, b in self.schedule():
            self.record(a, b, *play_fn(a, b, self.games_per_pair))
        return self.fit_ratings()

    def fit_ratings(self, iters: int = 500, lr: float = 0.1) -> dict[str, float]:
        """Bradley-Terry MLE by gradient ascent on the log-likelihood; ratings anchored to mean 1000
        and reported on the Elo scale. Standard, dependency-free."""
        idx = {p: i for i, p in enumerate(self.players)}
        theta = [0.0] * len(self.players)  # natural (logit) skill
        games = [(idx[a], idx[b], w, n) for (a, b), (w, n) in self.results.items() if n > 0]
        for _ in range(iters):
            grad = [0.0] * len(theta)
            for ia, ib, w, n in games:
                p = 1.0 / (1.0 + math.exp(-(theta[ia] - theta[ib])))
                grad[ia] += w - n * p
                grad[ib] -= w - n * p
            for i in range(len(theta)):
                theta[i] += lr * grad[i] / max(1, len(games))
            mean = sum(theta) / len(theta)
            theta = [t - mean for t in theta]  # anchor
        return {p: 1000.0 + ELO_SCALE * theta[idx[p]] for p in self.players}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo win probability of A over B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
