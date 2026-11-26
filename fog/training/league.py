"""Self-play league opponent sampling. Standard defense against strategy cycling.

Opponent sampled: 50% latest checkpoint, 35% uniform past checkpoints (snapshot every 200k decisions),
15% scripted (Max-Damage, SimpleHeuristics).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class League:
    latest_frac: float = 0.50
    past_frac: float = 0.35
    scripted_frac: float = 0.15
    # Fixed reference opponents (e.g. the starting network) kept at a guaranteed share, so the
    # learner can't drift into beating only its own recent snapshots.
    anchor_frac: float = 0.0
    anchor_ckpts: list[Path] = field(default_factory=list)
    snapshot_every_decisions: int = 200_000
    latest_ckpt: Path | None = None
    past_ckpts: list[Path] = field(default_factory=list)
    scripted: tuple[str, ...] = ("MaxDamage", "SimpleHeuristics")
    warmup_opponent: str | Path | None = None
    warmup_decisions: int = 0

    def __post_init__(self) -> None:
        fractions = (self.latest_frac, self.past_frac, self.scripted_frac, self.anchor_frac)
        if any(value < 0.0 for value in fractions):
            raise ValueError("league opponent fractions must be non-negative")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(
                "league latest_frac + past_frac + scripted_frac + anchor_frac must equal 1"
            )
        if self.anchor_frac > 0 and not self.anchor_ckpts:
            raise ValueError("league.anchor_frac > 0 needs league.anchor_ckpts")
        if not self.scripted:
            raise ValueError("league.scripted must contain at least one opponent")
        if self.warmup_decisions < 0:
            raise ValueError("league.warmup_decisions must be non-negative")
        if self.warmup_decisions and self.warmup_opponent is None:
            raise ValueError(
                "league.warmup_opponent is required when warmup_decisions is positive"
            )

    def sample_opponent(
        self,
        rng: random.Random | None = None,
        *,
        decisions_seen: int = 0,
    ) -> str | Path:
        """Return an opponent spec (checkpoint path or scripted name) per the league mixture."""
        if decisions_seen < self.warmup_decisions:
            assert self.warmup_opponent is not None
            return self.warmup_opponent
        rng = rng or random
        r = rng.random()
        if r < self.anchor_frac:
            return rng.choice(self.anchor_ckpts)
        r -= self.anchor_frac
        if r < self.scripted_frac or not self.past_ckpts:
            return rng.choice(self.scripted)
        if r < self.scripted_frac + self.past_frac:
            return rng.choice(self.past_ckpts)
        return self.latest_ckpt or rng.choice(self.scripted)
