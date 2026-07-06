"""Self-play league opponent sampling (spec §6). Standard defense against strategy cycling.

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
    snapshot_every_decisions: int = 200_000
    latest_ckpt: Path | None = None
    past_ckpts: list[Path] = field(default_factory=list)
    scripted: tuple[str, ...] = ("MaxDamage", "SimpleHeuristics")

    def maybe_snapshot(self, decisions_seen: int, ckpt_path: Path) -> None:
        if decisions_seen % self.snapshot_every_decisions == 0:
            self.past_ckpts.append(ckpt_path)
            self.latest_ckpt = ckpt_path

    def sample_opponent(self, rng: random.Random | None = None) -> str | Path:
        """Return an opponent spec (checkpoint path or scripted name) per the league mixture."""
        rng = rng or random
        r = rng.random()
        if r < self.scripted_frac or not self.past_ckpts:
            return rng.choice(self.scripted)
        if r < self.scripted_frac + self.past_frac:
            return rng.choice(self.past_ckpts)
        return self.latest_ckpt or rng.choice(self.scripted)
