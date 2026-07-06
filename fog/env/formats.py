"""POC formats and team-pool helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

FORMAT_3V3_POC = "gen9ou3v3poc"
FORMAT_OU = "gen9ou"


def load_team_pool(path: str | Path) -> list[str]:
    """Load a parsed team pool (list of Showdown paste strings)."""
    with open(path) as f:
        data = json.load(f)
    teams = data["teams"] if isinstance(data, dict) else data
    if not teams:
        raise ValueError(f"Empty team pool at {path}")
    return teams


def _randrange(rng: Any, n: int) -> int:
    if rng is None:
        return random.randrange(n)
    if hasattr(rng, "randrange"):
        return int(rng.randrange(n))
    if hasattr(rng, "integers"):
        return int(rng.integers(n))
    if hasattr(rng, "randint"):
        return int(rng.randint(n))
    raise TypeError("rng must provide randrange(), integers(), or randint().")


def hidden_pool_variant(pool: Sequence[str], *, rng: Any = None) -> tuple[str, int]:
    """Sample one team paste and carry its pool-id label for M1.5 belief metrics."""
    if not pool:
        raise ValueError("hidden_pool_variant requires a nonempty team pool")
    pool_id = _randrange(rng, len(pool))
    return pool[pool_id], pool_id
