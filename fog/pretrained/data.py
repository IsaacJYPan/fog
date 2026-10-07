"""Split and sampling contracts for matched Gen9OU experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

from fog.team_identity import team_group_key


def composition(paste: str) -> str:
    """Conservative grouping: all sets and presentations of the same species roster."""
    return team_group_key(paste, mode="roster")


def split_teams(teams: Sequence[str], *, counts=(500, 100, 100), seed=20261006) -> dict:
    unique = {}
    for team in teams:
        unique.setdefault(composition(team), team)
    if len(unique) < sum(counts):
        raise ValueError(f"Need {sum(counts)} legal distinct compositions; found {len(unique)}")
    keys = sorted(unique)
    random.Random(seed).shuffle(keys)
    result = {}
    offset = 0
    for name, count in zip(("train", "development", "final"), counts):
        selected = keys[offset : offset + count]
        result[name] = {"teams": [unique[key] for key in selected], "compositions": selected}
        offset += count
    final = result["final"]["compositions"]
    result["final"]["confirmation_cohorts"] = [final[::2], final[1::2]]
    return result


def assert_disjoint(pools: dict) -> None:
    groups = [set(composition(team) for team in pool["teams"]) for pool in pools.values()]
    for index, group in enumerate(groups):
        if any(group.intersection(other) for other in groups[index + 1 :]):
            raise ValueError("Team compositions overlap between splits")


def match_split(match_id: str, seed: int = 20261006) -> str:
    """A match, including both perspectives, has one stable assignment."""
    value = int.from_bytes(hashlib.sha256(f"{seed}:{match_id}".encode()).digest()[:8], "big") % 100
    return "train" if value < 90 else "development"


class MatchedMixture:
    """Serializable schedule: exactly 60/40 per five optimizer updates.

    Fractions apply to sampled trajectories, as in Metamon's dataset mixtures.
    Both arms use a dedicated generator, so auxiliary route RNG cannot change data.
    Within each update every microbatch follows the precomputed effective batch.
    """

    def __init__(self, published: Sequence[str], fresh: Sequence[str], seed: int):
        if not published or not fresh:
            raise ValueError("Both published and fresh training inputs are required")
        self.published = tuple(sorted(published))
        self.fresh = tuple(sorted(fresh))
        self.rng = random.Random(seed)
        self.update = 0
        self.signature = hashlib.sha256(
            json.dumps([self.published, self.fresh]).encode()
        ).hexdigest()

    def next_batch(self, size: int = 12) -> list[tuple[str, str, int]]:
        # 7/12,7/12,7/12,7/12,8/12 = 36/60 published, exactly 60%.
        before = round(self.update * size * 0.6)
        after = round((self.update + 1) * size * 0.6)
        published_count = after - before
        sources = ["published"] * published_count + ["fresh"] * (size - published_count)
        self.rng.shuffle(sources)
        samples = [
            (
                source,
                self.rng.choice(self.published if source == "published" else self.fresh),
                self.rng.randrange(2**63),
            )
            for source in sources
        ]
        self.update += 1
        return samples

    def state_dict(self) -> dict:
        return {"update": self.update, "random": self.rng.getstate(), "signature": self.signature}

    def load_state_dict(self, state: dict) -> None:
        if state["signature"] != self.signature:
            raise ValueError("Cannot resume on a different dataset")
        self.update = state["update"]
        self.rng.setstate(state["random"])


def file_hashes(paths: Sequence[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result
