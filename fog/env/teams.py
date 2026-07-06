"""Team pools for gen9ou training (spec §4.4): load Showdown pastes, sample per battle, validate.

poke-env's `Player(team=...)` accepts a `Teambuilder`; its `yield_team()` is called once per battle,
which is exactly the spec's "teams sampled per-game from the active pool" (§6). `PoolTeambuilder`
wraps a directory of paste files.

Validation goes through the PINNED simulator's own validator (`node pokemon-showdown validate-team`)
— the server is the source of truth for legality, so a pool that passes here cannot silently hang
battles on team-rejection popups.
"""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

SHOWDOWN_DIR = Path(__file__).resolve().parents[2] / "server" / "pokemon-showdown"


def load_pool(path: str | Path) -> list[str]:
    """Load a team pool: a directory of .txt Showdown pastes (one team per file, sorted order),
    a packaged .json ({"teams": [pastes]}), or a single paste file."""
    import json

    p = Path(path)
    if p.is_dir():
        teams = [f.read_text(encoding="utf-8") for f in sorted(p.glob("*.txt"))]
    elif p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        teams = data["teams"] if isinstance(data, dict) else data
    else:
        teams = [p.read_text(encoding="utf-8")]
    teams = [t for t in teams if t.strip()]
    if not teams:
        raise ValueError(f"no teams found at {path}")
    return teams


def make_pool_teambuilder(teams: list[str], seed: int | None = None):
    """A poke-env Teambuilder sampling uniformly from `teams` per battle."""
    from poke_env.teambuilder import Teambuilder

    class PoolTeambuilder(Teambuilder):
        def __init__(self) -> None:
            self._rng = random.Random(seed)
            self._packed = [self.join_team(self.parse_showdown_team(t)) for t in teams]
            self.last_index: int | None = None  # pool-id label for M1.5 belief metrics

        def yield_team(self) -> str:
            self.last_index = self._rng.randrange(len(self._packed))
            return self._packed[self.last_index]

    return PoolTeambuilder()


def to_packed(paste: str) -> str:
    """Showdown paste -> packed format (via poke-env's parser)."""
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self) -> str:  # pragma: no cover - never battled
            return ""

    t = _T()
    return t.join_team(t.parse_showdown_team(paste))


def validate_team(paste: str, battle_format: str = "gen9ou") -> tuple[bool, str]:
    """Validate one paste against the pinned simulator. Returns (ok, validator_output)."""
    packed = to_packed(paste)
    proc = subprocess.run(
        ["node", "pokemon-showdown", "validate-team", battle_format],
        input=packed,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(SHOWDOWN_DIR),
    )
    ok = proc.returncode == 0
    return ok, (proc.stdout + proc.stderr).strip()


def validate_pool(path: str | Path, battle_format: str = "gen9ou") -> list[str]:
    """Validate every team in a pool; returns a list of error strings (empty = all legal)."""
    errors = []
    p = Path(path)
    files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    for f in files:
        ok, out = validate_team(f.read_text(encoding="utf-8"), battle_format)
        if not ok:
            errors.append(f"{f.name}: {out}")
    return errors
