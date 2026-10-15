"""Team pools for gen9ou training: load Showdown pastes, sample per battle, validate.

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


def normalize_paste_newlines(paste: str) -> str:
    """Keep Windows/network line endings from breaking poke-env's blank-line parser."""
    return paste.replace("\r\n", "\n").replace("\r", "\n")


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
    """A poke-env Teambuilder sampling uniformly from `teams` per battle.

    Records what it yielded (`last_index`/`last_paste`, full `history`): with sequential battles
    (concurrent_battles == 1) the recorded paste IS the exact team of the battle in flight, which is
    the ground-truth source for self-play belief supervision — do NOT re-infer the team from revealed
    species (ambiguous on shared-species pools like POOL-A; that ambiguity silently disabled belief
    supervision on exactly the pool where sets are hidden).
    """
    from poke_env.teambuilder import Teambuilder

    class PoolTeambuilder(Teambuilder):
        def __init__(self) -> None:
            self._rng = random.Random(seed)
            self._pastes = list(teams)
            self._packed = [self.join_team(self.parse_showdown_team(normalize_paste_newlines(t))) for t in teams]
            self.last_index: int | None = None  # pool-id label for M1.5 belief metrics
            self.last_paste: str | None = None  # exact-truth source (see docstring)
            self.history: list[int] = []        # yield order == battle-start order (sequential play)

        def yield_team(self) -> str:
            self.last_index = self._rng.randrange(len(self._packed))
            self.last_paste = self._pastes[self.last_index]
            self.history.append(self.last_index)
            return self._packed[self.last_index]

    return PoolTeambuilder()


def make_paired_teambuilders(teams: list[str], n_games: int, seed: int | None = None):
    """Two teambuilders sharing a paired schedule: each sampled (team_i, team_j) matchup is played
    twice in consecutive battles with sides swapped. Cuts team-matchup variance in head-to-head
    evals (§7a L-vs-N) — on a 16-team pool the matchup, not the policy, dominates single-game
    outcomes. Requires sequential battles (yield order == battle order)."""
    from poke_env.teambuilder import Teambuilder

    rng = random.Random(seed)
    schedule_a: list[int] = []
    schedule_b: list[int] = []
    for _ in range((n_games + 1) // 2):
        i, j = rng.randrange(len(teams)), rng.randrange(len(teams))
        schedule_a += [i, j]
        schedule_b += [j, i]

    def _scheduled(schedule: list[int]):
        class ScheduledTeambuilder(Teambuilder):
            def __init__(self) -> None:
                self._packed = [self.join_team(self.parse_showdown_team(normalize_paste_newlines(t))) for t in teams]
                self._schedule = schedule
                self._n = 0
                self.last_index: int | None = None

            def yield_team(self) -> str:
                self.last_index = self._schedule[self._n % len(self._schedule)]
                self._n += 1
                return self._packed[self.last_index]

        return ScheduledTeambuilder()

    return _scheduled(schedule_a), _scheduled(schedule_b)


def to_packed(paste: str) -> str:
    """Showdown paste -> packed format (via poke-env's parser)."""
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self) -> str:  # pragma: no cover - never battled
            return ""

    t = _T()
    return t.join_team(t.parse_showdown_team(normalize_paste_newlines(paste)))


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
        check=False,
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
