"""Gauntlet vs fixed baselines (spec §9).

Random, MaxDamage, SimpleHeuristics (poke-env built-ins), BC-only checkpoint — each ≥2000 games.
Report Wilson 95% CIs. Never conclude from <1000 games. Report search on/off as a separate column
(§8).

STUB: match-running loop; the stats reporting (`wilson_interval`) is real.
"""
from __future__ import annotations

from dataclasses import dataclass

from fog.eval.stats import Interval, wilson_interval

BASELINES = ("Random", "MaxDamage", "SimpleHeuristics")


@dataclass
class GauntletResult:
    opponent: str
    wins: int
    games: int
    search: bool = False

    @property
    def winrate(self) -> Interval:
        return wilson_interval(self.wins, self.games)

    def line(self) -> str:
        iv = self.winrate
        warn = "" if self.games >= 1000 else "  [insufficient n: <1000, do not conclude]"
        return (f"vs {self.opponent:<18} {self.wins}/{self.games} = {iv.point:.3f} "
                f"[{iv.low:.3f}, {iv.high:.3f}]{warn}")


def run_gauntlet(
    ckpt, opponents=BASELINES, games_per: int = 2000, fmt: str = "gen9ou",
    team_pool=None, cfg=None, bc_ckpt=None, include_reading_opponent: bool = False,
) -> list[GauntletResult]:
    """Run `games_per` games of `ckpt` vs each baseline (+ optional BC checkpoint). Wilson CIs (§9).

    `ckpt`/`bc_ckpt` are checkpoint paths; `team_pool` a list of pastes (required for gen9ou).
    `include_reading_opponent` adds the same checkpoint as a set-oracle opponent — a gauntlet entry
    that actually USES reads, so concealment (§7) has a measurable payoff in win-rate.
    """
    from fog.eval.harness import run_match

    specs = list(opponents) + ([str(bc_ckpt)] if bc_ckpt else [])
    if include_reading_opponent:
        specs.append(f"oracle-set:{ckpt}")
    results = []
    for opp in specs:
        wins, finished = run_match(ckpt, opp, games_per, fmt, team_pool=team_pool, cfg=cfg)
        if opp == str(bc_ckpt):
            name = "BC"
        elif str(opp).startswith("oracle-set:"):
            name = "SetOracle(self)"
        else:
            name = opp
        results.append(GauntletResult(name, wins, finished))
        print("  " + results[-1].line(), flush=True)
    return results
