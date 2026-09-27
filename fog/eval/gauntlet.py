"""Gauntlet vs fixed baselines.

Random, MaxDamage, SimpleHeuristics (poke-env built-ins), BC-only checkpoint — each ≥2000 games.
Report Wilson 95% CIs. Never conclude from <1000 games. Report search on/off as a separate column
(§8).
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
        tag = f"{self.opponent} [search]" if self.search else self.opponent
        return (f"vs {tag:<25} {self.wins}/{self.games} = {iv.point:.3f} "
                f"[{iv.low:.3f}, {iv.high:.3f}]{warn}")


def run_gauntlet(
    ckpt, opponents=BASELINES, games_per: int = 2000, fmt: str = "gen9ou",
    team_pool=None, cfg=None, bc_ckpt=None, include_reading_opponent: bool = False,
    search=None,
) -> list[GauntletResult]:
    """Run `games_per` games of `ckpt` vs each baseline (+ optional BC checkpoint). Wilson CIs (§9).

    `ckpt`/`bc_ckpt` are checkpoint paths; `team_pool` a list of pastes (required for gen9ou).
    `include_reading_opponent` adds the same checkpoint as a set-oracle opponent. What that entry
    measures is the VALUE OF THE CONCEALED INFORMATION (an upper bound on any reader's gain): the
    oracle is handed the truth, so concealment cannot deny it and this number is insensitive to
    concealment by construction. The payoff of concealment is measured against a LEARNED reader
    (§7a L-vs-N), not here.

    `search` (dict {"k","prior_weight","seed"}) reports each opponent TWICE — search off and search
    on — as separate rows (§8 on/off column). Search reconstructs engine positions from the live
    Showdown battle; treat its numbers as EXPERIMENTAL until the reconstruction fuzz report
    (scripts/fuzz_recon.py) is clean on this format's pool.
    """
    from fog.eval.harness import run_match

    specs = list(opponents) + ([str(bc_ckpt)] if bc_ckpt else [])
    if include_reading_opponent:
        specs.append(f"oracle-set:{ckpt}")
    modes = [None] if not search else [None, dict(search)]
    results = []
    for opp in specs:
        for mode in modes:
            wins, finished = run_match(
                ckpt, opp, games_per, fmt, team_pool=team_pool, cfg=cfg, search=mode
            )
            if opp == str(bc_ckpt):
                name = "BC"
            elif str(opp).startswith("oracle-set:"):
                name = "SetOracle(self)"
            else:
                name = opp
            results.append(GauntletResult(name, wins, finished, search=mode is not None))
            print("  " + results[-1].line(), flush=True)
    return results
