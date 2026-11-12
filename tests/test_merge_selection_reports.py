from __future__ import annotations

import pytest

from fog.eval.merge_selection_reports import merge_reports
from fog.eval.showdown_direct_selection import MatchResult


def _report(seed: int, wins: int, losses: int, **protocol):
    result = MatchResult(label="m vs s", a="m", b="s", requested_games=wins + losses,
                         completed_games=wins + losses, a_wins=wins, b_wins=losses,
                         game_scores=[1.0] * wins + [0.0] * losses, turns=30 * (wins + losses))
    result.game_counters = [{"a": {"decisions": 20, "fallbacks": 0, "decision_seconds": 4.0}}
                            for _ in range(wins + losses)]
    return {"status": "complete", "checkpoints": {},
            "protocol": {"pool": "p", "games_per_row": wins + losses, "seed": seed, **protocol},
            "rows": [result.as_dict()]}


def test_merge_pools_games_scores_and_counters():
    merged = merge_reports([_report(1, 6, 2), _report(2, 7, 1)])
    row = merged["rows"][0]
    assert (row["completed_games"], row["a_wins"], row["b_wins"]) == (16, 13, 3)
    assert row["a_win_rate_decisive"] == pytest.approx(13 / 16)
    assert len(row["game_scores"]) == 16 and len(row["game_counters"]) == 16
    assert row["a_fidelity"]["decisions"] == 320
    assert merged["protocol"]["seeds"] == [1, 2] and merged["protocol"]["games_per_row"] == 16


def test_merge_refuses_shards_that_are_not_comparable():
    with pytest.raises(ValueError, match="share a seed"):
        merge_reports([_report(1, 6, 2), _report(1, 7, 1)])
    with pytest.raises(ValueError, match="more than their seed"):
        merge_reports([_report(1, 6, 2), _report(2, 7, 1, pool="q")])
    with pytest.raises(ValueError, match="sequential"):
        merge_reports([_report(1, 6, 2, sequential={"p0": 0.5}),
                       _report(2, 7, 1, sequential={"p0": 0.5})])
    unfinished = _report(3, 6, 2)
    unfinished["status"] = "in_progress"
    with pytest.raises(ValueError, match="complete"):
        merge_reports([_report(1, 6, 2), unfinished])
