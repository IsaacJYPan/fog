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


def test_run_shards_splits_whole_blocks_and_resumes(tmp_path):
    import json

    from fog.eval.run_shards import shard_commands, split_games

    assert split_games(1000, 3) == [336, 332, 332]
    assert split_games(8, 2) == [4, 4]
    with pytest.raises(ValueError):
        split_games(10, 2)
    with pytest.raises(ValueError):
        split_games(8, 3)

    (tmp_path / "shard_0.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "shard_1.json").write_text(json.dumps({"status": "in_progress"}))
    commands = shard_commands(["--pool", "p", "--opponent", "random"], tmp_path, 24, 3, 50)
    assert [report.name for report, _ in commands] == ["shard_1.json", "shard_2.json"]
    resumed, fresh = commands[0][1], commands[1][1]
    assert resumed[-1] == "--resume" and "--resume" not in fresh
    assert fresh[fresh.index("--seed") + 1] == "52" and fresh[fresh.index("--games") + 1] == "8"
    with pytest.raises(ValueError, match="--seed"):
        shard_commands(["--seed", "3"], tmp_path, 24, 3, 50)
