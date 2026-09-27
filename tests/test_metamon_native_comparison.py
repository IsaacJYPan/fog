from __future__ import annotations

import json

import numpy as np
import pytest

from fog.constants.tables import TABLES
from fog.eval.metamon_native_comparison import (
    METHODS, evidence_counts, export_partial, import_priors, import_team,
    paired_comparisons, prepare, score,
)
from fog.training.replay_policy_data import complete_private_team


def _partial():
    species = ["garchomp", "corviknight", "kingambit", "greattusk", "dragapult", "clefable"]
    result = np.full((6, 15), -1, dtype=np.int64)
    result[:, 0] = [TABLES.species_index(name) for name in species]
    result[0, 1] = TABLES.item_index("<NO_ITEM>")
    result[0, 2] = TABLES.ability_index("roughskin")
    result[0, 3] = TABLES.move_index("earthquake")
    result[0, 14] = 18
    return result


def test_native_partial_round_trip_preserves_unknowns_and_stellar():
    team = _partial()
    external = export_partial(team)
    np.testing.assert_array_equal(import_team(external[::-1], team), team)
    assert external[0]["item"] == "<noitem>"
    assert external[1]["nature"] == "$missing_nature$"


def test_import_does_not_repair_wrong_native_evidence_from_reference():
    team = _partial()
    external = export_partial(team)
    external[0]["moves"] = ["Protect", "$missing_move$", "$missing_move$", "$missing_move$"]
    external[0]["item"] = "Leftovers"
    result = import_team(external, team)
    assert result[0, 1] == TABLES.item_index("leftovers")
    assert TABLES.move_index("earthquake") not in result[0, 3:7]
    counts = evidence_counts(result, team)
    assert counts["item_wrong"] == 1
    assert counts["moves_wrong"] == 1


def test_native_priors_are_injected_without_fixed_fog_spreads():
    usage = {"species": {"Garchomp": {
        "items": {"Choice Scarf": 1.0, "Other": 0.1},
        "abilities": {"Rough Skin": 1.0}, "moves": {"Earthquake": 1.0, "Protect": 1.0},
        "tera_types": {"Stellar": 1.0}, "spreads": {"Jolly:0/252/0/0/4/252": 1.0},
    }}}
    priors, spreads = import_priors(usage)
    team = _partial()
    team[0, 1:] = -1
    completed, provenance = complete_private_team(team, seed=9, usage_priors=priors,
                                                  spread_priors=spreads)
    assert completed[0, 1] == TABLES.item_index("choicescarf")
    assert completed[0, 14] == 18
    np.testing.assert_array_equal(completed[0, 8:14], [0, 63, 0, 0, 1, 63])
    assert (provenance[0, 1:] == 1).all()


def test_worker_input_contains_no_truth_or_split_metadata(tmp_path, monkeypatch):
    team = _partial()
    monkeypatch.setattr("fog.eval.metamon_native_comparison.revealed_teams",
                        lambda _: {"p1": team, "p2": team})
    corpus, query = tmp_path / "corpus.jsonl", tmp_path / "query.jsonl"
    corpus.write_text(json.dumps({"replay": {"id": "game", "log": "public"},
                                  "truth": "secret-sheet", "team_hashes": ["secret-id"],
                                  "split": "test"}))
    prepare(corpus, query)
    data = json.loads(query.read_text())
    assert set(data) == {"replay", "fog_teams"}
    assert "secret" not in query.read_text()


def test_comparison_rejects_team_leakage_before_loading_predictions(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"split": split, "team_hashes": ["same-team"]})
                                 for split in ("train", "test")))
    with pytest.raises(ValueError, match="teams overlap"):
        score(corpus, tmp_path / "absent-native", tmp_path / "absent-bank",
              tmp_path / "report.json", seed=19, policy_steps=1)


def test_paired_intervals_exclude_unpaired_and_training_games():
    games = [{"split": "test", "all_methods_complete": True,
              "scores": {method: {"counts": {"item_correct": int(method == METHODS[5]),
                                               "item_count": 1}}
                         for method in METHODS}} for _ in range(2)]
    # Excluded records deliberately have no scores: they must never enter the bootstrap.
    games += [{"split": "train", "all_methods_complete": True},
              {"split": "test", "all_methods_complete": False}]
    result = paired_comparisons(games, 19)
    assert result[METHODS[5] + "_minus_" + METHODS[0]]["item"] == {
        "delta": 1.0, "lower": 1.0, "upper": 1.0}
