from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fog.constants.tables import TABLES
from fog.eval.replay_imputation import align_truth, score_samples
from fog.training.replay_candidates import (
    ReplayCandidateBank, build_bank, canonical_row, revealed_teams, vocabulary_digest,
)
from fog.training.replay_policy_data import EXACT, INFERRED, complete_private_team


def _row(item="leftovers", moves=("earthquake", "stealthrock", "protect", "toxic")):
    row = np.full(15, -1, dtype=np.int64)
    row[0] = TABLES.species_index("garchomp")
    row[1] = TABLES.item_index(item)
    row[2] = TABLES.ability_index("roughskin")
    row[3:3 + len(moves)] = [TABLES.move_index(move) for move in moves]
    return canonical_row(row)


def _bank(tmp_path: Path, rows, dates=None, ids=None, hashes=None):
    path = tmp_path / "bank.npz"
    n = len(rows)
    np.savez_compressed(
        path, tokens=np.asarray(rows), rosters=np.tile([rows[0][0], -1, -1, -1, -1, -1], (n, 1)),
        timestamps=np.asarray(dates or [100] * n), replay_ids=np.asarray(ids or ["donor"] * n),
        log_hashes=np.asarray(hashes or ["hash"] * n), ratings=np.full(n, 1700),
        metadata=json.dumps({"schema_version": 1, "format": "gen9ou",
                             "vocabulary_sha256": vocabulary_digest()}),
    )
    return ReplayCandidateBank(path)


def _team(row):
    team = np.full((6, 15), -1, dtype=np.int64)
    team[0] = row
    return team


def test_joint_donor_preserves_correlated_set_and_inferred_provenance(tmp_path):
    a = _row()
    b = _row("choicescarf", ("earthquake", "outrage", "stoneedge", "firefang"))
    bank = _bank(tmp_path, [a, b])
    revealed = _team(np.full(15, -1))
    revealed[0, 0] = a[0]
    for seed in range(12):
        completed, provenance = complete_private_team(revealed, seed=seed, candidate_bank=bank,
                                                       timestamp=200)
        assert any(np.array_equal(completed[0, :7], donor[:7]) for donor in (a, b))
        assert provenance[0, 0] == EXACT
        assert (provenance[0, 1:] == INFERRED).all()
        assert (completed[1:] == -1).all()
        baseline, _ = complete_private_team(revealed, seed=seed)
        np.testing.assert_array_equal(completed[:, 7:14], baseline[:, 7:14])


def test_donor_cannot_contradict_scalar_or_moves(tmp_path):
    a = _row()
    b = _row("choicescarf", ("earthquake", "outrage", "stoneedge", "firefang"))
    bank = _bank(tmp_path, [a, b])
    revealed = _team(np.full(15, -1))
    revealed[0, 0] = a[0]
    # A revealed move in a nonpacked position is still preserved and not duplicated.
    revealed[0, 5] = TABLES.move_index("protect")
    out, _ = complete_private_team(revealed, seed=3, candidate_bank=bank, timestamp=200)
    assert out[0, 1] == a[1]
    assert out[0, 5] == revealed[0, 5]
    assert set(out[0, 3:7]) == set(a[3:7])
    assert len(set(out[0, 3:7])) == 4
    revealed[0, 1] = b[1]
    _, report = bank.propose(revealed, rng=np.random.default_rng(0), timestamp=200)
    assert report["no_compatible_donor"] == 1


@pytest.mark.parametrize("context", [
    {"timestamp": 100}, {"timestamp": 99}, {"timestamp": 0},
    {"timestamp": 200, "replay_id": "donor"},
    {"timestamp": 200, "log_hash": "hash"},
])
def test_no_target_duplicate_or_future_evidence(tmp_path, context):
    a = _row()
    bank = _bank(tmp_path, [a])
    revealed = _team(np.full(15, -1))
    revealed[0, 0] = a[0]
    result, report = bank.propose(revealed, rng=np.random.default_rng(0), **context)
    assert np.array_equal(result, revealed)
    assert report.get("donor_slots", 0) == 0


def test_bank_rejects_changed_vocabulary(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, metadata=json.dumps({"schema_version": 1, "format": "gen9ou",
                                       "vocabulary_sha256": "wrong"}))
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        ReplayCandidateBank(path)


def test_truth_alignment_and_metrics_ignore_move_order_but_not_hidden_errors():
    truth = _team(_row())
    revealed = truth.copy()
    revealed[0, 1] = -1
    revealed[0, 3:7] = -1
    revealed[0, 3] = truth[0, 6]
    # Alignment expects six actual species for a corpus, so exercise metrics directly here.
    sample = truth.copy()
    sample[0, 3:7] = sample[0, 3:7][::-1]
    counts = score_samples([sample], truth, revealed)
    assert counts["hidden_move_correct"] == counts["hidden_move_count"] == 3
    assert counts["item_correct"] == counts["item_count"] == 1
    assert counts["joint_public_correct"] == 1
    sample[0, 1] = TABLES.item_index("choicescarf")
    counts = score_samples([sample], truth, revealed)
    assert counts["joint_public_correct"] == 0


def test_truth_alignment_rejects_wrong_public_facts():
    truth = np.tile(_row(), (6, 1))
    truth[:, 0] = np.arange(6)
    revealed = truth.copy()
    revealed[:, 3:7] = revealed[:, 3:7][:, ::-1]
    aligned = align_truth(truth, revealed)
    np.testing.assert_array_equal(aligned, revealed)
    revealed[0, 1] = TABLES.item_index("choicescarf")
    with pytest.raises(ValueError, match="scalar evidence"):
        align_truth(truth, revealed)


def _public_log():
    roster = ("Garchomp", "Corviknight", "Kingambit", "Great Tusk", "Dragapult", "Clefable")
    return "\n".join([
        "|gen|9", "|gametype|singles", "|tier|[Gen 9] OU",
        "|player|p1|A|1|1700", "|player|p2|B|2|1700", "|clearpoke",
        *[f"|poke|{role}|{species}|" for role in ("p1", "p2") for species in roster],
        "|teampreview", "|start", "|switch|p1a: Chomp|Garchomp|100/100",
        "|switch|p2a: Bird|Corviknight|100/100", "|turn|1",
        "|move|p1a: Chomp|Earthquake|p2a: Bird", "|-immune|p2a: Bird",
        "|move|p2a: Bird|Roost|p2a: Bird", "|turn|2",
        "|move|p1a: Chomp|Protect|p1a: Chomp", "|move|p2a: Bird|Brave Bird|p1a: Chomp",
        "|-enditem|p1a: Chomp|Leftovers|[from] move: Knock Off", "|win|A",
    ])


def test_fast_extractor_agrees_with_policy_parser_and_keeps_original_item():
    from fog.scripts.parse_replays import parse_replay

    replay = {"log": _public_log(), "players": ["A", "B"]}
    extracted = revealed_teams(replay)
    rows = parse_replay(replay)["rows"]
    for role in ("p1", "p2"):
        expected = next(r["self_belief"] for r in rows if r["role"] == role)
        np.testing.assert_array_equal(extracted[role], expected)
    assert extracted["p1"][0, 1] == TABLES.item_index("leftovers")
    assert (extracted["p1"][:, 7:14] == -1).all()


def test_build_deduplicates_logs_and_excludes_alias_of_holdout(tmp_path):
    paths = []
    for filename, rid, log in (("a", "heldout", _public_log()),
                               ("b", "alias", _public_log()),
                               ("c", "train", _public_log() + "\n|message|different game")):
        path = tmp_path / (filename + ".json")
        path.write_text(json.dumps({"id": rid, "log": log, "uploadtime": 100,
                                    "rating": 1700, "formatid": "gen9ou"}))
        paths.append(path)
    report = build_bank(paths, tmp_path / "all.npz")
    assert report["counts"]["duplicate_log"] == 1
    report = build_bank(paths, tmp_path / "train.npz", excluded_ids={"heldout"})
    assert report["counts"]["excluded"] == 2
    bank = ReplayCandidateBank(tmp_path / "train.npz")
    assert set(bank.replay_ids) == {"train"}


def test_stellar_donor_is_installed_without_quarantining_replay():
    from poke_env.battle import Pokemon, PokemonType

    from fog.training.replay_policy_data import apply_private_set

    row = _row()
    row[14] = 18
    mon = Pokemon(gen=9, species="garchomp")
    apply_private_set(mon, row)
    assert mon.tera_type == PokemonType.STELLAR
