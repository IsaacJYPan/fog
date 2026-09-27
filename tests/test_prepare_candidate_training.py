import json

import numpy as np
import pytest

from fog.scripts.prepare_candidate_training import finalize, prepare


def test_split_precedes_imputation_and_excludes_all_future_donors(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(12):
        # Last two games are donor-quality but below the policy-training threshold.
        replay = {"id": f"game-{i:02}", "formatid": "gen9ou", "uploadtime": 100 + i,
                  "rating": 1700 if i < 10 else 1500, "log": f"public log {i}"}
        (raw / f"game-{i:02}.json").write_text(json.dumps(replay))
    duplicate = json.loads((raw / "game-09.json").read_text())
    duplicate["id"] = "other-id"
    (raw / "other-id.json").write_text(json.dumps(duplicate))
    out = tmp_path / "prepared"
    result = prepare(raw, out, games=4)
    assert result["eligible_unique_games"] == 10
    assert [g["id"] for g in result["games"]] == [f"game-{i:02}" for i in range(6, 10)]
    assert json.loads((out / "holdout_ids.json").read_text()) == ["game-09"]
    assert set(json.loads((out / "donor_excluded_ids.json").read_text())) == {
        "game-09", "game-10", "game-11", "other-id"}
    assert len(list((out / "raw").glob("*.json"))) == 4
    with pytest.raises(FileExistsError):
        prepare(raw, out, games=4)


@pytest.mark.parametrize("id_overlap", [True, False])
def test_finalize_rejects_heldout_ids_or_duplicate_logs(tmp_path, id_overlap):
    selection = {"holdout_cutoff": 100, "games": [
        {"id": "held", "log_sha256": "held-hash", "split": "holdout"}]}
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    np.savez(tmp_path / "bank.npz", replay_ids=["held" if id_overlap else "duplicate-id"],
             log_hashes=["different" if id_overlap else "held-hash"], timestamps=[99])
    with pytest.raises(ValueError, match="held-out evidence"):
        finalize(tmp_path)
