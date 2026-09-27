from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fog.scripts.import_metamon_replays import _sha256, import_shards


def _row(index, **changes):
    return {"id": f"gen9ou-{index}", "formatid": "gen9ou", "format": "[Gen 9] OU",
            "rating": "1800", "uploadtime": 100 + index, "players": ["anon-a", "anon-b"],
            "log": "|gen|9\n|win|anon-a", **changes}


def _source(tmp_path, rows, name="raw.parquet"):
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return {"name": name, "path": str(path), "sha256": _sha256(path)}


def test_import_filters_and_resumes_inside_a_batch_without_losing_games(tmp_path):
    source = _source(tmp_path, [_row(0), _row(1, rating="1000"), _row(2, formatid="gen8ou"),
                                _row(3, log="|turn|1"), _row(4), _row(5)])
    out = tmp_path / "replays"
    first = import_shards([source], out, revision="fixture-v1", max_games=1)
    assert first["progress"]["raw.parquet"] == 1
    second = import_shards([source], out, revision="fixture-v1", max_games=3)
    assert second["kept_games"] == 3
    assert second["progress"]["raw.parquet"] == 6
    assert second["filter_counts"] == {"below_rating": 1, "other_format": 1, "unfinished": 1}
    replay = json.loads((out / "gen9ou-4.json").read_text())
    assert replay["_fog_source"]["shard_sha256"] == source["sha256"]
    assert replay["_fog_source"]["row"] == 4
    assert not replay["_fog_source"]["private_truth"]
    with pytest.raises(ValueError, match="revision/filters"):
        import_shards([source], out, revision="fixture-v2")
    with pytest.raises(ValueError, match="revision/filters"):
        import_shards([source], out, revision="fixture-v1", min_rating=1600)


def test_import_rejects_changed_shards_and_conflicting_replay_ids(tmp_path):
    source = _source(tmp_path, [_row(1)])
    out = tmp_path / "replays"
    import_shards([source], out, revision="v1")
    duplicate = _source(tmp_path, [_row(1)], "same.parquet")
    result = import_shards([duplicate], out, revision="v1")
    assert result["kept_games"] == 1 and result["filter_counts"]["duplicate_id"] == 1
    conflict = _source(tmp_path, [_row(1, log="|gen|9\n|win|anon-b")], "conflict.parquet")
    with pytest.raises(ValueError, match="conflicting immutable"):
        import_shards([conflict], out, revision="v1")
    (tmp_path / "raw.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input shard changed"):
        import_shards([source], out, revision="v1")


def test_import_dates_and_invalid_id_cannot_escape_output(tmp_path):
    source = _source(tmp_path, [_row(0), _row(1, id="gen9ou-../../escape"), _row(2), _row(3)])
    out = tmp_path / "replays"
    result = import_shards([source], out, revision="v1", after=101, before=103)
    assert result["kept_games"] == 1
    assert result["filter_counts"] == {"outside_dates": 2, "invalid_id": 1}
    assert (out / "gen9ou-2.json").is_file()
