from __future__ import annotations

import sys

import polars as pl
import pytest

from fog.scripts import split_replay_manifest as split


def _manifest(n: int = 60) -> pl.DataFrame:
    # Several games share an upload time so the id tie-break is exercised.
    return pl.DataFrame({
        "id": [f"game-{index:03d}" for index in range(n)],
        "n_rows": [40 + index % 7 for index in range(n)],
        "shard": [f"shards/game-{index:03d}.npz" for index in range(n)],
        "uploadtime": [1_700_000_000 + (index * 37 % n) // 3 for index in range(n)],
    })


def _run(tmp_path, *extra):
    source = tmp_path / "manifest.parquet"
    _manifest().write_parquet(source)
    out = tmp_path / "split"
    argv = ["split", "--manifest", str(source), "--out", str(out), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        split.main()
    finally:
        sys.argv = old
    return out


def test_chronological_holdout_is_newest_disjoint_and_nested(tmp_path):
    out = _run(
        tmp_path,
        "--chronological-holdout-games", "12",
        "--train-sizes", "5", "20", "48",
        "--seed", "7",
    )
    holdout = pl.read_parquet(out / "holdout_manifest.parquet")
    training = pl.read_parquet(out / "train_manifest.parquet")
    assert holdout.columns == _manifest().columns
    assert training.columns == _manifest().columns
    assert len(holdout) == 12 and len(training) == 48
    assert set(holdout["id"]).isdisjoint(training["id"])
    assert set(holdout["id"]) | set(training["id"]) == set(_manifest()["id"])
    # Newest training game is not newer than the oldest holdout game, ties by id.
    newest_train = max(zip(training["uploadtime"], training["id"], strict=True))
    oldest_holdout = min(zip(holdout["uploadtime"], holdout["id"], strict=True))
    assert newest_train < oldest_holdout
    assert training["uploadtime"].max() <= holdout["uploadtime"].min()

    previous: set[str] = set()
    for size in (5, 20, 48):
        subset = pl.read_parquet(out / f"train_{size}.parquet")
        ids = set(subset["id"])
        assert len(subset) == size == len(ids)
        assert subset.columns == _manifest().columns
        assert ids <= set(training["id"])
        assert ids.isdisjoint(holdout["id"])
        assert previous <= ids
        previous = ids


def test_nested_sets_depend_only_on_seed(tmp_path):
    training, _ = split._chronological_split(_manifest(), 10)
    first = split._nested_training_sets(training, [10, 30], seed=3)
    again = split._nested_training_sets(training.reverse(), [30, 10], seed=3)
    other = split._nested_training_sets(training, [10, 30], seed=4)
    assert first[10]["id"].to_list() == again[10]["id"].to_list()
    assert first[30]["id"].to_list() != other[30]["id"].to_list()


def test_refuses_to_overwrite_and_rejects_oversized_requests(tmp_path):
    out = _run(tmp_path, "--chronological-holdout-games", "10", "--train-sizes", "5")
    with pytest.raises(FileExistsError):
        _run(tmp_path, "--chronological-holdout-games", "10", "--train-sizes", "5")
    assert (out / "train_5.parquet").exists()
    training, _ = split._chronological_split(_manifest(), 10)
    with pytest.raises(ValueError):
        split._nested_training_sets(training, [51], seed=0)
    with pytest.raises(ValueError):
        split._chronological_split(_manifest(), 60)


def test_random_split_still_works(tmp_path):
    out = _run(tmp_path, "--holdout-fraction", "0.25", "--train-sizes", "10")
    holdout = pl.read_parquet(out / "holdout_manifest.parquet")
    training = pl.read_parquet(out / "train_manifest.parquet")
    assert len(holdout) == 15 and len(training) == 45
    assert set(holdout["id"]).isdisjoint(training["id"])
    assert set(pl.read_parquet(out / "train_10.parquet")["id"]) <= set(training["id"])
