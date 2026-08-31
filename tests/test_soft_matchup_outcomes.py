import numpy as np
import polars as pl

from fog.scripts.materialize_soft_matchup_outcomes import main


def test_soft_matchup_materialization_preserves_inputs(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index, outcome in enumerate((-1.0, 1.0)):
        shard = source / f"{index}.npz"
        np.savez_compressed(shard, outcome=np.array([outcome, outcome]), x=np.array([index]))
        rows.append({"shard": str(shard), "p1_team": 1, "p2_team": 2,
                     "outcome": outcome, "split": "train", "n_rows": 2})
    manifest = source / "manifest.parquet"
    pl.DataFrame(rows).write_parquet(manifest)
    out = tmp_path / "soft"
    monkeypatch.setattr("sys.argv", ["prog", "--manifest", str(manifest), "--out", str(out)])
    main()
    made = pl.read_parquet(out / "train_manifest.parquet")
    assert made["outcome"].to_list() == [0.0, 0.0]
    for row in made.iter_rows(named=True):
        with np.load(row["shard"]) as shard:
            assert shard["outcome"].tolist() == [0.0, 0.0]
            assert shard["x"].tolist() == [0 if row["hard_outcome"] == -1 else 1]


def test_initial_soft_targets_preserve_later_hard_outcomes(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index, outcome in enumerate((-1.0, 1.0)):
        shard = source / f"{index}.npz"
        np.savez_compressed(
            shard,
            outcome=np.array([outcome, outcome, outcome], dtype=np.float32),
            history_len=np.array([0, 1, 2]),
            x=np.array([[index], [index + 1], [index + 2]]),
        )
        rows.append({"shard": str(shard), "p1_team": 1, "p2_team": 2,
                     "outcome": outcome, "split": "train", "n_rows": 3})
    manifest = source / "manifest.parquet"
    pl.DataFrame(rows).write_parquet(manifest)

    hybrid = tmp_path / "hybrid"
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--manifest", str(manifest), "--out", str(hybrid),
         "--target-rows", "initial"],
    )
    main()
    for row in pl.read_parquet(hybrid / "train_manifest.parquet").iter_rows(named=True):
        with np.load(row["shard"]) as shard:
            assert shard["outcome"].tolist() == [0.0, row["hard_outcome"], row["hard_outcome"]]
            assert shard["history_len"].tolist() == [0, 1, 2]

    initial = tmp_path / "initial"
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--manifest", str(manifest), "--out", str(initial),
         "--target-rows", "initial", "--output-rows", "targeted"],
    )
    main()
    made = pl.read_parquet(initial / "train_manifest.parquet")
    assert made["n_rows"].to_list() == [1, 1]
    for row in made.iter_rows(named=True):
        with np.load(row["shard"]) as shard:
            assert shard["outcome"].tolist() == [0.0]
            assert shard["history_len"].tolist() == [0]
