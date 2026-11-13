"""Split converted replays by game so decisions from one battle stay together.

Three modes:

- Random (default): ``--holdout-fraction`` of games, chosen by a seeded hash of the game id.
- Chronological: ``--chronological-holdout-games N`` holds out the ``N`` newest games by
  ``uploadtime`` (ties broken by ``id``); every older game is training.
- Fixed: ``--holdout-ids FILE`` holds out exactly the listed game ids. Training excludes
  other games newer than the oldest holdout game.

``--train-sizes S1 S2 ...`` additionally shuffles the training games once with ``--seed`` and
writes ``train_<S>.parquet`` with the first ``S`` games of that order, so every smaller set is
contained in every larger one.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from fog.training.bc import _split_replay_manifest


def _chronological_split(manifest, holdout_games: int):
    """Return (training, holdout): the holdout is the ``holdout_games`` newest games."""
    import polars as pl

    if holdout_games < 1:
        raise ValueError("--chronological-holdout-games must be positive")
    if holdout_games >= len(manifest):
        raise ValueError(
            f"holdout of {holdout_games} games leaves no training games "
            f"(manifest has {len(manifest)})"
        )
    for column in ("id", "uploadtime"):
        if column not in manifest.columns:
            raise ValueError(f"chronological split needs a {column!r} column")
    ordered = manifest.with_columns(pl.col("id").cast(pl.String).alias("_key")).sort(
        ["uploadtime", "_key"]
    )
    cut = len(ordered) - holdout_games
    training = ordered.slice(0, cut).drop("_key")
    holdout = ordered.slice(cut).drop("_key")
    return training, holdout


def _nested_training_sets(training, sizes: list[int], seed: int) -> dict[int, object]:
    """Shuffle training games once and return the first ``S`` games for every size ``S``."""
    import polars as pl

    if not sizes:
        return {}
    if min(sizes) < 1:
        raise ValueError("--train-sizes must be positive")
    if max(sizes) > len(training):
        raise ValueError(
            f"largest training size {max(sizes)} exceeds the {len(training)} training games"
        )
    keyed = training.with_columns(pl.col("id").cast(pl.String).alias("_key")).sort("_key")
    order = list(range(len(keyed)))
    random.Random(seed).shuffle(order)
    shuffled = keyed[order].drop("_key")
    return {size: shuffled.slice(0, size) for size in sorted(set(sizes))}


def _fixed_holdout_split(manifest, ids: list[str]):
    """Return (training, exact holdout, excluded newer game count).

    An upload-time tie with the oldest holdout game is allowed in training, as in
    ``--chronological-holdout-games``. An id must identify exactly one manifest row.
    """
    import polars as pl

    for column in ("id", "uploadtime"):
        if column not in manifest.columns:
            raise ValueError(f"fixed holdout needs a {column!r} column")
    if not ids:
        raise ValueError("--holdout-ids must contain at least one game id")
    if len(ids) != len(set(ids)):
        raise ValueError("--holdout-ids contains duplicate game ids")
    keyed = manifest.with_columns(pl.col("id").cast(pl.String).alias("_key"))
    if keyed["_key"].null_count() or keyed["_key"].n_unique() != len(keyed):
        raise ValueError("fixed holdout requires unique, non-null manifest game ids")
    missing = set(ids) - set(keyed["_key"])
    if missing:
        raise ValueError(f"holdout ids missing from manifest: {', '.join(sorted(missing))}")
    holdout = keyed.filter(pl.col("_key").is_in(ids)).sort(["uploadtime", "_key"])
    if holdout["uploadtime"].null_count():
        raise ValueError("fixed holdout requires non-null holdout upload times")
    oldest = holdout["uploadtime"].min()
    other = keyed.filter(~pl.col("_key").is_in(ids))
    if other["uploadtime"].null_count():
        raise ValueError("fixed holdout requires non-null training upload times")
    training = other.filter(pl.col("uploadtime") <= oldest).sort(["uploadtime", "_key"])
    excluded = len(other) - len(training)
    return training.drop("_key"), holdout.drop("_key"), excluded


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260804)
    holdout_mode = parser.add_mutually_exclusive_group()
    holdout_mode.add_argument(
        "--chronological-holdout-games",
        type=int,
        default=None,
        help="hold out the N newest games by uploadtime instead of a random fraction",
    )
    holdout_mode.add_argument(
        "--holdout-ids",
        type=Path,
        help="hold out exactly these game ids (one per line), excluding newer training games",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="*",
        default=[],
        help="also write nested train_<S>.parquet subsets of the training games",
    )
    parser.add_argument(
        "--skip-oversized-train-sizes",
        action="store_true",
        help="skip and report requested subsets larger than the training set",
    )
    args = parser.parse_args()
    names = ["train_manifest.parquet", "holdout_manifest.parquet"]
    names.extend(f"train_{size}.parquet" for size in args.train_sizes)
    for name in names:
        if (args.out / name).exists():
            raise FileExistsError(f"refusing to replace an existing split: {args.out / name}")

    if args.chronological_holdout_games is None and args.holdout_ids is None:
        train_path, holdout_path = _split_replay_manifest(
            args.manifest, args.out, args.holdout_fraction, args.seed
        )
        training = pl.read_parquet(train_path)
        paths = [train_path, holdout_path]
    else:
        manifest = pl.read_parquet(args.manifest)
        if args.holdout_ids is not None:
            ids = [line.strip() for line in args.holdout_ids.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
            training, holdout, excluded = _fixed_holdout_split(manifest, ids)
            print(f"dropped_newer_games={excluded}")
        else:
            training, holdout = _chronological_split(manifest, args.chronological_holdout_games)
        args.out.mkdir(parents=True, exist_ok=True)
        train_path = args.out / "train_manifest.parquet"
        holdout_path = args.out / "holdout_manifest.parquet"
        training.write_parquet(train_path)
        holdout.write_parquet(holdout_path)
        paths = [train_path, holdout_path]

    if args.train_sizes:
        if "id" not in training.columns:
            raise ValueError("--train-sizes needs an 'id' column")
        sizes = args.train_sizes
        if args.skip_oversized_train_sizes:
            for size in sorted(set(sizes)):
                if size > len(training):
                    print(f"skipped_train_size={size} available_training_games={len(training)}")
            sizes = [size for size in sizes if size <= len(training)]
        for size, subset in _nested_training_sets(training, sizes, args.seed).items():
            path = args.out / f"train_{size}.parquet"
            subset.write_parquet(path)
            paths.append(path)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
