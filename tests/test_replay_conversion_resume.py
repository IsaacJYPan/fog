from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from fog.scripts import convert_replay_decisions as converter


def test_interrupted_conversion_reuses_only_completed_replays(tmp_path, monkeypatch):
    progress = tmp_path / "progress"
    progress.mkdir()
    shards = tmp_path / "shards"
    shards.mkdir()
    calls = []

    def convert(path, **kwargs):
        calls.append(path.stem)
        if path.stem == "second" and calls == ["first", "second"]:
            raise RuntimeError("interrupted before second replay checkpoint")
        shard = shards / f"{path.stem}.npz"
        shard.write_bytes(path.stem.encode())
        return {"manifest": {"id": path.stem, "shard": str(shard)}, "filters": {"accepted": 2}}

    monkeypatch.setattr(converter, "_convert_one", convert)

    def run(name):
        return converter.convert_one(Path(name), shard_dir=shards, min_rating=0,
                                     imputations=1, seed=7, progress_dir=progress)

    expected = run("first")
    with pytest.raises(RuntimeError, match="interrupted"):
        run("second")
    assert not (progress / "second.json").exists()
    assert run("first") == expected
    run("second")
    assert calls == ["first", "second", "second"]
    (shards / "first.npz").unlink()
    with pytest.raises(FileNotFoundError, match="shard is missing"):
        run("first")


def test_resume_requires_the_same_conversion_settings(tmp_path):
    args = Namespace(resume=True, raw=tmp_path / "raw", out=tmp_path / "converted",
                     limit=0, min_rating=1500, imputations=1, seed=7, candidate_bank=None)
    progress = converter._resume_settings(args)
    assert progress.is_dir()
    assert converter._resume_settings(args) == progress
    args.seed = 8
    with pytest.raises(ValueError, match="changed settings"):
        converter._resume_settings(args)


def test_resume_refuses_old_or_unidentified_partial_output(tmp_path):
    (tmp_path / "shards").mkdir()
    args = Namespace(resume=True, raw=tmp_path / "raw", out=tmp_path,
                     limit=0, min_rating=1500, imputations=1, seed=7, candidate_bank=None)
    with pytest.raises(ValueError, match="created without --resume"):
        converter._resume_settings(args)
    args.resume = False
    assert converter._resume_settings(args) is None
