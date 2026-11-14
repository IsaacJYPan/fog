"""Exercise the two Kaggle input layouts without launching a hosted job."""
import importlib.util
import tarfile
from pathlib import Path

import pytest


def _job():
    spec = importlib.util.spec_from_file_location(
        "replay_gpu_job", Path(__file__).resolve().parents[1]
        / "docs/briefs/05-artifacts/gpu_job.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("expanded", [False, True])
def test_restore_prepared_accepts_archive_and_expanded_dataset(tmp_path, expanded):
    source = tmp_path / "input/dataset"
    source.mkdir(parents=True)
    files = {"run.json": b'{"steps": {}}', "decisions/manifest.parquet": b"manifest",
             "split/holdout_manifest.parquet": b"holdout", "decisions/shards/a.npz": b"shard"}
    for name, contents in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    if not expanded:
        archive_path = tmp_path / "input/replay-work.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in files:
                archive.add(source / name, arcname=name)
        import shutil

        shutil.rmtree(source)
    work = tmp_path / "work"
    _job().restore_prepared(tmp_path / "input", work)
    assert {p.relative_to(work).as_posix(): p.read_bytes()
            for p in work.rglob("*") if p.is_file()} == files


def test_restore_prepared_refuses_missing_input(tmp_path):
    with pytest.raises(AssertionError, match="found 0"):
        _job().restore_prepared(tmp_path, tmp_path / "work")
