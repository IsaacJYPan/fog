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


@pytest.mark.parametrize("expanded,filename", [
    (False, "replay-work.tar.gz"), (False, "replay-work.bin"), (True, None),
])
def test_restore_prepared_accepts_archive_and_expanded_dataset(tmp_path, expanded, filename):
    source = tmp_path / "input/dataset"
    source.mkdir(parents=True)
    files = {"run.json": b'{"steps": {}}', "decisions/manifest.parquet": b"manifest",
             "split/holdout_manifest.parquet": b"holdout", "decisions/shards/a.npz": b"shard"}
    for name, contents in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    if not expanded:
        archive_path = tmp_path / "input" / filename
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in files:
                archive.add(source / name, arcname=name)
        import shutil

        shutil.rmtree(source)
    else:
        for path in source.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        source.chmod(0o555)
    work = tmp_path / "work"
    _job().restore_prepared(tmp_path / "input", work)
    assert {p.relative_to(work).as_posix(): p.read_bytes()
            for p in work.rglob("*") if p.is_file()} == files
    (work / "run.json").write_text("restored")
    (work / "decisions/new-manifest.tmp").write_text("rebased")


def test_restore_prepared_refuses_missing_input(tmp_path):
    with pytest.raises(AssertionError, match="found 0"):
        _job().restore_prepared(tmp_path, tmp_path / "work")


def test_worker_deadline_preserves_completed_output_and_stops_process(tmp_path):
    import os
    import sys

    job = _job()
    job.ROOT = tmp_path
    with (tmp_path / "log.txt").open("w") as log:
        status = job.run_worker(
            [sys.executable, "-c", "import pathlib,time; "
             "pathlib.Path('checkpoint').write_text('saved'); time.sleep(60)"],
            seconds=1, log=log, env=dict(os.environ),
        )
    assert status == "interrupted_for_export"
    assert (tmp_path / "checkpoint").read_text() == "saved"


def test_worker_retains_and_forwards_epoch_output(tmp_path, capsys):
    import os
    import sys

    job = _job()
    job.ROOT = tmp_path
    with (tmp_path / "log.txt").open("w") as log:
        status = job.run_worker([sys.executable, "-c", "print('epoch saved', flush=True)"],
                                seconds=5, log=log, env=dict(os.environ))
    assert status == "returned"
    assert (tmp_path / "log.txt").read_text() == "epoch saved\n"
    assert "epoch saved" in capsys.readouterr().out
