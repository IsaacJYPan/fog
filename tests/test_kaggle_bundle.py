from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from fog.scripts.kaggle_bundle import bundle_files, bundle_for_kaggle
from fog.scripts.kaggle_setup import runtime_requirements


@pytest.fixture
def portable_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for directory in ("fog", "configs", "notebooks", "docs", "data/teams",
                      "server/pokemon-showdown/dist", "server/pokemon-showdown/sim",
                      "server/pokemon-showdown/data", "server/pokemon-showdown/lib"):
        path = tmp_path / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "fixture.txt").write_text("fixture")
    for name in ("package.json", "package-lock.json", "pokemon-showdown"):
        (tmp_path / "server/pokemon-showdown" / name).write_text("{}")
    (tmp_path / "initial.pt").write_bytes(b"checkpoint")
    (tmp_path / "opponent.pt").write_bytes(b"opponent")
    (tmp_path / "data/teams/train.json").write_text("{}")
    shard = tmp_path / "replay/shard.npz"
    shard.parent.mkdir()
    shard.write_bytes(b"shard")
    pq.write_table(pa.table({"id": ["game"], "shard": ["replay\\shard.npz"]}),
                   tmp_path / "replay/manifest.parquet")
    config = {"env": {"backend": "showdown_direct", "team_pool": "data/teams/train.json"},
              "replay_policy": {"manifest": "replay/manifest.parquet"}}
    (tmp_path / "configs/base.yaml").write_text(yaml.safe_dump(config))
    job = {"version": 1, "name": "portable", "output_dir": "runs/job",
           "base_config": "configs/base.yaml", "initial_checkpoint": "initial.pt",
           "initial_league_checkpoints": ["opponent.pt"],
           "training": {"blocks": 1, "decisions_per_block": 100},
           "evaluation": {"enabled": False}}
    path = tmp_path / "configs/job.yaml"
    path.write_text(yaml.safe_dump(job))
    return tmp_path, path


def test_bundle_includes_opponents_runtime_and_portable_shard_paths(portable_job):
    root, job = portable_job
    destination = root / "bundle.zip"
    bundle_for_kaggle(job, destination)
    with zipfile.ZipFile(destination) as archive:
        assert "fog/opponent.pt" in archive.namelist()
        assert "fog/server/pokemon-showdown/package-lock.json" in archive.namelist()
        assert "fog/server/pokemon-showdown/sim/fixture.txt" in archive.namelist()
        table = pq.read_table(io.BytesIO(archive.read("fog/replay/manifest.parquet")))
        assert table.column("shard").to_pylist() == ["replay/shard.npz"]
        assert "fog/replay/shard.npz" in archive.namelist()
        assert json.loads(archive.read("fog/bundle_manifest.json"))["ready_for_training"]


def test_account_local_settings_are_excluded_from_training_archives(portable_job):
    root, job = portable_job
    settings = root / "configs/kaggle_results.json"
    settings.write_text('{"kernel": "private-account/saved-notebook"}')
    example = root / "configs/kaggle_results.example.json"
    example.write_text('{"kernel": "your-account/your-notebook"}')
    destination = root / "bundle.zip"
    bundle_for_kaggle(job, destination)
    with zipfile.ZipFile(destination) as archive:
        assert "fog/configs/kaggle_results.json" not in archive.namelist()
        assert "fog/configs/kaggle_results.example.json" in archive.namelist()
        assert "fog/configs/base.yaml" in archive.namelist()


def test_missing_opponent_blocks_full_bundle_but_is_reported_in_setup_bundle(portable_job):
    root, job = portable_job
    (root / "opponent.pt").unlink()
    with pytest.raises(FileNotFoundError, match="opponent.pt"):
        bundle_files(job, root)
    bundle_for_kaggle(job, root / "setup.zip", allow_missing_inputs=True)
    with zipfile.ZipFile(root / "setup.zip") as archive:
        manifest = json.loads(archive.read("fog/bundle_manifest.json"))
        assert not manifest["ready_for_training"]
        assert manifest["missing_inputs"] == ["opponent.pt"]


def test_bundle_checks_overridden_training_pool(portable_job):
    root, path = portable_job
    job = yaml.safe_load(path.read_text())
    job["overrides"] = {"env.team_pool": "data/teams/missing.json"}
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(FileNotFoundError, match="data/teams/missing.json"):
        bundle_files(path, root)


def test_kaggle_installer_preserves_host_torch_and_omits_rust_engine():
    root = Path(__file__).resolve().parents[1]
    requirements = runtime_requirements(root)
    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert not any(requirement.startswith(("torch=", "poke-engine=")) for requirement in requirements)


def test_explicit_provenance_inputs_are_required_and_bundled(portable_job):
    root, path = portable_job
    job = yaml.safe_load(path.read_text())
    job["bundle_inputs"] = ["replay/selection.json"]
    path.write_text(yaml.safe_dump(job))
    with pytest.raises(FileNotFoundError, match="selection.json"):
        bundle_files(path, root)
    provenance = root / "replay/selection.json"
    provenance.write_text('{"private_truth": false}')
    assert provenance in bundle_files(path, root)
