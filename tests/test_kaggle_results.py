import json
from pathlib import Path
import zipfile
from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fog.scripts.kaggle_results import import_archive, remote


def bundle(path: Path, extra=None):
    files = {
        "pipeline_state.json": json.dumps({"status": "complete", "blocks": []}),
        "block_001/checkpoints/ppo_final.pt": b"test weights",
        "block_001/checkpoints/ppo_final.state.pt": b"test optimizer",
        "block_001/metrics.csv": "step,loss\n98,0.9\n",
        "audits/direct/summary.json": json.dumps({"passed": True, "error_count": 0}),
    }
    files.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            entry = zipfile.ZipInfo()
            entry.filename = name  # Preserve malicious separators on Windows too.
            archive.writestr(entry, data)


def test_import_is_repeatable_and_handoff_paths_are_local(tmp_path):
    archive = tmp_path / "result.zip"
    bundle(archive)
    output = tmp_path / "runs"
    result = import_archive(archive, output, "owner/notebook", 5)
    report = json.loads((result / "RESULTS.json").read_text())
    checkpoint = report["checkpoints"][0]
    assert Path(checkpoint["path"]).is_file()
    assert Path(checkpoint["optimizer_state"]).is_file()
    assert report["metrics"][0]["last"]["step"] == "98"
    assert report["audits"][0]["passed"] is True
    assert report["source"]["version"] == 5
    assert import_archive(archive, output, "owner/notebook", 5) == result
    assert (result / "HANDOFF.md").is_file()


@pytest.mark.parametrize("name", ["../escaped", "C:/escaped", "safe/../../escaped",
                                   "safe\\escaped", "NUL.txt", "safe/file:stream"])
def test_unsafe_archive_writes_no_result(tmp_path, name):
    archive = tmp_path / "bad.zip"
    bundle(archive, {name: "bad"})
    output = tmp_path / "runs"
    with pytest.raises(ValueError, match="Unsafe"):
        import_archive(archive, output, "owner/notebook", 5)
    assert not (output / "latest-import.json").exists()
    assert not list(output.rglob("pipeline_state.json"))


def test_malformed_metadata_does_not_publish_partial_import(tmp_path):
    archive = tmp_path / "bad.zip"
    bundle(archive, {"audits/direct/summary.json": "not json"})
    output = tmp_path / "runs"
    with pytest.raises(json.JSONDecodeError):
        import_archive(archive, output, "owner/notebook", 5)
    assert not (output / "latest-import.json").exists()
    assert not list(output.rglob("pipeline_state.json"))


def test_different_results_never_overwrite_existing_run(tmp_path):
    archive = tmp_path / "result.zip"
    bundle(archive)
    first = import_archive(archive, tmp_path / "runs", "owner/notebook", 5)
    bundle(archive, {"block_001/checkpoints/ppo_final.pt": b"new weights"})
    second = import_archive(archive, tmp_path / "runs", "owner/notebook", 5)
    assert first != second
    assert (first / "block_001/checkpoints/ppo_final.pt").read_bytes() == b"test weights"


def test_remote_checks_run_id_and_downloads_only_export(tmp_path, monkeypatch):
    seen = []
    def listing(request):
        seen.append((request.user_name, request.kernel_slug,
                     getattr(request, "page_token", None)))
        if len(seen) == 1:
            return SimpleNamespace(files=[SimpleNamespace(file_name="unrelated.pt")],
                                   next_page_token="next")
        return SimpleNamespace(files=[SimpleNamespace(file_name="result.zip",
                               url="https://www.kaggleusercontent.com/kf/351470011/signed/result.zip")],
                               next_page_token="")
    service = SimpleNamespace(list_kernel_session_output=listing)
    connection = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=service))
    api_module = ModuleType("kaggle.api.kaggle_api_extended")
    api_module.KaggleApi = lambda: SimpleNamespace(
        authenticate=lambda: None, build_kaggle_client=lambda: nullcontext(connection))
    types_module = ModuleType("kagglesdk.kernels.types.kernels_api_service")
    types_module.ApiGetKernelSessionStatusRequest = SimpleNamespace
    types_module.ApiListKernelSessionOutputRequest = SimpleNamespace
    requests_module = ModuleType("requests")
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return nullcontext(SimpleNamespace(status_code=200, iter_content=lambda size: [b"archive"]))
    requests_module.get = get
    monkeypatch.setitem(sys.modules, api_module.__name__, api_module)
    monkeypatch.setitem(sys.modules, types_module.__name__, types_module)
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    remote("owner/notebook", 5, tmp_path, "result.zip", 351470011)
    assert seen == [("owner", "notebook", None), ("owner", "notebook", "next")]
    assert len(calls) == 1
    assert (tmp_path / "result.zip").read_bytes() == b"archive"
    seen.clear()
    calls.clear()
    with pytest.raises(RuntimeError, match="different"):
        remote("owner/notebook", 4, tmp_path, "result.zip", 111)
    assert not calls
