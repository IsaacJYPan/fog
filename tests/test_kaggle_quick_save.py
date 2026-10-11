from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

from fog.scripts.kaggle_quick_save import quick_save


def test_registration_overrides_gpu_recipe_with_a_cpu_only_placeholder(tmp_path, monkeypatch):
    metadata = {
        "id": "owner/training",
        "title": "training",
        "code_file": "run.py",
        "kernel_type": "script",
        "language": "python",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["owner/private-training-data"],
        "kernel_sources": [],
    }
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "run.py").write_text("raise RuntimeError('Training must not execute')")
    calls = []

    def save(request):
        calls.append(request)
        return SimpleNamespace(
            error=None, invalid_dataset_sources=[], invalid_kernel_sources=[], version_number=1
        )

    connection = SimpleNamespace(
        kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(save_kernel=save))
    )
    api = SimpleNamespace(
        authenticate=lambda: None, build_kaggle_client=lambda: nullcontext(connection)
    )
    modules = {
        "kaggle.api.kaggle_api_extended": {"KaggleApi": lambda: api},
        "kagglesdk.kernels.types.kernels_api_service": {"ApiSaveKernelRequest": SimpleNamespace},
        "kagglesdk.kernels.types.kernels_enums": {
            "KernelExecutionType": SimpleNamespace(SAVE_AND_RUN_ALL="run", QUICK_SAVE="save")
        },
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    result = quick_save(tmp_path, 7200, bootstrap=True, experiment_sha256="a" * 64)
    (request,) = calls
    assert request.enable_gpu is False and request.enable_tpu is False
    assert request.kernel_execution_type == "run" and request.session_timeout_seconds == 600
    assert request.dataset_data_sources == [] and request.kernel_data_sources == []
    assert "Training must not execute" not in request.text
    assert "REGISTERED.json" in request.text and "a" * 64 in request.text
    assert result["registration_only"] and result["gpu_requested"] is False
