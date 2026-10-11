"""Register a private frozen notebook through Kaggle's non-executing save API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def quick_save(folder, timeout, *, bootstrap=False, experiment_sha256=None):
    # Run this module in the separately authenticated Kaggle client environment.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    if metadata.get("is_private") is not True:
        raise ValueError("Only private notebooks may be registered")
    source = (folder / metadata["code_file"]).resolve()
    if not source.is_relative_to(folder.resolve()):
        raise ValueError("Notebook source must stay inside its frozen package")
    request = ApiSaveKernelRequest()
    request.slug = metadata["id"]
    request.new_title = metadata["title"]
    request.text = source.read_text(encoding="utf-8")
    if bootstrap:
        if not metadata["enable_gpu"] or not experiment_sha256:
            raise ValueError("CPU registration must identify a frozen GPU training package")
        receipt = json.dumps(
            {
                "kernel": metadata["id"],
                "experiment_sha256": experiment_sha256,
                "registration_only": True,
                "gpu_requested": False,
            },
            indent=2,
        )
        request.text = (
            "from pathlib import Path\n"
            f"Path('/kaggle/working/REGISTERED.json').write_text({receipt!r})\n"
            "print('CPU registration complete; training has not started')\n"
        )
    request.language = metadata["language"]
    request.kernel_type = metadata["kernel_type"]
    request.is_private = True
    request.enable_gpu = False if bootstrap else metadata["enable_gpu"]
    request.enable_tpu = False
    request.enable_internet = False if bootstrap else metadata["enable_internet"]
    request.dataset_data_sources = [] if bootstrap else metadata["dataset_sources"]
    request.kernel_data_sources = [] if bootstrap else metadata["kernel_sources"]
    request.session_timeout_seconds = 600 if bootstrap else timeout
    request.kernel_execution_type = (
        KernelExecutionType.SAVE_AND_RUN_ALL if bootstrap else KernelExecutionType.QUICK_SAVE
    )
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.save_kernel(request)
    if response.error or response.invalid_dataset_sources or response.invalid_kernel_sources:
        raise RuntimeError(
            "Kaggle rejected notebook registration: "
            + json.dumps(
                {
                    "error": response.error,
                    "invalid_datasets": response.invalid_dataset_sources,
                    "invalid_notebooks": response.invalid_kernel_sources,
                }
            )
        )
    return {
        "kernel": metadata["id"],
        "version": response.version_number,
        "executed": bootstrap,
        "registration_only": bootstrap,
        "gpu_requested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--experiment-sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            quick_save(
                args.folder,
                args.timeout,
                bootstrap=args.bootstrap,
                experiment_sha256=args.experiment_sha256,
            )
        )
    )


if __name__ == "__main__":
    main()
