"""Bounded exports at shared, predeclared optimizer-update milestones."""

from __future__ import annotations

import math
from pathlib import Path
import shutil

import torch

from fog.pretrained.kakuna import INTERFACE_VERSION, sha256
from fog.pretrained.state import atomic_save

AUXILIARY_VERSION = 1


def anchor_updates(planned: int) -> set[int]:
    if planned < 1:
        raise ValueError("An experiment must plan at least one optimizer update")
    return {math.ceil(planned * fraction / 4) for fraction in range(1, 5)}


def export_candidates(model, out: Path, *, update: int, identity: str, config: dict, final=False):
    anchors = anchor_updates(config["updates"])
    native = out / f"native_{update:06d}.pt"
    atomic_save(model.native.state_dict(), native)
    atomic_save(
        {
            "version": AUXILIARY_VERSION,
            "interface": INTERFACE_VERSION,
            "identity": identity,
            "update": update,
            "config": config,
            "native_sha256": sha256(native),
            "heads": {
                key: value
                for key, value in model.state_dict().items()
                if not key.startswith("native.")
            },
        },
        out / f"auxiliary_{update:06d}.pt",
    )
    # Both arms preserve the same quarter-update anchors even if one is stopped
    # earlier. Keep only the latest periodic export in addition to those anchors.
    saved = sorted(out.glob("native_[0-9]*.pt"))
    latest = max(int(path.stem.split("_")[1]) for path in saved)
    for path in saved:
        completed = int(path.stem.split("_")[1])
        if completed not in anchors and completed != latest:
            path.unlink()
            path.with_name(f"auxiliary_{completed:06d}.pt").unlink(missing_ok=True)
    if final:
        # Numerical validation and milestone selection must identify the same
        # serialized bytes, independently of torch.save's archive metadata.
        temporary = out / "native_final.pt.tmp"
        shutil.copyfile(native, temporary)
        temporary.replace(out / "native_final.pt")


def load_auxiliary(model, checkpoint: Path, native: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state["version"] != AUXILIARY_VERSION
        or state["interface"] != INTERFACE_VERSION
        or state["native_sha256"] != sha256(native)
    ):
        raise ValueError("Auxiliary heads belong to a different native checkpoint or interface")
    expected = {key for key in model.state_dict() if not key.startswith("native.")}
    if set(state["heads"]) != expected:
        raise ValueError("Auxiliary checkpoint architecture changed")
    current = model.state_dict()
    current.update(state["heads"])
    model.load_state_dict(current, strict=True)
    return state
