"""Versioned, atomic full-state checkpoints at optimizer-update boundaries."""

from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch

STATE_VERSION = 1


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA training state must resume on CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_training_state(
    path, *, model, optimizer, sampler, route_generator, update, identity, config, elapsed_seconds
):
    atomic_save(
        {
            "version": STATE_VERSION,
            "identity": identity,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "route_rng": route_generator.get_state(),
            "rng": rng_state(),
            "update": update,
            "elapsed_seconds": elapsed_seconds,
            "precision": "fp32",
            "scheduler": {"completed_updates": update, "warmup_updates": config["warmup_updates"]},
            "save_boundary": "completed optimizer update; no pending accumulated gradients",
        },
        Path(path),
    )


def load_training_state(path, *, model, optimizer, sampler, route_generator, identity, config):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state["version"] != STATE_VERSION
        or state["identity"] != identity
        or state["config"] != config
    ):
        raise ValueError("Training-state version, experiment identity or immutable recipe mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    route_generator.set_state(state["route_rng"].cpu())
    if (
        state["update"] != sampler.update
        or state["scheduler"]["completed_updates"] != state["update"]
    ):
        raise ValueError("Training-state progress counters disagree")
    restore_rng(state["rng"])
    return state
