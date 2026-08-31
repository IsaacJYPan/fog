"""Versioned, exact-resume state for PPO training.

Weight-only ``ppo_*.pt`` files remain the portable inference/league format.  Each one can have a
``*.state.pt`` sidecar containing the mutable training process state required for an exact resume
at an update boundary.  Keeping the formats separate avoids making frozen league opponents depend
on optimizer internals.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRAINING_STATE_VERSION = 1


def state_path_for(weights_path: str | Path) -> Path:
    path = Path(weights_path)
    return path.with_name(f"{path.stem}.state{path.suffix}")


def _config_digest(config: dict[str, Any]) -> str:
    normalized = deepcopy(config)
    normalized.get("ppo", {}).pop("resume_ckpt", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_training_state(setup: Any) -> dict[str, Any]:
    """Capture a setup at an update boundary, where its rollout buffer is empty."""
    if setup.buf.n_decisions:
        raise RuntimeError("exact checkpoints may only be saved with an empty rollout buffer")
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    state: dict[str, Any] = {
        "format": "fog.ppo.training-state",
        "version": TRAINING_STATE_VERSION,
        "config_digest": _config_digest(setup.cfg),
        "agent": setup.agent.state_dict(),
        "optimizer": setup.opt.state_dict(),
        "progress": {
            "decisions_seen": int(setup.decisions_seen),
            "next_snapshot": int(setup.next_snapshot),
            "next_refreeze": int(setup.next_refreeze),
            "metrics": dict(setup.metrics),
            "pool_sample_counts": np.asarray(setup.pool_sample_counts).copy(),
        },
        "league": {
            "past_ckpts": [str(path) for path in setup.league.past_ckpts],
            "latest_ckpt": (
                str(setup.league.latest_ckpt) if setup.league.latest_ckpt is not None else None
            ),
        },
        "random": {
            "python_global": random.getstate(),
            "python_training": setup.rng.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": cuda_rng,
            "backend": (
                setup.backend_rng.getstate()
                if getattr(setup, "backend_rng", None) is not None
                else None
            ),
        },
        "replay_regularizer": setup.replay_regularizer.state_dict(),
    }
    if setup.fresh_probe is not None:
        state["fresh_probe"] = setup.fresh_probe.state_dict()
    if setup.fresh_opt is not None:
        state["fresh_optimizer"] = setup.fresh_opt.state_dict()
    return state


def save_training_state(setup: Any, weights_path: str | Path) -> Path:
    """Save inference weights plus an exact-resume sidecar and return the sidecar path."""
    weights_path = Path(weights_path)
    torch.save(setup.agent.state_dict(), weights_path)
    state_path = state_path_for(weights_path)
    torch.save(build_training_state(setup), state_path)
    return state_path


def restore_training_state(setup: Any, state_path: str | Path) -> None:
    payload = torch.load(state_path, map_location=setup.device, weights_only=False)
    if payload.get("format") != "fog.ppo.training-state":
        raise ValueError(f"not a FOG PPO training-state checkpoint: {state_path}")
    if int(payload.get("version", -1)) != TRAINING_STATE_VERSION:
        raise ValueError(
            f"unsupported PPO training-state version {payload.get('version')}; "
            f"expected {TRAINING_STATE_VERSION}"
        )
    expected_digest = _config_digest(setup.cfg)
    if payload.get("config_digest") != expected_digest:
        raise ValueError(
            "resume configuration differs from the saved run; use ppo.init_ckpt for a "
            "weight-only warm start or resume with the original configuration"
        )

    setup.agent.load_state_dict(payload["agent"])
    setup.opt.load_state_dict(payload["optimizer"])
    progress = payload["progress"]
    setup.decisions_seen = int(progress["decisions_seen"])
    setup.next_snapshot = int(progress["next_snapshot"])
    setup.next_refreeze = int(progress["next_refreeze"])
    setup.metrics = dict(progress.get("metrics", {}))
    counts = np.asarray(progress["pool_sample_counts"], dtype=np.int64)
    if counts.shape != setup.pool_sample_counts.shape:
        raise ValueError(
            "resume team-pool size differs from the saved run: "
            f"{counts.shape} != {setup.pool_sample_counts.shape}"
        )
    setup.pool_sample_counts[...] = counts

    league = payload["league"]
    setup.league.past_ckpts[:] = [Path(path) for path in league["past_ckpts"]]
    latest = league.get("latest_ckpt")
    setup.league.latest_ckpt = Path(latest) if latest is not None else None

    rng = payload["random"]
    random.setstate(rng["python_global"])
    setup.rng.setstate(rng["python_training"])
    np.random.set_state(rng["numpy_global"])
    torch.random.set_rng_state(rng["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda"]])
    setup.resume_backend_rng_state = rng.get("backend")
    setup.replay_regularizer.load_state_dict(payload["replay_regularizer"])

    if "fresh_probe" in payload:
        if setup.fresh_probe is None or setup.fresh_opt is None:
            raise ValueError("saved run has fresh-probe state but resumed config disables it")
        setup.fresh_probe.load_state_dict(payload["fresh_probe"])
        setup.fresh_opt.load_state_dict(payload["fresh_optimizer"])
