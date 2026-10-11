"""Explicit weight continuation without weakening immutable full-state resume."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from fog.pretrained.checkpoints import AUXILIARY_VERSION, load_auxiliary
from fog.pretrained.kakuna import INTERFACE_VERSION, sha256


def validate_warm_start(config, *, samples=None):
    warm = config.get("warm_start")
    if warm is None:
        return {}
    if (
        warm.get("version") != 1
        or warm.get("optimizer") != "reset"
        or warm.get("sampler") != "reset"
        or set(warm.get("sources", {})) != {"ordinary", "joint"}
    ):
        raise ValueError("Weight continuation requires two matched sources and explicit resets")
    configs, updates, metadata = [], [], {}
    for condition in ("ordinary", "joint"):
        source = warm["sources"][condition]
        for name in ("native", "auxiliary", "reference"):
            if sha256(Path(source[name])) != source[name + "_sha256"]:
                raise ValueError("Warm-start input hash changed: " + condition + "/" + name)
        state = torch.load(source["auxiliary"], map_location="cpu", weights_only=True)
        parent = state["config"]
        identity = hashlib.sha256(
            json.dumps(parent, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            state["version"] != AUXILIARY_VERSION
            or state["interface"] != INTERFACE_VERSION
            or state["native_sha256"] != source["native_sha256"]
            or state["identity"] != identity
            or parent["condition"] != condition
            or state["update"] != source["update"]
            or type(source["update"]) is not int or source["update"] < 1
        ):
            raise ValueError("Warm-start source lineage or native/auxiliary pairing differs")
        reference = torch.load(source["reference"], map_location="cpu", weights_only=True)
        if (
            reference["checkpoint_sha256"] != source["native_sha256"]
            or (samples is not None and reference["samples_sha256"] != sha256(samples))
            or not reference["probabilities"]
            or any(not bool(torch.isfinite(row).all()) for row in reference["probabilities"])
        ):
            raise ValueError("Warm-start CPU probability reference differs")
        configs.append({key: value for key, value in parent.items() if key != "condition"})
        updates.append(state["update"])
        metadata[condition] = {"native_sha256": source["native_sha256"],
                               "auxiliary_sha256": source["auxiliary_sha256"],
                               "source_identity": identity, "source_update": state["update"]}
    if configs[0] != configs[1] or updates[0] != updates[1]:
        raise ValueError("Warm-start sources must have matched parent recipes and updates")
    return metadata


def starting_checkpoint(config):
    warm = config.get("warm_start")
    return Path(warm["sources"][config["condition"]]["native"] if warm else config["checkpoint"])


def initialize_auxiliary(model, config):
    warm = config.get("warm_start")
    if warm:
        source = warm["sources"][config["condition"]]
        load_auxiliary(model, Path(source["auxiliary"]), Path(source["native"]))


def training_optimizer(model, config):
    rate = config["learning_rate"]
    head_rate = config.get("head_learning_rate", rate)
    if not all(math.isfinite(value) and value > 0 for value in (rate, head_rate)):
        raise ValueError("Native and auxiliary learning rates must be finite and positive")
    native = list(model.native.trainable_params)
    if config["condition"] != "joint":
        parameters = native
    else:
        parameters = model.trainable_parameters()
    if config["condition"] == "joint" and "head_learning_rate" in config:
        native_ids = {id(parameter) for parameter in native}
        heads = [parameter for parameter in parameters if id(parameter) not in native_ids]
        groups = [{"params": native, "lr": rate, "role": "native"},
                  {"params": heads, "lr": head_rate, "role": "auxiliary"}]
    else:
        # Preserve the old one-group optimizer layout for existing full-state resumes.
        groups = parameters
    return torch.optim.AdamW(groups, lr=rate, weight_decay=0.0), parameters


def set_learning_rates(optimizer, config, update):
    fraction = min(1.0, (update + 1) / config["warmup_updates"])
    for group in optimizer.param_groups:
        base = (config.get("head_learning_rate", config["learning_rate"])
                if group.get("role") == "auxiliary" else config["learning_rate"])
        group["lr"] = base * fraction
    return {group.get("role", "native"): group["lr"] for group in optimizer.param_groups}
