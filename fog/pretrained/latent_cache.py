"""Verified complete-history native features for final-block belief adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from fog.pretrained.kakuna import INTERFACE_VERSION, KAKUNA_34_SHA256, sha256
from fog.pretrained.retention import action_values, legal_policy, upper_forward, upper_modules
from fog.pretrained.state import atomic_save

CACHE_FORMAT = "fog.kakuna-upper-features"
CACHE_VERSION = 1


def tensor_digest(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def windows(record, indices, device="cpu"):
    """Right padding leaves every selected valid output causally unchanged."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    features = record["features"]
    if indices.ndim != 1 or not len(indices) or bool((indices < 0).any()) or bool((indices >= len(features)).any()):
        raise ValueError("Feature queries must index actual nonterminal decisions")
    lengths = torch.minimum(indices + 1, torch.tensor(97))
    maximum = int(lengths.max())
    packed = features.new_zeros((len(indices), maximum, features.shape[-1]))
    positions = torch.zeros((len(indices), maximum), dtype=torch.long)
    for row, index in enumerate(indices.tolist()):
        length = int(lengths[row])
        start = index + 1 - length
        packed[row, :length] = features[start:index + 1]
        positions[row, :length] = record["positions"][start:index + 1]
    return packed.to(device), positions.to(device), lengths.to(device)


@torch.no_grad()
def extract(runtime, example, targets, *, verify_cached=False, tolerance=1e-3):
    """No target tensor is passed to native embedding, actor or critic inference."""
    native = runtime.model
    native.eval()
    block, _ = upper_modules(native)
    length = len(example.actions)
    if length < 1 or len(example.rl2s) != length + 1 or int(example.time_idxs[0]) != 0:
        raise ValueError("Extraction requires a complete chronological prefix starting at zero")
    observations = {key: value[None, :length] for key, value in example.obs.items()}
    captured = []
    handle = block.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    try:
        latent, _ = native.get_state_embedding(
            obs=observations,
            rl2s=example.rl2s[None, :length],
            time_idxs=example.time_idxs[None, :length],
        )
    finally:
        handle.remove()
    if len(captured) != 1 or latent.shape[:2] != (1, length):
        raise ValueError("Unexpected native final-block boundary")
    actor_obs = {key: observations[key] for key in native.pass_obs_keys_to_actor}
    probabilities = legal_policy(native, latent, actor_obs)
    values = []
    for start in range(0, length, 16):
        values.append(action_values(native, latent[:, start:start + 16]).cpu())
    reference_values = torch.cat(values, dim=1)
    public = {
        "features": captured[0][0].cpu(),
        "positions": example.time_idxs[:length].reshape(length).cpu(),
        "reference_latent": latent[0].cpu(),
        "reference_probabilities": probabilities[0].cpu(),
        "reference_values": reference_values[0],
        "actor_observations": {key: value[0].cpu() for key, value in actor_obs.items()},
    }
    indices = sorted({0, min(96, length - 1), length // 2, length - 1})
    features, positions, lengths = windows(public, indices)
    recreated = upper_forward(native, features, positions, lengths)
    obs = {key: value[indices, None] for key, value in public["actor_observations"].items()}
    recreated_probabilities = legal_policy(native, recreated[:, None], obs)[:, 0]
    latent_error = float((recreated - public["reference_latent"][indices]).abs().max())
    probability_error = float((recreated_probabilities - public["reference_probabilities"][indices]).abs().max())
    if probability_error > tolerance or latent_error > 5e-3:
        raise ValueError(f"Final-block replay differs from complete native history: {probability_error=}, {latent_error=}")
    cached_error = None
    if verify_cached:
        hidden = native.traj_encoder.init_hidden_state(1, torch.device("cpu"))
        cached_error = 0.0
        for index in range(length):
            p, _latent, hidden = runtime.distribution(
                {key: value[:, index:index + 1] for key, value in observations.items()},
                example.rl2s[None, index:index + 1],
                example.time_idxs[None, index:index + 1],
                hidden,
            )
            cached_error = max(cached_error, float((p[0, 0] - probabilities[0, index]).abs().max()))
        if cached_error > tolerance:
            raise ValueError("Full-history and live cached native probabilities disagree")
    # Targets are attached only after player-visible feature extraction and checks.
    public["labels"] = {key: value.cpu().clone() for key, value in targets.items()}
    if any(not torch.isfinite(x).all() for x in (public["features"], public["reference_latent"], public["reference_probabilities"], public["reference_values"])):
        raise FloatingPointError("Nonfinite cached native features")
    return public, {"positions": length, "upper_probability_max_error": probability_error,
                    "upper_latent_max_error": latent_error, "live_cached_probability_max_error": cached_error}


def build(config, out, *, maximum_games=None):
    from fog.pretrained.data import assert_disjoint
    from fog.pretrained.kakuna import load_kakuna
    from fog.pretrained.loading import SequenceLoader, read_record
    from fog.team_identity import team_group_key

    torch.set_num_threads(config.get("cpu_threads", 2))
    out = Path(out)
    if out.exists():
        raise FileExistsError("Keep the existing cache; use a new experiment identity")
    out.mkdir(parents=True)
    started = time.monotonic()
    checkpoint = Path(config["checkpoint"])
    if sha256(checkpoint) != KAKUNA_34_SHA256:
        raise ValueError("Cache requires protected Kakuna checkpoint 34")
    source = Path(config["source_manifest"])
    if sha256(source) != config["source_manifest_sha256"]:
        raise ValueError("Trace-verification manifest changed")
    manifest = json.loads(source.read_text())
    if not manifest.get("complete") or manifest.get("format") != "fog.teacher-replay-data":
        raise ValueError("Require the completed trace-verified source corpus")
    pools = {split: json.loads(Path(row["pool"]).read_text()) for split, row in config["cohorts"].items()}
    assert_disjoint(pools)
    for split, row in config["cohorts"].items():
        if sha256(Path(row["pool"])) != manifest["recipe"]["inputs"][split]["pool_sha256"]:
            raise ValueError("Pool differs from trace-verified source split")
    runtime = load_kakuna(checkpoint)
    fingerprint = tensor_digest(runtime.model.state_dict())
    loader = SequenceLoader(runtime, sequence_length=config["maximum_decisions"])
    inventory, counts, groups = [], {}, {}
    for split, cohort in config["cohorts"].items():
        records = [r for r in manifest["records"] if r["split"] == split]
        if len(records) != manifest["counts"][split + "_games"]:
            raise ValueError("Trace-verification cohort count changed")
        records = records[:maximum_games] if maximum_games is not None else records
        allowed = {team_group_key(team) for team in pools[split]["teams"]}
        groups[split] = set()
        counts[split] = {"games": 0, "perspectives": 0, "positions": 0}
        for game, row in enumerate(records):
            path = Path(cohort["folder"]) / row["source"]
            if sha256(path) != row["source_sha256"] or not row["exact_public_trace"] or not row["native_legality_exact"]:
                raise ValueError("Source game lacks the recorded exact trace or changed bytes")
            record = read_record(path)
            if record.get("choice_rejections") or not record["training_eligible"] or record.get("failures") or record.get("fallbacks"):
                raise ValueError("Unverified/rejected source game cannot enter this cache")
            compositions = {team_group_key(team) for team in record["submitted_teams"].values()}
            if not compositions.issubset(allowed):
                raise ValueError("Source team is outside its composition split")
            groups[split].update(compositions)
            for pov_seed in (0, 1):
                example, targets = loader.load(("fresh", str(path), pov_seed))
                if len(example.actions) != row["decisions"]["p2" if pov_seed == 0 else "p1"]:
                    raise ValueError("A complete source perspective was cropped or remapped")
                cached, proof = extract(runtime, example, targets, verify_cached=game == 0)
                name = f"{split}/{game:06d}-{pov_seed}.pt"
                atomic_save(cached, out / name)
                inventory.append({"path": name, "sha256": sha256(out / name), "split": split,
                                  "game": row["id"], "pov_seed": pov_seed,
                                  "source_sha256": row["source_sha256"], "compositions": sorted(compositions), **proof})
                counts[split]["perspectives"] += 1
                counts[split]["positions"] += proof["positions"]
            counts[split]["games"] += 1
            if game < 2 or (game + 1) % 25 == 0:
                print(json.dumps({"split": split, "games": game + 1, "counts": counts[split],
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    if groups["train"] & groups["development"]:
        raise ValueError("Actual cached team compositions overlap")
    if tensor_digest(runtime.model.state_dict()) != fingerprint or sha256(checkpoint) != KAKUNA_34_SHA256:
        raise ValueError("Frozen native parameters or buffers changed during extraction")
    result = {"format": CACHE_FORMAT, "version": CACHE_VERSION, "complete": True,
              "interface": INTERFACE_VERSION, "checkpoint_sha256": KAKUNA_34_SHA256,
              "source_manifest_sha256": sha256(source), "recipe": config,
              "whole_history_extraction": True, "window_size": [96, 0], "records": inventory,
              "counts": counts, "original_native_unchanged": True, "gpu_requested": False,
              "local_trial": maximum_games is not None, "elapsed_seconds": time.monotonic() - started,
              "feature_sources": {name: sha256(Path(__file__).with_name(name)) for name in (
                  "kakuna.py", "attention.py", "retention.py", "latent_cache.py")}}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
