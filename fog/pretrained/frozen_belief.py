"""Frozen-Kakuna belief data, residual heads and resumable matched updates.

There is no native policy module or native optimizer in this training path.
The exact immutable native checkpoint is an input identity, not an output to
overwrite. Complete team sheets remain separate from public retrieval inputs.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fog.constants.tables import TABLES
from fog.encoding.reveals import masks_for_tokens
from fog.pretrained.joint import BeliefHeads
from fog.pretrained.state import atomic_save

DATA_FORMAT = "fog.frozen-native-belief-data"
DATA_VERSION = 2
HEAD_FORMAT = "fog.prior-belief-heads"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_data(source, out, *, pools):
    """Drop unused lower features, preserving every audited latent and label."""
    from fog.pretrained.latent_training import FeatureData

    source, out = Path(source), Path(out)
    data = FeatureData(source, digest(source / "manifest.json"))
    from fog.team_identity import team_group_key

    roster_keys = {}
    for split in ("train", "development"):
        teams = json.loads((Path(pools) / (split + ".json")).read_text())["teams"]
        for team in teams:
            roster_keys[team_group_key(team)] = team_group_key(team, mode="roster")
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for row in data.manifest["records"]:
        record = data.record(row["path"])
        value = {"latent": record["reference_latent"], "labels": record["labels"]}
        atomic_save(value, out / row["path"])
        records.append(
            {
                **row,
                "source_shard_sha256": row["sha256"],
                "sha256": digest(out / row["path"]),
                "roster_groups": sorted({roster_keys[k] for k in row["compositions"]}),
            }
        )
    manifest = {
        "format": DATA_FORMAT,
        "version": DATA_VERSION,
        "complete": True,
        "native_sha256": data.manifest["checkpoint_sha256"],
        "native_dim": 900,
        "source_cache_sha256": data.signature,
        "feature_sources": data.manifest["feature_sources"],
        "complete_chronological_history": True,
        "both_perspectives": True,
        "counts": data.manifest["counts"],
        "records": records,
        "private_labels_separate_from_latents": True,
        "local_trial": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class FrozenBeliefData:
    def __init__(
        self,
        folder,
        expected_sha256,
        *,
        allow_trial=False,
        required_splits=("train", "development"),
    ):
        self.folder = Path(folder).resolve()
        if digest(self.folder / "manifest.json") != expected_sha256:
            raise ValueError("Frozen belief-data manifest changed")
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        m = self.manifest
        if (
            m["format"] != DATA_FORMAT
            or m["version"] != DATA_VERSION
            or not m["complete"]
            or not m["complete_chronological_history"]
            or not m["both_perspectives"]
            or not m["private_labels_separate_from_latents"]
            or m["native_dim"] != 900
        ):
            raise ValueError("Unverified frozen latent/label data")
        if m.get("local_trial", False) and not allow_trial:
            raise ValueError("Local trial data cannot enter a production run")
        if not required_splits or any(s not in ("train", "development") for s in required_splits):
            raise ValueError("Unknown/empty frozen-data splits")
        self.rows = {
            split: [r for r in m["records"] if r["split"] == split] for split in required_splits
        }
        for split, rows in self.rows.items():
            if not rows or sum(r["positions"] for r in rows) != m["counts"][split]["positions"]:
                raise ValueError("Frozen-data position counts disagree")
            games = {r["game"] for r in rows}
            if (
                len(rows) != m["counts"][split]["perspectives"]
                or len(games) != m["counts"][split]["games"]
            ):
                raise ValueError("Frozen-data game/perspective counts disagree")
            if any(
                sorted(r["pov_seed"] for r in rows if r["game"] == game) != [0, 1] for game in games
            ):
                raise ValueError("Both complete perspectives are required")
        compositions = [
            {g for r in rows for g in r["roster_groups"]} for rows in self.rows.values()
        ]
        if len(compositions) == 2 and compositions[0] & compositions[1]:
            raise ValueError("Frozen training/development rosters overlap")
        if len({r["path"] for r in m["records"]}) != len(m["records"]):
            raise ValueError("Duplicate frozen perspective")
        for row in m["records"]:
            path = (self.folder / row["path"]).resolve()
            if not path.is_relative_to(self.folder) or digest(path) != row["sha256"]:
                raise ValueError("Unsafe or changed frozen perspective")

    @lru_cache(maxsize=64)
    def record(self, name):
        record = torch.load(self.folder / name, map_location="cpu", weights_only=True)
        if set(record) != {"latent", "labels"} or not torch.isfinite(record["latent"]).all():
            raise ValueError("Unexpected/nonfinite frozen inference inputs")
        if record["latent"].ndim != 2 or record["latent"].shape[1] != 900:
            raise ValueError("Wrong native latent shape")
        if any(len(v) != len(record["latent"]) for v in record["labels"].values()):
            raise ValueError("Labels/latents do not share a decision timeline")
        return record

    def draw(self, rng, count, *, split="train"):
        result = []
        for _ in range(count):
            row = rng.choice(self.rows[split])
            result.append((row["path"], rng.randrange(row["positions"])))
        return result

    def batch(self, specifications, *, device="cpu"):
        rows = [(self.record(name), index) for name, index in specifications]
        return {
            "latent": torch.stack([r["latent"][i] for r, i in rows]).to(device),
            "labels": {
                key: torch.stack([r["labels"][key][i] for r, i in rows]).to(device)
                for key in rows[0][0]["labels"]
            },
        }


def public_payload(model, reveal_tokens, prior, *, device=None):
    """No truth/verified input is accepted by retrieval or legality construction."""
    device = reveal_tokens.device if device is None else device
    tokens = reveal_tokens.detach().cpu()
    known = model.belief._revealed_mask(tokens).reshape_as(tokens)
    public = torch.where(known, tokens, -1).numpy()
    rows = [masks_for_tokens(row.numpy(), TABLES) for row in tokens]
    masks = {
        key: torch.as_tensor(np.stack([r[key] for r in rows]), device=device) for key in rows[0]
    }
    return {
        "tokens": reveal_tokens.to(device),
        "masks": masks,
        "prior": prior.prepare(public, device=device),
    }


def subset_payload(payload, start, end):
    return {
        "tokens": payload["tokens"][start:end],
        "masks": {key: value[start:end] for key, value in payload["masks"].items()},
        "prior": payload["prior"].index_select(slice(start, end)),
    }


def initialize_residual(path, *, expected_sha256, native_sha256, device="cpu"):
    if digest(path) != expected_sha256:
        raise ValueError("Warm feature-head checkpoint changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["native_sha256"] != native_sha256:
        raise ValueError("Warm feature heads belong to another native checkpoint")
    model = BeliefHeads()
    model.load_state_dict(state["heads"], strict=True)
    # Retain learned fact encoders, but start as the prior-only conditional.
    for head in [*model.belief.heads.values(), *model.belief.ev_stat_heads]:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    return model.to(device)


def update(model, optimizer, batch, payload, generator, *, mode, chain_queries=8, microbatch=8):
    if mode not in ("facts", "native"):
        raise ValueError("Unknown frozen-feature ablation")
    if len(batch["latent"]) % microbatch:
        raise ValueError("Microbatch must divide the effective context count")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = dict(marginal=0.0, autoregressive=0.0, outcome=0.0)
    for start in range(0, len(batch["latent"]), microbatch):
        end = start + microbatch
        latent = batch["latent"][start:end]
        labels = {key: value[start:end] for key, value in batch["labels"].items()}
        part = subset_payload(payload, start, end)
        belief_latent = latent if mode == "native" else torch.zeros_like(latent)
        # Outcome heads retain real frozen features in both arms. Zeroing the
        # entire input would confound the belief-context ablation with value data.
        outcome = model.outcome(latent).squeeze(-1)
        valid = labels["outcome_valid"].bool() & torch.isfinite(labels["outcomes"])
        outcome_loss = (
            (outcome[valid].tanh() - labels["outcomes"][valid]).square().mean()
            if bool(valid.any())
            else outcome[:0].sum()
        )
        loss = model.auxiliary_losses(
            belief_latent,
            labels["reveals"],
            labels["truth"],
            labels["verified"],
            labels["outcomes"],
            torch.zeros_like(labels["outcome_valid"]),
            masks=part["masks"],
            prior=part["prior"],
            generator=generator,
            chain_queries=chain_queries,
            selected_chain=True,
        )
        loss["outcome"] = outcome_loss
        scale = microbatch / len(batch["latent"])
        total = loss["marginal"] + loss["autoregressive"] + 0.1 * loss["outcome"]
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite frozen belief objective")
        (total * scale).backward()
        for key, value in loss.items():
            totals[key] += float(value.detach()) * scale
    belief_parameters = [*model.context_projection.parameters(), *model.belief.parameters()]
    # The outcome task is independent while the backbone is frozen. A joint
    # clipping norm would couple its updates to the belief-context ablation.
    norm = float(torch.nn.utils.clip_grad_norm_(belief_parameters, 1.5, error_if_nonfinite=True))
    outcome_norm = float(
        torch.nn.utils.clip_grad_norm_(model.outcome.parameters(), 1.5, error_if_nonfinite=True)
    )
    optimizer.step()
    labels = batch["labels"]
    valid = labels["verified"] & (labels["truth"] >= 0)
    valid &= labels["truth"] < model.belief.pos_n.reshape(6, 15)
    valid &= ~model.belief._revealed_mask(labels["reveals"]).reshape_as(valid)
    return {
        "losses": totals,
        "gradient_norm": norm,
        "outcome_gradient_norm": outcome_norm,
        "context_mode": mode,
        "AR_queries": int(valid.flatten(1).any(1).sum()) * chain_queries,
        "AR_query_draws": len(batch["latent"]) * chain_queries,
        "native_optimizer_present": False,
    }


def export_heads(model, path, *, config, mode, update, identity):
    atomic_save(
        {
            "format": HEAD_FORMAT,
            "version": 1,
            "identity": identity,
            "native_sha256": config["native_sha256"],
            "prior_sha256": config["prior_sha256"],
            "mode": mode,
            "update": update,
            "config": config,
            "heads": model.state_dict(),
            "native_weights_included": False,
            "native_policy_changed": False,
        },
        Path(path),
    )


def load_heads(model, path, *, native_sha256, prior_sha256):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state["format"] != HEAD_FORMAT
        or state["version"] != 1
        or state["native_sha256"] != native_sha256
        or state["prior_sha256"] != prior_sha256
        or state["native_weights_included"]
        or state["native_policy_changed"]
    ):
        raise ValueError("Prior/native identity or head checkpoint contract differs")
    heads = {
        key: value for key, value in model.state_dict().items() if not key.startswith("native.")
    }
    if heads.keys() != state["heads"].keys():
        raise ValueError("Prior head architecture changed")
    current = model.state_dict()
    current.update(state["heads"])
    model.load_state_dict(current, strict=True)
    return state
