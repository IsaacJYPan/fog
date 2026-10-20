"""Belief-first learning on verified native features with measured behavior limits."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fog.pretrained.joint import normalized_belief_loss
from fog.pretrained.kakuna import INTERFACE_VERSION, KAKUNA_34_SHA256, sha256
from fog.pretrained.latent_cache import CACHE_FORMAT, CACHE_VERSION, windows
from fog.pretrained.gradient_balance import OutcomeGradientCap
from fog.pretrained.retention import NativeStep, RetentionLimits, action_values, legal_policy, retention_terms, upper_forward

HEAD_SCHEMA_FILES = ("fog/models/belief.py", "fog/constants/tables.py", "fog/constants/schema.py") + tuple(
    "fog/constants/generated/" + name + ".json" for name in ("abilities", "items", "moves", "priors", "species", "spreads")
)

class FeatureData:
    def __init__(self, folder, expected_sha256, *, allow_trial=False):
        self.folder = Path(folder).resolve()
        manifest = self.folder / "manifest.json"
        if sha256(manifest) != expected_sha256:
            raise ValueError("Feature cache manifest changed")
        self.manifest = json.loads(manifest.read_text())
        m = self.manifest
        if (m.get("format") != CACHE_FORMAT or m.get("version") != CACHE_VERSION or not m.get("complete")
                or m.get("interface") != INTERFACE_VERSION or m.get("checkpoint_sha256") != KAKUNA_34_SHA256
                or not m.get("whole_history_extraction") or not m.get("original_native_unchanged")
                or m.get("window_size") != [96, 0] or (m.get("local_trial") and not allow_trial)):
            raise ValueError("Cache does not satisfy the fixed native feature contract")
        for name, expected in m["feature_sources"].items():
            if Path(name).name != name or sha256(Path(__file__).with_name(name)) != expected:
                raise ValueError("Feature implementation differs from the extracted cache: " + name)
        self.rows = {split: [row for row in m["records"] if row["split"] == split] for split in ("train", "development")}
        if any(not rows for rows in self.rows.values()):
            raise ValueError("Both cache splits must be nonempty")
        if len({r["path"] for r in m["records"]}) != len(m["records"]):
            raise ValueError("Duplicate cache perspective")
        for split, rows in self.rows.items():
            games = {row["game"] for row in rows}
            counts = m["counts"][split]
            if (len(rows) != counts["perspectives"] or len(games) != counts["games"]
                    or sum(row["positions"] for row in rows) != counts["positions"]
                    or any(sorted(row["pov_seed"] for row in rows if row["game"] == game) != [0, 1] for game in games)):
                raise ValueError("Cache counts or both-perspective coverage disagree")
        if not allow_trial and (m["counts"]["train"]["games"] < 999 or m["counts"]["development"]["games"] < 32):
            raise ValueError("Production cache lacks the verified minimum source games")
        for row in m["records"]:
            path = (self.folder / row["path"]).resolve()
            if not path.is_relative_to(self.folder) or sha256(path) != row["sha256"]:
                raise ValueError("Unsafe or changed feature shard")
        groups = [{group for row in self.rows[split] for group in row["compositions"]} for split in self.rows]
        if groups[0] & groups[1]:
            raise ValueError("Cached team compositions overlap")
        self.signature = expected_sha256

    @lru_cache(maxsize=32)
    def record(self, name):
        return torch.load(self.folder / name, map_location="cpu", weights_only=True)

    def draw(self, rng, count, *, split="train"):
        result = []
        for _ in range(count):
            row = rng.choice(self.rows[split])
            result.append((row["path"], rng.randrange(row["positions"])))
        return result

    def batch(self, specifications, device):
        rows = [(self.record(name), index) for name, index in specifications]
        packed = [windows(record, [index]) for record, index in rows]
        output = {
            "features": pad_sequence([p[0][0] for p in packed], batch_first=True).to(device),
            "positions": pad_sequence([p[1][0] for p in packed], batch_first=True).to(device),
            "lengths": torch.cat([p[2] for p in packed]).to(device),
        }
        for key in ("reference_latent", "reference_probabilities", "reference_values"):
            output[key] = torch.stack([r[key][i] for r, i in rows]).to(device)
        for section in ("actor_observations", "labels"):
            output[section] = {key: torch.stack([r[section][key][i] for r, i in rows]).to(device) for key in rows[0][0][section]}
        return output


def select_batch(batch, indices):
    return {key: select_batch(value, indices) if isinstance(value, dict) else value[indices] for key, value in batch.items()}


def head_parameters(model):
    return list(model.context_projection.parameters()) + list(model.belief.parameters()) + list(model.outcome.parameters())


def head_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if not key.startswith("native.")}


def load_heads(model, state):
    if set(state) != {key for key in model.state_dict() if not key.startswith("native.")}:
        raise ValueError("Auxiliary head schema changed")
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def initialize_heads(model, path, *, expected_sha256, paired_native_sha256,
                     source_interface=INTERFACE_VERSION, source_schema=None):
    if sha256(Path(path)) != expected_sha256:
        raise ValueError("Auxiliary initialization bytes changed")
    source = torch.load(path, weights_only=True, map_location="cpu")
    compatible = {INTERFACE_VERSION, "kakuna34-metamon-default13-window96-fp32-legal-v3"}
    if (source.get("version") != 1 or source["interface"] != source_interface
            or source_interface not in compatible or source["native_sha256"] != paired_native_sha256):
        raise ValueError("Auxiliary initialization lacks its declared native pairing")
    if source_interface != INTERFACE_VERSION:
        required = set(HEAD_SCHEMA_FILES)
        root = Path(__file__).resolve().parents[2]
        if not source_schema or set(source_schema) != required or any(sha256(root / name) != digest for name, digest in source_schema.items()):
            raise ValueError("Legacy head transfer requires identical decoder and token schemas")
    load_heads(model, source["heads"])
    return {"auxiliary_sha256": expected_sha256, "source_native_sha256": paired_native_sha256,
            "source_identity": source["identity"], "source_update": source["update"],
            "source_interface": source_interface, "source_schema": source_schema,
            "source_native_weights_loaded": False,
            "new_native_initialization": KAKUNA_34_SHA256, "frozen_original_rebase_required": True}


def latent_for(model, batch, *, adapt):
    if not adapt:
        return batch["reference_latent"]
    return upper_forward(model.native, batch["features"], batch["positions"], batch["lengths"])


def behavior(model, batch, *, adapt):
    latent = latent_for(model, batch, adapt=adapt)
    obs = {key: value[:, None] for key, value in batch["actor_observations"].items()}
    p = legal_policy(model.native, latent[:, None], obs)[:, 0]
    q = action_values(model.native, latent[:, None])[:, 0]
    return retention_terms(p, batch["reference_probabilities"], q, batch["reference_values"], ~obs["illegal_actions"][:, 0].bool())


@torch.no_grad()
def assess_behavior(model, batch, *, adapt, microbatch=8):
    probabilities, values = [], []
    for start in range(0, len(batch["lengths"]), microbatch):
        row = select_batch(batch, slice(start, start + microbatch))
        latent = latent_for(model, row, adapt=adapt)
        obs = {key: value[:, None] for key, value in row["actor_observations"].items()}
        probabilities.append(legal_policy(model.native, latent[:, None], obs)[:, 0])
        values.append(action_values(model.native, latent[:, None])[:, 0])
    return retention_terms(torch.cat(probabilities), batch["reference_probabilities"], torch.cat(values),
                           batch["reference_values"], ~batch["actor_observations"]["illegal_actions"].bool())[2]


def belief_losses(model, latent, labels, generator):
    from fog.constants.tables import TABLES
    from fog.encoding.reveals import masks_for_tokens

    reveals = labels["reveals"]
    masks = [masks_for_tokens(row, TABLES) for row in reveals.detach().cpu().numpy()]
    masks = {key: torch.as_tensor(np.stack([row[key] for row in masks]), device=latent.device) for key in masks[0]}
    return model.auxiliary_losses(latent, reveals, labels["truth"], labels["verified"],
                                  labels["outcomes"], labels["outcome_valid"], masks=masks, generator=generator)


def train_update(model, batch, *, head_optimizer, native_optimizer, route_generator,
                 weights, policy_penalty=20.0, value_penalty=1.0, limits=None,
                 sentinel=None, microbatch=8, outcome_backbone_ratio=None):
    adapt = native_optimizer is not None
    model.native.eval()
    for module in (model.context_projection, model.belief, model.outcome):
        module.train()
    head_optimizer.zero_grad(set_to_none=True)
    if adapt:
        native_optimizer.zero_grad(set_to_none=True)
    native_parameters = [] if not adapt else [p for group in native_optimizer.param_groups for p in group["params"]]
    balance = OutcomeGradientCap(native_parameters, outcome_backbone_ratio) if adapt and outcome_backbone_ratio is not None else None
    size = len(batch["lengths"])
    totals = {key: 0.0 for key in ("marginal", "autoregressive", "outcome", "policy_kl", "value_retention")}
    for start in range(0, size, microbatch):
        row = select_batch(batch, slice(start, start + microbatch))
        latent = latent_for(model, row, adapt=adapt)
        losses = belief_losses(model, latent, row["labels"], route_generator)
        if adapt:
            # Reuse this latent for the readouts; no labels enter either readout.
            obs = {key: value[:, None] for key, value in row["actor_observations"].items()}
            p = legal_policy(model.native, latent[:, None], obs)[:, 0]
            q = action_values(model.native, latent[:, None])[:, 0]
            kl, value, _ = retention_terms(p, row["reference_probabilities"], q, row["reference_values"], ~obs["illegal_actions"][:, 0].bool())
        else:
            kl = value = latent.sum() * 0
        loss = sum(weights[key] * term for key, term in losses.items()) + policy_penalty * kl + value_penalty * value
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite belief/retention objective")
        scale = len(row["lengths"]) / size
        if balance is not None:
            belief = weights["marginal"] * losses["marginal"] + weights["autoregressive"] * losses["autoregressive"]
            balance.accumulate(belief * scale, weights["outcome"] * losses["outcome"] * scale)
        (loss * scale).backward()
        for key, term in {**losses, "policy_kl": kl, "value_retention": value}.items():
            totals[key] += float(term.detach()) * scale
    heads = head_parameters(model)
    balance_metrics = balance.apply() if balance is not None else {}
    head_norm = float(torch.nn.utils.clip_grad_norm_(heads, 1.5, error_if_nonfinite=True))
    native_norm = float(torch.nn.utils.clip_grad_norm_(native_parameters, 1.0, error_if_nonfinite=True)) if adapt else 0.0
    step = NativeStep(native_parameters, native_optimizer) if adapt else None
    head_optimizer.step()
    restored, error = False, None
    try:
        if adapt:
            native_optimizer.step()
        current = assess_behavior(model, batch, adapt=adapt, microbatch=microbatch)
        fixed = assess_behavior(model, sentinel, adapt=adapt, microbatch=microbatch) if sentinel is not None else None
        accepted = not adapt or ((limits or RetentionLimits()).accepts(current) and (fixed is None or (limits or RetentionLimits()).accepts(fixed)))
    except FloatingPointError as failure:
        if not adapt:
            raise
        step.restore()
        restored = True
        error = str(failure)
        current, fixed, accepted = {}, None, False
    except BaseException:
        if adapt:
            step.restore()
        raise
    if not accepted:
        if not restored:
            step.restore()
        retained = assess_behavior(model, batch, adapt=adapt, microbatch=microbatch)
    else:
        retained = current
    return {"losses": totals, "head_gradient_norm": head_norm, "native_gradient_norm": native_norm,
            **balance_metrics,
            "native_update_attempted": adapt, "native_update_accepted": bool(adapt and accepted),
            "attempted_batch_retention": current, "attempted_sentinel_retention": fixed,
            "retained_batch_retention": retained,
            "retained_batch_within_limits": (limits or RetentionLimits()).accepts(retained),
            "native_rejection_error": error,
            "head_optimizer_steps": 1, "native_optimizer_steps": int(adapt and accepted)}


def query_schedule(specifications):
    return hashlib.sha256(json.dumps(specifications, separators=(",", ":")).encode()).hexdigest()


def fixed_queries(data, *, seed, count, split="train"):
    return data.draw(random.Random(seed), count, split=split)


@torch.no_grad()
def score_beliefs(model, data, specifications, *, adapt, device="cpu", seed=0):
    """Same exact-prefix query RNG for both arms; direct metrics remain separate."""
    from fog.constants.tables import TABLES
    from fog.encoding.reveals import masks_for_tokens
    from fog.models.belief import BeliefModel
    from fog.pretrained.belief_diagnostics import DistributionMetrics

    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    marginal, conditional = DistributionMetrics(), DistributionMetrics()
    outcome_errors, latents = [], []
    for specification in specifications:
        batch = data.batch([specification], device)
        latent = latent_for(model, batch, adapt=adapt)
        labels = batch["labels"]
        public, verified = labels["reveals"], labels["verified"]
        truth = torch.where(verified, labels["truth"], -1)
        mask = {key: torch.from_numpy(value[None]).to(device) for key, value in masks_for_tokens(public[0].cpu().numpy(), TABLES).items()}
        payload = {"tokens": public, "masks": mask}
        context = model.context_projection(latent)
        direct = model.belief(context, payload)
        hidden = verified & ~model.belief._revealed_mask(public).reshape_as(verified)
        marginal.update(direct.logits, BeliefModel._split_flat(truth), BeliefModel._split_flat(hidden))
        chain = model.belief(context, payload, teacher_values=truth, generator=generator)
        valid = {key: value & (chain.loss_targets[key] >= 0) for key, value in chain.loss_valid.items()}
        conditional.update(chain.logits, chain.loss_targets, valid)
        loss = normalized_belief_loss(chain, truth, verified)
        if not torch.isfinite(loss):
            raise FloatingPointError("Impossible verified conditional target")
        outcome_errors.append(float((model.outcome(latent).squeeze(-1).tanh() - labels["outcomes"]).square().mean()))
        latents.append(float((latent - batch["reference_latent"]).square().mean()))
    return {"contexts": len(specifications), "schedule_sha256": query_schedule(specifications),
            "public_marginal": marginal.report(), "true_prefix_chain": conditional.report(),
            "outcome_mse": sum(outcome_errors) / len(outcome_errors),
            "latent_mse_from_original": sum(latents) / len(latents), "ancestral_sampling_evaluated": False}
