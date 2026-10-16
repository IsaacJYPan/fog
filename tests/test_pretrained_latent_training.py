import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fog.pretrained.kakuna import INTERFACE_VERSION, KAKUNA_34_SHA256, sha256
from fog.pretrained.latent_cache import CACHE_FORMAT, CACHE_VERSION
from fog.pretrained.latent_training import HEAD_SCHEMA_FILES, FeatureData, head_parameters, head_state, initialize_heads, train_update


def cache(tmp_path):
    root = Path(__file__).resolve().parents[1]
    records = []
    for split in ("train", "development"):
        for pov in (0, 1):
            path = tmp_path / f"{split}-{pov}.pt"
            payload = {"features": torch.randn(3, 8), "positions": torch.arange(3),
                       "reference_latent": torch.randn(3, 8), "reference_probabilities": torch.ones(3, 13) / 13,
                       "reference_values": torch.zeros(3, 13),
                       "actor_observations": {"illegal_actions": torch.zeros(3, 13, dtype=torch.bool)},
                       "labels": {"truth": torch.full((3, 6, 15), -1), "verified": torch.zeros(3, 6, 15, dtype=torch.bool)}}
            torch.save(payload, path)
            records.append({"path": path.name, "sha256": sha256(path), "split": split,
                            "game": split + "-game", "pov_seed": pov, "positions": 3, "compositions": [split + "-team"]})
    manifest = {"format": CACHE_FORMAT, "version": CACHE_VERSION, "complete": True, "interface": INTERFACE_VERSION,
                "checkpoint_sha256": KAKUNA_34_SHA256, "whole_history_extraction": True,
                "original_native_unchanged": True, "window_size": [96, 0], "local_trial": True,
                "records": records, "counts": {split: {"games": 1, "perspectives": 2, "positions": 6} for split in ("train", "development")},
                "feature_sources": {name: sha256(root / "fog/pretrained" / name) for name in ("kakuna.py", "attention.py", "retention.py", "latent_cache.py")}}
    return manifest


def load(tmp_path, manifest, **kwargs):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return FeatureData(tmp_path, sha256(path), **kwargs)


def test_verified_feature_batches_have_separate_nonaliasing_labels(tmp_path):
    data = load(tmp_path, cache(tmp_path), allow_trial=True)
    batch = data.batch([("train-0.pt", 2), ("train-1.pt", 0)], "cpu")
    before = batch["features"].clone()
    batch["labels"]["truth"].fill_(999999)
    assert torch.equal(before, batch["features"])
    assert data.record("train-0.pt")["labels"]["truth"].max() == -1
    assert batch["lengths"].tolist() == [3, 1]


def test_incomplete_source_coverage_and_trial_cache_cannot_be_promoted(tmp_path):
    m = cache(tmp_path)
    with pytest.raises(ValueError, match="contract"):
        load(tmp_path, m)
    m["records"].pop()
    with pytest.raises(ValueError, match="coverage"):
        load(tmp_path, m, allow_trial=True)


def test_feature_source_changes_and_changed_shards_fail_closed(tmp_path):
    m = cache(tmp_path)
    path = tmp_path / m["records"][0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load(tmp_path, m, allow_trial=True)


def small_model():
    m = nn.Module()
    m.native = nn.Linear(2, 2)
    m.context_projection = nn.Linear(2, 2)
    m.belief = nn.Linear(2, 2)
    m.outcome = nn.Linear(2, 1)
    return m


def test_legacy_decoder_transfer_requires_matching_schema_and_preserves_native(tmp_path):
    model = small_model()
    path = tmp_path / "heads.pt"
    state = {"version": 1, "interface": "kakuna34-metamon-default13-window96-fp32-legal-v3",
             "native_sha256": "a" * 64, "heads": head_state(model), "identity": "source", "update": 1800}
    torch.save(state, path)
    kwargs = dict(expected_sha256=sha256(path), paired_native_sha256="a" * 64, source_interface=state["interface"])
    with pytest.raises(ValueError, match="schemas"):
        initialize_heads(model, path, **kwargs)
    root = Path(__file__).resolve().parents[1]
    schema = {n: sha256(root / n) for n in HEAD_SCHEMA_FILES}
    before = {k: v.clone() for k, v in model.native.state_dict().items()}
    proof = initialize_heads(model, path, source_schema=schema, **kwargs)
    assert proof["frozen_original_rebase_required"] and not proof["source_native_weights_loaded"]
    assert all(torch.equal(v, before[k]) for k, v in model.native.state_dict().items())


def test_nonfinite_native_proposal_restores_adam_and_weights_but_keeps_head_step(monkeypatch):
    from fog.pretrained import latent_training as module

    torch.manual_seed(3)
    m = small_model()
    h = torch.ones(2, 2)
    monkeypatch.setattr(module, "latent_for", lambda model, batch, adapt: model.native(batch["reference_latent"]))

    def losses(model, latent, labels, generator):
        context = model.context_projection(latent)
        return {"marginal": (model.belief(context) - 3).square().mean(),
                "autoregressive": model.belief(context).square().mean(),
                "outcome": model.outcome(context).square().mean()}

    monkeypatch.setattr(module, "belief_losses", losses)
    monkeypatch.setattr(module, "legal_policy", lambda model, latent, obs: latent.sum(-1, keepdim=True).expand(*latent.shape[:2], 13).softmax(-1))
    monkeypatch.setattr(module, "action_values", lambda model, latent: latent.sum(-1, keepdim=True).expand(*latent.shape[:2], 13))
    calls = []

    def assess(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FloatingPointError("Candidate critic overflow")
        return dict(policy_kl_mean=0., policy_kl_max=0., argmax_disagreement=0., value_relative_mean=0., value_relative_max=0.)

    monkeypatch.setattr(module, "assess_behavior", assess)
    head_optimizer = torch.optim.Adam(head_parameters(m), lr=.01)
    native_optimizer = torch.optim.Adam(m.native.parameters(), lr=.01)
    before = {k: v.clone() for k, v in m.native.state_dict().items()}
    batch = {"reference_latent": h, "reference_probabilities": torch.ones(2, 13) / 13,
             "reference_values": m.native(h).sum(-1, keepdim=True).expand(2, 13).detach(),
             "lengths": torch.ones(2, dtype=torch.long), "actor_observations": {"illegal_actions": torch.zeros(2, 13, dtype=torch.bool)},
             "labels": {"ignored": torch.ones(2)}}
    result = train_update(m, batch, head_optimizer=head_optimizer, native_optimizer=native_optimizer,
                          route_generator=None, weights=dict(marginal=1., autoregressive=1., outcome=.1), microbatch=2)
    assert not result["native_update_accepted"] and "overflow" in result["native_rejection_error"]
    assert all(torch.equal(v, before[k]) for k, v in m.native.state_dict().items())
    assert not native_optimizer.state and head_optimizer.state
    assert result["head_optimizer_steps"] == 1 and result["native_optimizer_steps"] == 0
