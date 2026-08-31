"""Tests for config loading + hashing + `extends` resolution (spec §0 reproducibility)."""
from __future__ import annotations

import pytest

from fog.config import apply_overrides, config_hash, load_config, load_config_with_args


def test_default_loads():
    cfg = load_config("default.yaml")
    assert cfg["encoding"]["state_dim"] == 256
    assert cfg["model"]["aux_into_trunk"] is True


def test_extends_merges_and_overrides():
    cfg = load_config("archive/m1_poc.yaml")
    # inherited from default
    assert cfg["encoding"]["state_dim"] == 256
    # overridden in m1 (belief net off at the PoC baseline)
    assert cfg["loss"]["belief_coef"] == 0.0
    assert cfg["env"]["battle_format"] == "gen9ou"
    assert cfg["env"]["team_pool"] == "data/teams/pool_a"


def test_config_hash_stable_and_sensitive():
    a = load_config("archive/m1_poc.yaml")
    b = load_config("archive/m1_poc.yaml")
    assert config_hash(a) == config_hash(b)
    b["seed"] = 999
    assert config_hash(a) != config_hash(b)


def test_extends_cycle_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(f"extends: {second}\n", encoding="utf-8")
    second.write_text(f"extends: {first}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extends cycle"):
        load_config(first)


def test_relative_extends_resolves_beside_custom_config(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("seed: 7\nmodel:\n  belief_dim: 32\n", encoding="utf-8")
    child.write_text("extends: base.yaml\nmodel:\n  belief_dim: 16\n", encoding="utf-8")

    assert load_config(child) == {"seed": 7, "model": {"belief_dim": 16}}


@pytest.mark.parametrize("override", ["ppo.lr", "=1", "ppo..lr=1"])
def test_malformed_override_is_rejected(override):
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, [override])


def test_override_cannot_descend_through_scalar():
    with pytest.raises(TypeError, match="non-mapping"):
        apply_overrides({"ppo": 1}, ["ppo.lr=0.1"])


def test_public_ou_selfplay_uses_collected_package():
    cfg = load_config("train_ou_public.yaml")
    assert cfg["env"]["battle_format"] == "gen9ou"
    assert cfg["env"]["team_pool"] == "data/teams/public_gen9ou/pool.json"
    assert cfg["env"]["min_team_pool_size"] == 100
    assert cfg["log"]["dir"] == "runs/train_ou_public"


def test_unknown_root_config_key_is_rejected_with_suggestion(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("sead: 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did you mean 'seed'"):
        load_config(path)


def test_unknown_nested_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("ppo:\n  learning_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key ppo.learning_rate"):
        load_config(path)


def test_cli_override_cannot_create_unknown_key(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("ppo:\n  lr: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key ppo.lrr"):
        load_config_with_args(path, ["ppo.lrr=0.2"])
