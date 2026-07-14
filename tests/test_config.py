"""Tests for config loading + hashing + `extends` resolution (spec §0 reproducibility)."""
from __future__ import annotations

from fog.config import config_hash, load_config


def test_default_loads():
    cfg = load_config("default.yaml")
    assert cfg["encoding"]["state_dim"] == 256
    assert cfg["model"]["aux_into_trunk"] is True


def test_extends_merges_and_overrides():
    cfg = load_config("m1_poc.yaml")
    # inherited from default
    assert cfg["encoding"]["state_dim"] == 256
    # overridden in m1 (belief net off at the PoC baseline)
    assert cfg["loss"]["belief_coef"] == 0.0
    assert cfg["env"]["battle_format"] == "gen9ou"
    assert cfg["env"]["team_pool"] == "data/teams/pool_a"


def test_config_hash_stable_and_sensitive():
    a = load_config("m1_poc.yaml")
    b = load_config("m1_poc.yaml")
    assert config_hash(a) == config_hash(b)
    b["seed"] = 999
    assert config_hash(a) != config_hash(b)
