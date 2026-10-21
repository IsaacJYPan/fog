"""Value-checkpoint expansion, posterior-feature regression, and logged training data."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.env.engine_battle import HAVE_ENGINE
from fog.models.agent import FogAgent
from fog.models.heads import N_POSTERIOR_FEATURES
from fog.training.value_finetune import (
    _split_by_game,
    expand_state_dict_for_posterior,
    finetune_value,
    fit_value_mlp,
    load_posterior_agent,
)
from tests.test_joint_search import P_ME, P_ME_VARIANT, P_OPP

AWARE_CFG = {"model": {"use_posterior_features": True}}


def _batch(b=3, seed=0, n_species=300):
    torch.manual_seed(seed)
    tokens = torch.full((b, 6, 15), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, n_species, (b, 6))
    legal = torch.zeros(b, 14, dtype=torch.bool)
    legal[:, [0, 1, 8]] = True
    return {
        "entities": torch.randn(b, 12, ENTITY_RAW_DIM),
        "entity_ids": torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.randn(b, DIMS.field_dim),
        "history": torch.randn(b, 4, DIMS.turn_dim),
        "active_idx": torch.tensor([[0, 6]] * b),
        "legality_mask": legal,
        "opp_revealed": {"tokens": tokens, "masks": None},
    }


# -------------------------------------------------------------------------- checkpoint expansion
def test_expand_conditional_head_preserves_action_scores():
    from fog.models.heads import PolicyValueHead
    torch.manual_seed(42)
    kwargs = dict(state_dim=8, dyn_dim=6, action_feature_dim=3, action_context_dim=5,
                  structured_dim=2)
    blind = PolicyValueHead(**kwargs).eval()
    aware = PolicyValueHead(**kwargs, posterior_dim=N_POSTERIOR_FEATURES).eval()
    torch.nn.init.normal_(blind.conditional_action_scorer[-1].weight)
    expanded = expand_state_dict_for_posterior(
        {"policy_value." + k: v for k, v in blind.state_dict().items()})
    aware.load_state_dict({k.removeprefix("policy_value."): v for k, v in expanded.items()})
    features = torch.randn(4, 24)  # 14 core + 2 structured + 5 context + 3 action
    padded = torch.cat([features[:, :14], torch.zeros(4, N_POSTERIOR_FEATURES),
                        features[:, 14:]], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(blind.conditional_action_scorer(features),
                                   aware.conditional_action_scorer(padded))


def test_expand_blind_state_dict_preserves_behavior():
    """Zero-padded posterior columns must leave the head byte-equivalent at zero/absent features —
    the expanded checkpoint plays exactly like the blind one until the fine-tune trains it."""
    torch.manual_seed(1)
    blind = FogAgent({}).eval()
    aware = FogAgent(AWARE_CFG).eval()
    aware.load_state_dict(expand_state_dict_for_posterior(blind.state_dict()))

    batch = _batch()
    with torch.no_grad():
        out_b = blind(batch)
        out_a = aware(batch)  # posterior features absent -> zeros
    torch.testing.assert_close(out_a.log_probs, out_b.log_probs)
    torch.testing.assert_close(out_a.value, out_b.value)


def test_load_posterior_agent_expands_and_rejects(tmp_path):
    blind = FogAgent.from_tables({}).eval()
    path = tmp_path / "blind.pt"
    torch.save(blind.state_dict(), path)
    agent = load_posterior_agent({}, path, torch.device("cpu")).eval()
    assert agent.policy_value.posterior_dim == N_POSTERIOR_FEATURES

    n_species = blind.sizes["n_species"]
    batch = _batch(n_species=n_species)
    with torch.no_grad():
        out_b = blind(batch)
        out_a = agent(batch)
    torch.testing.assert_close(out_a.value, out_b.value)

    # a head width that is neither blind nor aware is a config mismatch, not expandable
    sd = blind.state_dict()
    sd["policy_value.policy.0.weight"] = sd["policy_value.policy.0.weight"][:, :-2]
    torch.save(sd, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_posterior_agent({}, path, torch.device("cpu"))


# ------------------------------------------------------------------------------- fine-tune core
def test_fit_gate_prefers_posterior_when_signal_exists():
    """Posterior-dependent fitting: outcomes driven by a posterior feature must give the
    aware head a lower held-out MSE than the blind head trained identically."""
    torch.manual_seed(2)
    n, d = 512, 16
    x_base = torch.randn(n, d)
    feats = torch.rand(n, N_POSTERIOR_FEATURES)
    y = torch.tanh(3.0 * (feats[:, 1] - 0.5)) + 0.05 * torch.randn(n)

    def mlp():
        torch.manual_seed(3)
        return torch.nn.Sequential(
            torch.nn.Linear(d + N_POSTERIOR_FEATURES, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    hold = slice(0, 128)
    train = slice(128, n)
    x_aware = torch.cat([x_base, feats], dim=-1)
    x_blind = torch.cat([x_base, torch.zeros_like(feats)], dim=-1)
    aware, blind = mlp(), mlp()
    fit_value_mlp(aware, x_aware[train], y[train], epochs=40, lr=1e-2, batch_size=128, seed=0)
    fit_value_mlp(blind, x_blind[train], y[train], epochs=40, lr=1e-2, batch_size=128, seed=0)

    def mse(m, x):
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(torch.tanh(m(x[hold])).squeeze(-1), y[hold]))

    assert mse(aware, x_aware) < 0.5 * mse(blind, x_blind)


def test_fit_value_mlp_deterministic():
    torch.manual_seed(4)
    x, y = torch.randn(64, 8), torch.randn(64)

    def run():
        torch.manual_seed(5)
        m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        losses = fit_value_mlp(m, x, y, epochs=3, lr=1e-3, batch_size=16, seed=7)
        return losses, [p.detach().clone() for p in m.parameters()]

    l1, p1 = run()
    l2, p2 = run()
    assert l1 == l2
    for a, b in zip(p1, p2):
        torch.testing.assert_close(a, b)


def test_split_by_game_is_game_level():
    games = np.repeat(np.arange(5), 4)
    train, hold = _split_by_game(games, holdout_frac=0.4, seed=0)
    assert len(train) + len(hold) == 20
    assert not set(games[train]) & set(games[hold]), "a game must never straddle the split"
    with pytest.raises(ValueError):
        _split_by_game(np.zeros(8, dtype=np.int64), 0.2, 0)


def test_finetune_value_requires_posterior_agent():
    agent = FogAgent({})
    with pytest.raises(ValueError):
        finetune_value(agent, {"outcome": np.zeros(4, dtype=np.float32),
                               "game": np.zeros(4, dtype=np.int64)})


# --------------------------------------------------------------------- engine-backed end-to-end

engine = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")


@pytest.fixture(scope="module")
def posterior_agent():
    a = FogAgent.from_tables(AWARE_CFG)
    a.eval()
    return a


@engine
def test_collect_and_finetune_end_to_end(tmp_path, posterior_agent):
    """Logger -> npz -> loader -> fine-tune, on real (short) engine games. Smoke data from a fresh
    agent carries no guarantee the gate PASSES — assert the protocol, not the outcome."""
    from fog.search.posterior_log import collect_search_decisions, load_decisions

    meta = collect_search_decisions(
        posterior_agent, [P_ME, P_ME_VARIANT, P_OPP], tmp_path,
        n_games=2, iters=2, temperature=1.0, seed=0, max_turns=10,
    )
    stats = meta["stats"]
    assert stats["games"] == 2 and stats["decisions"] > 0
    # the true world holds every real-battle action, so the executed label must always resolve
    # (regression: paste-cased vs table-cased engine strings made every label miss)
    assert stats["skipped_unlabeled"] == 0

    data = load_decisions(tmp_path)
    n = data["outcome"].shape[0]
    assert n == stats["decisions"]
    assert data["posterior_features"].shape == (n, N_POSTERIOR_FEATURES)
    assert np.isfinite(data["posterior_features"]).all()
    assert (data["posterior_features"] >= 0).all()
    assert set(np.unique(data["outcome"])) <= {-1.0, 0.0, 1.0}
    assert data["search_dist"].shape == (n, 14)
    np.testing.assert_allclose(data["search_dist"].sum(axis=1), 1.0, rtol=1e-5)
    # the executed action was legal in its own row
    assert all(data["legality_mask"][i, data["action"][i]] for i in range(n))
    # masks round-tripped per family
    for fam in ("species", "item", "ability", "move"):
        assert f"mask_{fam}" in data

    if len(np.unique(data["game"])) < 2:
        pytest.skip("one game logged no decisions — nothing to split")
    report = finetune_value(posterior_agent, data, holdout_frac=0.5,
                            epochs=2, lr=1e-3, batch_size=64, seed=0)
    assert report["n_decisions"] == n
    assert np.isfinite(report["aware_holdout_mse"]) and np.isfinite(report["blind_holdout_mse"])
    assert isinstance(report["gate_passed"], bool)
