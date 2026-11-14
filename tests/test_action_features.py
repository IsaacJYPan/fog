import numpy as np
import torch

from fog.encoding.action_features import (
    ACTION_FEATURE_DIM,
    ACTION_FEATURE_NAMES,
    encode_action_features,
)
from fog.models.heads import PolicyValueHead
from tests.golden_fixture import GoldenBattle


def test_action_features_are_finite_aligned_and_pick_a_legal_choice():
    battle = GoldenBattle()
    features = encode_action_features(battle)
    assert features.shape == (14, ACTION_FEATURE_DIM)
    assert np.isfinite(features).all()
    picked = np.flatnonzero(features[:, ACTION_FEATURE_NAMES.index("simple_rule_pick")])
    assert len(picked) == 1
    assert features[int(picked[0]), ACTION_FEATURE_NAMES.index("legal")] == 1


def test_rule_prior_starts_from_the_rule_choice():
    head = PolicyValueHead(
        state_dim=4,
        dyn_dim=4,
        action_feature_dim=ACTION_FEATURE_DIM,
        simple_rule_prior_strength=50.0,
    )
    for parameter in head.parameters():
        torch.nn.init.zeros_(parameter)
    action_features = torch.zeros(2, 14, ACTION_FEATURE_DIM)
    action_features[0, 7, -1] = 1
    action_features[1, 11, -1] = 1
    legal = torch.ones(2, 14, dtype=torch.bool)
    log_probs, _ = head(
        torch.zeros(2, 4), legal, torch.zeros(2, 4), action_features=action_features
    )
    assert log_probs.argmax(-1).tolist() == [7, 11]


def test_old_policy_path_does_not_require_action_features():
    head = PolicyValueHead(state_dim=4, dyn_dim=4)
    legal = torch.ones(1, 14, dtype=torch.bool)
    log_probs, value = head(torch.zeros(1, 4), legal, torch.zeros(1, 4))
    assert log_probs.shape == (1, 14)
    assert value.shape == (1,)


def test_conditional_action_context_is_optional_and_behavior_preserving_at_init():
    head = PolicyValueHead(
        state_dim=4,
        dyn_dim=4,
        action_feature_dim=ACTION_FEATURE_DIM,
        action_context_dim=7,
    )
    legal = torch.ones(2, 14, dtype=torch.bool)
    private = torch.randn(2, 4)
    public = torch.randn(2, 4)
    features = torch.randn(2, 14, ACTION_FEATURE_DIM)
    context = torch.randn(2, 14, 7)
    with_context, _ = head(
        private, legal, public, action_features=features, action_context=context
    )
    without_context, _ = head(private, legal, public, action_features=features)
    assert torch.equal(with_context, without_context)


def test_matchup_reuse_is_confined_to_one_feature_table():
    """Remembered rule-player matchup scores never outlive `encode_action_features`."""
    from poke_env.player import SimpleHeuristicsPlayer

    from fog.encoding import action_features

    class Boom:
        @property
        def active_pokemon(self):
            raise RuntimeError("boom")

    try:
        action_features.encode_action_features(Boom())
    except RuntimeError:
        pass
    assert action_features._MATCHUP_MEMO is None

    calls = []
    original = action_features._MATCHUP_ORIGINAL
    assert original is not None
    try:
        action_features._MATCHUP_ORIGINAL = None
        SimpleHeuristicsPlayer._estimate_matchup = staticmethod(
            lambda mon, opponent: calls.append((mon, opponent)) or 1.5
        )
        action_features._install_matchup_memo()
        # Outside a feature table every call goes straight through.
        assert SimpleHeuristicsPlayer._estimate_matchup("a", "b") == 1.5
        assert SimpleHeuristicsPlayer._estimate_matchup("a", "b") == 1.5
        assert len(calls) == 2
    finally:
        SimpleHeuristicsPlayer._estimate_matchup = staticmethod(original)
        action_features._MATCHUP_ORIGINAL = None
        action_features._install_matchup_memo()
