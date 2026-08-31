import torch

from fog.models.agent import FogAgent
from fog.training.bc import _policy_supervision_mask, _select_trainable_parameters


def test_policy_supervision_can_require_exact_chosen_provenance() -> None:
    batch = {
        "action": torch.tensor([1, 0, 2]),
        "policy_valid": torch.tensor([True, True, True]),
        "legal_provenance": torch.tensor([[1, 2, 0], [2, 0, 0], [0, 0, 2]]),
        "action_feature_provenance": torch.tensor(
            [[1, 2, 0], [1, 0, 0], [0, 0, 0]]
        ),
    }
    mask = _policy_supervision_mask(
        batch,
        {
            "require_exact_chosen_action_features": True,
            "require_exact_chosen_legality": True,
        },
    )
    assert mask.tolist() == [True, False, False]


def test_policy_supervision_fails_closed_without_requested_provenance() -> None:
    batch = {"action": torch.tensor([0]), "policy_valid": torch.tensor([True])}
    mask = _policy_supervision_mask(
        batch, {"require_exact_chosen_action_features": True}
    )
    assert mask.tolist() == [False]


def test_value_scope_trains_only_value_head() -> None:
    agent = FogAgent.from_tables({})
    selected = _select_trainable_parameters(agent, "value")
    selected_ids = {id(parameter) for parameter in selected}
    value_ids = {id(parameter) for parameter in agent.policy_value.value.parameters()}
    assert selected_ids == value_ids
    assert selected_ids
    assert all(
        parameter.requires_grad == (id(parameter) in value_ids)
        for parameter in agent.parameters()
    )


def test_belief_scope_remains_narrow() -> None:
    agent = FogAgent.from_tables({})
    selected = _select_trainable_parameters(agent, "belief")
    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in agent.belief.parameters()
    }


def test_shared_belief_scope_protects_exact_information_paths() -> None:
    agent = FogAgent.from_tables(
        {"model": {"shared_latent_trunk": True, "shared_latent_dim": 4096}}
    )
    selected = _select_trainable_parameters(agent, "belief_shared")
    selected_ids = {id(parameter) for parameter in selected}
    expected_modules = [
        agent.belief,
        agent.shared_to_belief,
        agent.shared_latent.semantic_entity,
        agent.shared_latent.global_context,
    ]
    expected_ids = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert selected_ids
    assert not any(
        parameter.requires_grad for parameter in agent.shared_latent.exact_entity.parameters()
    )
    assert not any(parameter.requires_grad for parameter in agent.shared_latent.field.parameters())
    assert not any(parameter.requires_grad for parameter in agent.policy_value.parameters())
    assert not any(parameter.requires_grad for parameter in agent.self_belief.parameters())


def _contextual_agent() -> FogAgent:
    return FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_latent_dim": 2048,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 1,
                "shared_contextual_heads": 4,
                "shared_contextual_ff_dim": 256,
                "shared_contextual_head_readers": True,
                "token_action_policy": True,
                "use_action_features": True,
            }
        }
    )


def _flat_shared_agent() -> FogAgent:
    return FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": False,
                "shared_latent_flat_autoencoder": True,
                "shared_latent_dim": 2048,
                "shared_flat_hidden_dim": 256,
                "shared_flat_layers": 1,
                "use_action_features": True,
            }
        }
    )


def test_flat_shared_scopes_train_the_entire_learned_trunk() -> None:
    for scope, primary_modules in (
        ("belief_shared", lambda agent: [agent.belief, agent.shared_to_belief]),
        ("value_shared", lambda agent: [agent.policy_value.value]),
        ("policy_shared", _policy_modules_for_test),
    ):
        agent = _flat_shared_agent()
        selected = _select_trainable_parameters(agent, scope)
        expected_modules = [*primary_modules(agent), agent.shared_latent]
        assert {id(parameter) for parameter in selected} == {
            id(parameter)
            for module in expected_modules
            for parameter in module.parameters()
        }


def _policy_modules_for_test(agent: FogAgent) -> list:
    return [agent.policy_value.policy, agent.policy_value.action_scorer]


def test_contextual_belief_shared_scope_trains_reader_belief_and_entire_trunk() -> None:
    agent = _contextual_agent()
    selected = _select_trainable_parameters(agent, "belief_shared")
    selected_ids = {id(parameter) for parameter in selected}
    expected_modules = [agent.belief, agent.shared_to_belief, agent.shared_latent]
    expected_ids = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert not any(parameter.requires_grad for parameter in agent.policy_value.parameters())
    assert not any(parameter.requires_grad for parameter in agent.backbone.parameters())


def test_contextual_value_shared_scope_trains_value_and_entire_trunk() -> None:
    agent = _contextual_agent()
    selected = _select_trainable_parameters(agent, "value_shared")
    selected_ids = {id(parameter) for parameter in selected}
    expected_ids = {
        id(parameter)
        for module in [
            agent.policy_value.value,
            agent.shared_to_value,
            agent.shared_latent,
        ]
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert not any(parameter.requires_grad for parameter in agent.belief.parameters())
    assert not any(parameter.requires_grad for parameter in agent.policy_value.policy.parameters())


def test_contextual_belief_value_shared_scope_trains_both_heads_and_one_trunk() -> None:
    agent = _contextual_agent()
    selected = _select_trainable_parameters(agent, "belief_value_shared")
    selected_ids = {id(parameter) for parameter in selected}
    expected_ids = {
        id(parameter)
        for module in [
            agent.belief,
            agent.shared_to_belief,
            agent.policy_value.value,
            agent.shared_to_value,
            agent.shared_latent,
        ]
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert len(selected_ids) == len(selected)
    assert not any(parameter.requires_grad for parameter in agent.policy_value.policy.parameters())


def test_contextual_policy_shared_scope_trains_policy_scorers_and_entire_trunk() -> None:
    agent = _contextual_agent()
    selected = _select_trainable_parameters(agent, "policy_shared")
    selected_ids = {id(parameter) for parameter in selected}
    expected_modules = [
        agent.policy_value.policy,
        agent.policy_value.action_scorer,
        agent.policy_value.conditional_action_scorer,
        agent.shared_to_policy,
        agent.shared_latent,
    ]
    expected_ids = {
        id(parameter)
        for module in expected_modules
        if module is not None
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert not any(parameter.requires_grad for parameter in agent.belief.parameters())
    assert not any(parameter.requires_grad for parameter in agent.policy_value.value.parameters())


def test_typed_policy_scope_trains_shared_candidate_scorer_not_fixed_index_head() -> None:
    agent = FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_typed_world_tokens": True,
                "shared_recent_history_tokens": 8,
                "shared_latent_dim": 3072,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 1,
                "shared_contextual_heads": 4,
                "shared_contextual_ff_dim": 256,
                "shared_contextual_head_readers": True,
                "token_action_policy": True,
                "action_query_policy": True,
            }
        }
    )
    selected = _select_trainable_parameters(agent, "policy")
    selected_ids = {id(parameter) for parameter in selected}
    expected_modules = [
        agent.policy_value.action_query_input,
        agent.policy_value.action_query_attention,
        agent.policy_value.action_query_scorer,
        agent.shared_to_policy,
    ]
    assert selected_ids == {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert not any(parameter.requires_grad for parameter in agent.policy_value.policy.parameters())


def test_typed_belief_scope_includes_embeddings_and_temporal_encoder() -> None:
    agent = FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_typed_world_tokens": True,
                "shared_recent_history_tokens": 8,
                "shared_latent_dim": 3072,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 1,
                "shared_contextual_heads": 4,
                "shared_contextual_ff_dim": 256,
            }
        }
    )
    selected = _select_trainable_parameters(agent, "belief_shared")
    selected_ids = {id(parameter) for parameter in selected}
    assert all(id(parameter) in selected_ids for parameter in agent.backbone.embeds.parameters())
    assert all(
        id(parameter) in selected_ids
        for parameter in agent.backbone.turn_history_encoder.parameters()
    )


def test_value_prior_scope_trains_only_typed_matchup_prior() -> None:
    agent = FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_typed_world_tokens": True,
                "shared_recent_history_tokens": 8,
                "shared_latent_dim": 3072,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 1,
                "shared_contextual_heads": 4,
                "shared_contextual_ff_dim": 256,
                "matchup_value_prior": True,
            }
        }
    )
    selected = _select_trainable_parameters(agent, "value_prior")
    selected_ids = {id(parameter) for parameter in selected}
    expected_ids = {id(parameter) for parameter in agent.matchup_value_prior.parameters()}
    assert selected_ids == expected_ids
    assert selected_ids
    assert all(
        parameter.requires_grad == (id(parameter) in expected_ids)
        for parameter in agent.parameters()
    )


def test_belief_value_prior_shared_scope_includes_prior_and_excludes_policy() -> None:
    agent = FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_typed_world_tokens": True,
                "shared_recent_history_tokens": 8,
                "shared_latent_dim": 3072,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 1,
                "shared_contextual_heads": 4,
                "shared_contextual_ff_dim": 256,
                "shared_contextual_head_readers": True,
                "matchup_value_prior": True,
            }
        }
    )
    selected = _select_trainable_parameters(agent, "belief_value_prior_shared")
    selected_ids = {id(parameter) for parameter in selected}
    assert all(
        id(parameter) in selected_ids
        for parameter in agent.matchup_value_prior.parameters()
    )
    assert all(id(parameter) in selected_ids for parameter in agent.belief.parameters())
    assert all(
        id(parameter) in selected_ids for parameter in agent.policy_value.value.parameters()
    )
    assert not any(
        id(parameter) in selected_ids for parameter in agent.policy_value.policy.parameters()
    )


def test_joint_policy_belief_value_prior_scope_includes_all_heads_and_typed_trunk() -> None:
    agent = FogAgent.from_tables(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_typed_world_tokens": True,
                "shared_recent_history_tokens": 8,
                "shared_latent_dim": 3072,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 1,
                "shared_contextual_heads": 4,
                "shared_contextual_ff_dim": 256,
                "shared_contextual_head_readers": True,
                "token_action_policy": True,
                "action_query_policy": True,
                "matchup_value_prior": True,
            }
        }
    )
    selected = _select_trainable_parameters(
        agent, "policy_belief_value_prior_shared"
    )
    selected_ids = {id(parameter) for parameter in selected}
    expected_modules = [
        agent.policy_value.action_query_input,
        agent.policy_value.action_query_attention,
        agent.policy_value.action_query_scorer,
        agent.shared_to_policy,
        agent.belief,
        agent.shared_to_belief,
        agent.policy_value.value,
        agent.shared_to_value,
        agent.matchup_value_prior,
        agent.shared_latent,
        agent.backbone.embeds,
        agent.backbone.turn_history_encoder,
    ]
    expected_ids = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert len(selected_ids) == len(selected)
    assert not any(parameter.requires_grad for parameter in agent.self_belief.parameters())
    assert not any(parameter.requires_grad for parameter in agent.policy_value.policy.parameters())


def test_policy_scope_trains_only_policy_and_public_action_scorer() -> None:
    agent = FogAgent.from_tables({"model": {"use_action_features": True}})
    selected = _select_trainable_parameters(agent, "policy")
    selected_ids = {id(parameter) for parameter in selected}
    expected_ids = {id(parameter) for parameter in agent.policy_value.policy.parameters()}
    expected_ids.update(
        id(parameter) for parameter in agent.policy_value.action_scorer.parameters()
    )
    assert selected_ids == expected_ids
    assert selected_ids
    assert all(
        parameter.requires_grad == (id(parameter) in expected_ids)
        for parameter in agent.parameters()
    )


def test_policy_late_encoder_scope_keeps_early_encoders_and_other_heads_frozen() -> None:
    agent = FogAgent.from_tables({"model": {"use_action_features": True}})
    selected = _select_trainable_parameters(agent, "policy_late_encoders")
    selected_ids = {id(parameter) for parameter in selected}
    expected_modules = [
        agent.policy_value.policy,
        agent.policy_value.action_scorer,
        agent.backbone.current_state_encoder.head,
        agent.backbone.turn_history_encoder.head,
        agent.backbone.private_state_encoder.head,
        agent.backbone.public_fusion,
    ]
    expected_ids = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert selected_ids == expected_ids
    assert selected_ids
    assert not any(parameter.requires_grad for parameter in agent.backbone.embeds.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in agent.backbone.turn_history_encoder.sequence.parameters()
    )
    assert not any(parameter.requires_grad for parameter in agent.belief.parameters())
    assert not any(parameter.requires_grad for parameter in agent.self_belief.parameters())
    assert not any(parameter.requires_grad for parameter in agent.policy_value.value.parameters())


def test_structured_scopes_train_only_their_head_and_private_reader() -> None:
    agent = FogAgent.from_tables(
        {
            "model": {
                "structured_memory": True,
                "token_action_policy": True,
                "exact_action_context": True,
            }
        }
    )
    cases = {
        "belief_structured": [
            agent.belief,
            agent.belief_structured_memory,
            agent.structured_to_belief,
        ],
        "value_structured": [
            agent.policy_value.value,
            agent.policy_value.structured_value,
            agent.value_structured_memory,
        ],
        "policy_structured": [
            agent.policy_value.policy,
            agent.policy_value.conditional_action_scorer,
            agent.policy_structured_memory,
        ],
    }
    for scope, modules in cases.items():
        selected = _select_trainable_parameters(agent, scope)
        selected_ids = {id(parameter) for parameter in selected}
        expected_ids = {
            id(parameter)
            for module in modules
            for parameter in module.parameters()
        }
        assert selected_ids == expected_ids
        assert selected_ids
        assert all(
            parameter.requires_grad == (id(parameter) in expected_ids)
            for parameter in agent.parameters()
        )
