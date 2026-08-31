import pytest
import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.models.agent import FogAgent
from fog.models.shared_latent import ContextualLatentReader, OpponentContextualLatentReader


def _agent() -> FogAgent:
    return FogAgent(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_dim": 3072,
                "shared_entity_dim": 192,
                "shared_field_dim": 64,
                "shared_residual_dim": 128,
            },
            "encoding": {
                "entity_hidden": 16,
                "state_dim": 24,
                "current_dim": 20,
                "history_dim": 20,
                "dyn_dim": 24,
                "history_cap": 8,
                "dropout": 0.0,
            },
        },
        sizes={
            "n_species": 16,
            "n_items": 12,
            "n_abilities": 12,
            "n_moves": 24,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    ).eval()


def _batch() -> dict:
    generator = torch.Generator().manual_seed(701)
    batch = {
        "entities": torch.randn(2, 12, ENTITY_RAW_DIM, generator=generator),
        "entity_ids": torch.zeros(2, 12, N_ID_FIELDS, dtype=torch.long),
        "field": torch.randn(2, DIMS.field_dim, generator=generator),
        "history": torch.randn(2, 4, DIMS.turn_dim, generator=generator),
        "history_ids": torch.zeros(2, 4, DIMS.hist_id_fields, dtype=torch.long),
        "history_state": torch.zeros(2, 4, DIMS.hist_state),
        "history_len": torch.full((2,), 4, dtype=torch.long),
        "active_idx": torch.tensor([[0, 6], [2, 9]]),
        "legality_mask": torch.ones(2, 14, dtype=torch.bool),
    }
    return batch


def test_shared_latent_is_one_vector_used_by_all_three_heads() -> None:
    agent = _agent()
    batch = _batch()
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        latent = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        output = agent(batch)
    assert latent.shape == (2, 3072)
    assert torch.isfinite(latent).all()
    assert output.log_probs.shape == (2, 14)
    assert output.value.shape == (2,)
    assert agent.shared_to_belief.in_features == 3072
    assert agent.policy_value.shared_state_dim == 3072


def test_shared_latent_changes_when_exact_current_state_changes() -> None:
    agent = _agent()
    batch = _batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["entities"][:, 7, 0] += 0.5
    with torch.no_grad():
        first = agent.shared_latent(batch, agent.encode_contexts(batch), agent.backbone.embeds)
        second = agent.shared_latent(
            changed, agent.encode_contexts(changed), agent.backbone.embeds
        )
    assert not torch.allclose(first, second)


def test_exact_entity_projection_starts_full_column_rank() -> None:
    agent = _agent()
    weight = agent.shared_latent.exact_entity.weight.detach().float()
    assert weight.shape[0] >= weight.shape[1]
    assert int(torch.linalg.matrix_rank(weight)) == weight.shape[1]
    field = agent.shared_latent.field.weight.detach().float()
    assert field.shape[0] >= field.shape[1]
    assert int(torch.linalg.matrix_rank(field)) == field.shape[1]


def test_learned_only_shared_latent_has_no_exact_information_bypass() -> None:
    agent = FogAgent(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_dim": 4096,
                "shared_entity_dim": 256,
                "shared_field_dim": 128,
                "shared_residual_dim": 256,
            },
            "encoding": {
                "entity_hidden": 16,
                "state_dim": 24,
                "current_dim": 20,
                "history_dim": 20,
                "dyn_dim": 24,
                "history_cap": 8,
                "dropout": 0.0,
            },
        },
        sizes={
            "n_species": 16,
            "n_items": 12,
            "n_abilities": 12,
            "n_moves": 24,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    ).eval()
    trunk = agent.shared_latent
    assert trunk.learned_only
    assert trunk.exact_entity is None
    assert trunk.field is None
    assert trunk.semantic_entity is None
    assert trunk.entity_encoder is not None
    assert trunk.field_encoder is not None
    assert trunk.history_encoder is not None
    with torch.no_grad():
        latent = trunk(_batch(), agent.encode_contexts(_batch()), agent.backbone.embeds)
    assert latent.shape == (2, 4096)
    assert torch.isfinite(latent).all()


def test_flat_autoencoder_is_one_unpartitioned_learned_vector() -> None:
    agent = FogAgent(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_flat_autoencoder": True,
                "shared_latent_dim": 2048,
                "shared_flat_hidden_dim": 256,
                "shared_flat_layers": 2,
            },
            "encoding": {
                "entity_hidden": 16,
                "state_dim": 24,
                "current_dim": 20,
                "history_dim": 20,
                "dyn_dim": 24,
                "history_cap": 8,
                "dropout": 0.0,
            },
        },
        sizes={
            "n_species": 16,
            "n_items": 12,
            "n_abilities": 12,
            "n_moves": 24,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    ).train()
    trunk = agent.shared_latent
    assert trunk.flat_autoencoder
    assert not trunk.contextual
    assert trunk.entity_encoder is None
    assert trunk.exact_entity is None
    assert trunk.global_context is None

    batch = _batch()
    latent = trunk(batch, agent.encode_contexts(batch), agent.backbone.embeds)
    assert latent.shape == (2, 2048)
    assert torch.isfinite(latent).all()
    latent.square().mean().backward()
    assert trunk.flat_input_encoder[0].weight.grad is not None
    assert trunk.flat_blocks[0].net[0].weight.grad is not None

    with torch.no_grad():
        output = agent.eval()(batch)
    assert output.log_probs.shape == (2, 14)
    assert output.value.shape == (2,)


def _contextual_agent(
    *, enhanced: bool = False, head_readers: bool = False
) -> FogAgent:
    return FogAgent(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_latent_dim": 2048,
                "shared_entity_dim": 128,
                "shared_contextual_layers": 2,
                "shared_contextual_heads": 8,
                "shared_contextual_ff_dim": 256,
                "shared_contextual_dynamic_residual": enhanced,
                "shared_contextual_data_global": enhanced,
                "shared_contextual_head_readers": head_readers,
                "token_action_policy": True,
            },
            "encoding": {
                "entity_hidden": 16,
                "state_dim": 24,
                "current_dim": 20,
                "history_dim": 20,
                "dyn_dim": 24,
                "history_cap": 8,
                "dropout": 0.0,
            },
        },
        sizes={
            "n_species": 16,
            "n_items": 12,
            "n_abilities": 12,
            "n_moves": 24,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    ).eval()


def _typed_contextual_agent(
    *, entity_dim: int = 128, matchup_value_prior: bool = False
) -> FogAgent:
    if entity_dim % 32:
        raise ValueError("typed test entity width must be divisible by 32")
    return FogAgent(
        {
            "model": {
                "shared_latent_trunk": True,
                "shared_latent_learned_only": True,
                "shared_latent_contextual": True,
                "shared_typed_world_tokens": True,
                "shared_recent_history_tokens": 8,
                "shared_latent_dim": 24 * entity_dim,
                "shared_entity_dim": entity_dim,
                "shared_contextual_layers": 2,
                "shared_contextual_heads": entity_dim // 32,
                "shared_contextual_ff_dim": 256,
                "shared_contextual_dropout": 0.0,
                "shared_contextual_head_readers": True,
                "token_action_policy": True,
                "action_query_policy": True,
                "matchup_value_prior": matchup_value_prior,
            },
            "encoding": {
                "entity_hidden": 16,
                "state_dim": 24,
                "current_dim": 20,
                "history_dim": 20,
                "dyn_dim": 24,
                "history_cap": 8,
                "dropout": 0.0,
            },
        },
        sizes={
            "n_species": 16,
            "n_items": 12,
            "n_abilities": 12,
            "n_moves": 24,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    ).eval()


def test_typed_matchup_prior_is_permutation_symmetric_and_compatible() -> None:
    base = _typed_contextual_agent()
    prior = _typed_contextual_agent(matchup_value_prior=True)
    manifest = prior.load_compatible_state_dict(base.state_dict())
    assert any(name.startswith("matchup_value_prior.") for name in manifest["initialized"])

    batch = _batch()
    with torch.no_grad():
        base_output = base(batch)
        prior_output = prior(batch)
        contexts = prior.encode_contexts(batch)
        latent, entity_tokens, field_token = prior.shared_latent(
            batch, contexts, prior.backbone.embeds, return_precontext=True
        )
        expected = prior.matchup_value_prior(entity_tokens, field_token)
        permutation = torch.tensor([2, 0, 5, 1, 4, 3, 9, 7, 11, 8, 6, 10])
        actual = prior.matchup_value_prior(entity_tokens[:, permutation], field_token)

    assert latent.shape == (2, 3072)
    assert entity_tokens.shape == (2, 12, 128)
    assert field_token.shape == (2, 128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(base_output.log_probs, prior_output.log_probs, rtol=0, atol=0)
    torch.testing.assert_close(base_output.value, prior_output.value, rtol=0, atol=0)


def test_initial_only_matchup_prior_cannot_change_later_state_value() -> None:
    base = _typed_contextual_agent()
    gated = _typed_contextual_agent(matchup_value_prior=True)
    gated.matchup_value_prior_initial_only = True
    gated.load_compatible_state_dict(base.state_dict())
    with torch.no_grad():
        gated.matchup_value_prior.output[-1].bias.fill_(1.0)
    batch = _batch()
    batch["history_len"] = torch.tensor([0, 4])
    with torch.no_grad():
        base_value = base(batch).value
        gated_value = gated(batch).value
    assert gated_value[0] != base_value[0]
    torch.testing.assert_close(gated_value[1], base_value[1], rtol=0, atol=0)


def test_contextual_shared_trunk_is_sixteen_learned_tokens() -> None:
    agent = _contextual_agent()
    batch = _batch()
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        latent = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        tokens = agent.shared_latent.tokens(latent)
        output = agent(batch)

    assert latent.shape == (2, 2048)
    assert tokens.shape == (2, 16, 128)
    assert torch.isfinite(tokens).all()
    assert agent.shared_latent.exact_entity is None
    assert agent.shared_latent.field is None
    assert isinstance(agent.shared_to_belief, ContextualLatentReader)
    assert agent.policy_value.shared_token_dim == 128
    assert agent.policy_value.policy[0].in_features == 128
    assert agent.policy_value.value[0].in_features == 128
    assert output.log_probs.shape == (2, 14)
    assert output.value.shape == (2,)


def test_contextual_global_and_action_tokens_mix_battle_information() -> None:
    agent = _contextual_agent()
    batch = _batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["entities"][:, 7, 0] += 0.5

    with torch.no_grad():
        first_contexts = agent.encode_contexts(batch)
        first = agent.shared_latent(batch, first_contexts, agent.backbone.embeds)
        second_contexts = agent.encode_contexts(changed)
        second = agent.shared_latent(changed, second_contexts, agent.backbone.embeds)
        first_tokens = agent.shared_latent.tokens(first)
        second_tokens = agent.shared_latent.tokens(second)
        action_context = agent.action_context(first_contexts, batch, first)

    # The changed opponent slot affects the shared global token through contextual attention.
    assert not torch.allclose(first_tokens[:, -1], second_tokens[:, -1])
    # Every candidate receives a contextual candidate/opponent pair plus its move embedding.
    assert action_context.shape == (2, 14, 2 * 128 + 24 + 4)


def test_contextual_policy_and_value_read_all_sixteen_tokens() -> None:
    agent = _contextual_agent(head_readers=True).eval()
    assert isinstance(agent.shared_to_policy, ContextualLatentReader)
    assert isinstance(agent.shared_to_value, ContextualLatentReader)
    assert agent.shared_to_policy is not agent.shared_to_value

    batch = _batch()
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        latent = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        policy_read = agent.shared_to_policy(latent)
        value_read = agent.shared_to_value(latent)
        output = agent(batch)

        # Alter a creature token while leaving the final global token unchanged.  Both
        # task readers inspect the complete token set rather than only tokens[:, -1].
        changed = latent.clone().reshape(latent.shape[0], 16, 128)
        changed[:, 0] += 0.25
        changed = changed.flatten(1)
        changed_policy = agent.shared_to_policy(changed)
        changed_value = agent.shared_to_value(changed)

    assert policy_read.shape == (2, 128)
    assert value_read.shape == (2, 128)
    assert not torch.allclose(policy_read, changed_policy)
    assert not torch.allclose(value_read, changed_value)
    assert output.log_probs.shape == (2, 14)
    assert output.value.shape == (2,)


def test_contextual_task_readers_receive_only_their_head_gradients() -> None:
    agent = _contextual_agent(head_readers=True).train()
    output = agent(_batch())
    (-output.log_probs[:, 0].mean()).backward(retain_graph=True)
    assert agent.shared_to_policy.query.grad is not None
    assert agent.shared_to_value.query.grad is None

    agent.zero_grad(set_to_none=True)
    output.value.mean().backward()
    assert agent.shared_to_value.query.grad is not None
    assert agent.shared_to_policy.query.grad is None


def test_contextual_trunk_receives_gradients_from_shared_consumers() -> None:
    agent = _contextual_agent().train()
    batch = _batch()
    output = agent(batch)
    contexts = agent.encode_contexts(batch)
    latent = agent.shared_latent(batch, contexts, agent.backbone.embeds)
    belief_context = agent.belief_context(contexts, shared_state=latent)
    loss = -output.log_probs[:, 0].mean() + output.value.mean() + belief_context.mean()
    loss.backward()

    assert agent.shared_latent.entity_encoder[0].weight.grad is not None
    layer = agent.shared_latent.contextual_encoder.layers[0]
    assert layer.self_attn.in_proj_weight.grad is not None


def test_contextual_dynamic_and_global_routes_are_learned_not_bypasses() -> None:
    agent = _contextual_agent(enhanced=True).train()
    trunk = agent.shared_latent
    assert trunk.dynamic_entity_encoder is not None
    assert trunk.global_state_encoder is not None
    assert trunk.exact_entity is None

    batch = _batch()
    latent = trunk(batch, agent.encode_contexts(batch), agent.backbone.embeds)
    assert latent.shape == (2, 2048)
    latent.square().mean().backward()

    assert trunk.dynamic_entity_encoder[-1].weight.grad is not None
    assert trunk.global_state_encoder[-1].weight.grad is not None


def test_contextual_enhancement_starts_output_identical_after_compatible_load() -> None:
    base = _contextual_agent()
    enhanced = _contextual_agent(enhanced=True)
    enhanced.load_compatible_state_dict(base.state_dict())
    batch = _batch()
    with torch.no_grad():
        base_output = base.shared_latent(
            batch, base.encode_contexts(batch), base.backbone.embeds
        )
        enhanced_output = enhanced.shared_latent(
            batch, enhanced.encode_contexts(batch), enhanced.backbone.embeds
        )
    torch.testing.assert_close(base_output, enhanced_output, rtol=0, atol=0)


def test_typed_world_trunk_emits_twenty_four_distinct_tokens() -> None:
    agent = _typed_contextual_agent()
    batch = _batch()
    batch["decision_phase"] = torch.tensor([0, 1])
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        latent = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        tokens = agent.shared_latent.tokens(latent)
        output = agent(batch)

    assert latent.shape == (2, 3072)
    assert tokens.shape == (2, 24, 128)
    assert agent.shared_latent.contextual_kind_ids.tolist() == (
        [0] * 6 + [1] * 6 + [2, 3] + [4] * 8 + [5, 6]
    )
    assert agent.shared_latent.contextual_side_ids.tolist() == [0] * 6 + [1] * 6 + [2] * 12
    assert isinstance(agent.shared_to_belief, OpponentContextualLatentReader)
    assert agent.belief_context(contexts, shared_state=latent).shape == (2, 6, 24)
    assert output.log_probs.shape == (2, 14)
    assert output.value.shape == (2,)


def test_typed_world_trunk_supports_evidence_justified_256_wide_tokens() -> None:
    agent = _typed_contextual_agent(entity_dim=256)
    batch = _batch()
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        latent = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        tokens = agent.shared_latent.tokens(latent)
        output = agent(batch)

    assert latent.shape == (2, 6144)
    assert tokens.shape == (2, 24, 256)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(output.value).all()
    assert output.log_probs.shape == (2, 14)


def test_typed_world_recent_history_uses_only_real_prefix() -> None:
    agent = _typed_contextual_agent()
    batch = _batch()
    batch["history_len"] = torch.tensor([2, 4])
    contexts = agent.encode_contexts(batch)
    recent = agent.shared_latent._recent_history(contexts)

    assert recent.shape == (2, 8, 16)
    torch.testing.assert_close(recent[0, :2], contexts.history_tokens[0, :2])
    assert torch.count_nonzero(recent[0, 2:]) == 0
    torch.testing.assert_close(recent[1, :4], contexts.history_tokens[1, :4])
    assert torch.count_nonzero(recent[1, 4:]) == 0


def test_typed_world_rejects_invalid_request_phase() -> None:
    agent = _typed_contextual_agent()
    batch = _batch()
    batch["decision_phase"] = torch.tensor([0, 6])
    contexts = agent.encode_contexts(batch)
    with torch.no_grad(), pytest.raises(ValueError, match="decision_phase"):
        agent.shared_latent(batch, contexts, agent.backbone.embeds)


def test_action_query_policy_is_candidate_permutation_equivariant() -> None:
    agent = _typed_contextual_agent().eval()
    batch = _batch()
    permutation = torch.tensor([8, 0, 4, 9, 1, 5, 10, 2, 6, 11, 3, 7, 12, 13])
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        shared = agent.shared_latent(batch, contexts, agent.backbone.embeds)
        action_context = agent.action_context(contexts, batch, shared)
        original, _ = agent.policy_value(
            contexts.private_state,
            batch["legality_mask"],
            agent.public_context(contexts),
            action_context=action_context,
            shared_state=shared,
            policy_shared_context=agent.shared_to_policy(shared),
            value_shared_context=agent.shared_to_value(shared),
        )
        permuted, _ = agent.policy_value(
            contexts.private_state,
            batch["legality_mask"][:, permutation],
            agent.public_context(contexts),
            action_context=action_context[:, permutation],
            shared_state=shared,
            policy_shared_context=agent.shared_to_policy(shared),
            value_shared_context=agent.shared_to_value(shared),
        )

    torch.testing.assert_close(permuted, original[:, permutation], rtol=0, atol=1e-6)
