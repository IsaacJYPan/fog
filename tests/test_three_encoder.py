"""Contracts for the public-current / causal-history / private-state backbone."""
from __future__ import annotations

import copy

import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    ENTITY_RAW_DIM,
    EV_ID_FIELDS,
    N_ID_FIELDS,
    NATURE_ID_FIELD,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
    L,
)
from fog.models.agent import FogAgent
from fog.models.encoder import (
    Backbone,
    CurrentStateEncoder,
    PrivateStateEncoder,
    TurnHistoryEncoder,
)


def _backbone() -> Backbone:
    torch.manual_seed(200)
    return Backbone(
        {
            "entity_hidden": 16,
            "state_dim": 24,
            "current_dim": 20,
            "history_dim": 20,
            "dyn_dim": 24,
            "history_encoder": "causal_transformer",
            "history_heads": 4,
            "history_transformer_layers": 2,
            "history_cap": 12,
            "dropout": 0.0,
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


def _inputs(batch: int = 2, turns: int = 5) -> dict:
    generator = torch.Generator().manual_seed(201)
    entities = torch.randn(batch, 12, ENTITY_RAW_DIM, generator=generator)
    entity_ids = torch.zeros(batch, 12, N_ID_FIELDS, dtype=torch.long)
    entity_ids[..., 0] = torch.randint(0, 16, (batch, 12), generator=generator)
    entity_ids[..., 1] = torch.randint(0, 12, (batch, 12), generator=generator)
    entity_ids[..., 2] = torch.randint(0, 12, (batch, 12), generator=generator)
    entity_ids[..., 3:8] = torch.randint(0, 24, (batch, 12, 5), generator=generator)
    entity_ids[..., NATURE_ID_FIELD] = torch.randint(
        0, 25, (batch, 12), generator=generator
    )
    entity_ids[..., list(EV_ID_FIELDS)] = torch.randint(
        0, 64, (batch, 12, 6), generator=generator
    )
    entity_ids[..., ORIGINAL_ITEM_ID_FIELD] = torch.randint(
        0, 12, (batch, 12), generator=generator
    )
    entity_ids[..., ORIGINAL_ABILITY_ID_FIELD] = torch.randint(
        0, 12, (batch, 12), generator=generator
    )
    return {
        "entities": entities,
        "entity_ids": entity_ids,
        "field": torch.randn(batch, DIMS.field_dim, generator=generator),
        "history": torch.randn(batch, turns, DIMS.turn_dim, generator=generator),
        "active_idx": torch.tensor([[0, 6]] * batch, dtype=torch.long),
        "history_len": torch.full((batch,), turns, dtype=torch.long),
    }


def test_backbone_contains_three_distinct_encoder_modules():
    backbone = _backbone()
    assert isinstance(backbone.current_state_encoder, CurrentStateEncoder)
    assert isinstance(backbone.turn_history_encoder, TurnHistoryEncoder)
    assert isinstance(backbone.private_state_encoder, PrivateStateEncoder)
    assert backbone.history_encoder == "causal_transformer"

    contexts = backbone.encode_contexts(**_inputs())
    assert contexts.public_state.shape == (2, 20)
    assert contexts.history.shape == (2, 20)
    assert contexts.private_state.shape == (2, 24)


def test_structured_memory_retains_exact_available_inputs():
    backbone = _backbone()
    inputs = _inputs()
    contexts = backbone.encode_contexts(**inputs)
    assert torch.equal(contexts.private_exact_raw, inputs["entities"][:, :6])
    assert torch.equal(contexts.private_exact_ids, inputs["entity_ids"][:, :6])
    assert torch.equal(contexts.field_exact, inputs["field"])
    assert torch.equal(contexts.history_exact_raw, inputs["history"])
    assert torch.equal(contexts.history_lengths, inputs["history_len"])
    # Public memory preserves public facts but intentionally masks private sheet identities.
    assert torch.equal(contexts.public_exact_ids[:, 6:], inputs["entity_ids"][:, 6:])
    assert not torch.equal(contexts.public_exact_ids[:, :6], inputs["entity_ids"][:, :6])


def test_structured_head_reader_is_finite_sensitive_and_zero_start_compatible():
    cfg = {
        "model": {
            "structured_memory": True,
            "token_action_policy": True,
            "exact_action_context": True,
        },
        "encoding": {
            "entity_hidden": 16,
            "state_dim": 24,
            "current_dim": 20,
            "history_dim": 20,
            "dyn_dim": 24,
            "history_cap": 12,
            "dropout": 0.0,
        },
    }
    agent = FogAgent(
        cfg,
        sizes={
            "n_species": 16,
            "n_items": 12,
            "n_abilities": 12,
            "n_moves": 24,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    ).eval()
    batch = _inputs(batch=2)
    batch["history_ids"] = torch.zeros(2, 5, DIMS.hist_id_fields, dtype=torch.long)
    batch["history_state"] = torch.zeros(2, 5, DIMS.hist_state)
    batch["legality_mask"] = torch.ones(2, 14, dtype=torch.bool)
    with torch.no_grad():
        contexts = agent.encode_contexts(batch)
        structured = agent.structured_context(contexts, "belief")
        assert structured is not None
        assert structured.shape == (2, 512)
        assert torch.isfinite(structured).all()
        base = agent(batch)

        changed = copy.deepcopy(batch)
        changed["field"][:, 0] += 1.0
        changed_structured = agent.structured_context(
            agent.encode_contexts(changed), "belief"
        )
        assert not torch.allclose(structured, changed_structured)

        # The new readers start as zero residuals, so architecture migration cannot silently
        # change a seeded policy or value before those readers are explicitly trained.
        policy_private = contexts.private_state
        public = agent.public_context(contexts)
        action = agent.action_context(contexts, batch)
        without_memory = agent.policy_value(
            policy_private,
            batch["legality_mask"],
            public,
            action_context=action,
            policy_structured_context=torch.zeros_like(structured),
            value_structured_context=torch.zeros_like(structured),
        )
        with_memory = agent.policy_value(
            policy_private,
            batch["legality_mask"],
            public,
            action_context=action,
            policy_structured_context=agent.structured_context(contexts, "policy"),
            value_structured_context=agent.structured_context(contexts, "value"),
        )
    assert torch.equal(without_memory[0], with_memory[0])
    assert torch.equal(without_memory[1], with_memory[1])
    assert torch.equal(base.log_probs, with_memory[0])
    assert torch.equal(base.value, with_memory[1])


def test_private_sheet_changes_private_context_but_not_public_context():
    backbone = _backbone()
    base = _inputs(batch=1)
    base["entities"][:, :6, L.tera_used] = 0.0
    changed = copy.deepcopy(base)
    ids = changed["entity_ids"]
    ids[:, :6, 1] = (ids[:, :6, 1] + 1) % 12
    ids[:, :6, 2] = (ids[:, :6, 2] + 2) % 12
    ids[:, :6, 3:7] = (ids[:, :6, 3:7] + 3) % 24
    ids[:, :6, NATURE_ID_FIELD] = (ids[:, :6, NATURE_ID_FIELD] + 1) % 25
    ids[:, :6, list(EV_ID_FIELDS)] = (
        ids[:, :6, list(EV_ID_FIELDS)] + 7
    ) % 64
    ids[:, :6, ORIGINAL_ITEM_ID_FIELD] = (
        ids[:, :6, ORIGINAL_ITEM_ID_FIELD] + 1
    ) % 12
    ids[:, :6, ORIGINAL_ABILITY_ID_FIELD] = (
        ids[:, :6, ORIGINAL_ABILITY_ID_FIELD] + 1
    ) % 12
    changed["entities"][:, :6, L.move_pp : L.move_pp + DIMS.move_pp] += 0.25
    changed["entities"][:, :6, L.tera_used] = 0.0
    changed["entities"][:, :6, L.tera_type : L.tera_type + DIMS.tera_type] = 0.0
    changed["entities"][:, :6, L.tera_type] = 1.0

    with torch.no_grad():
        contexts_a = backbone(**base)
        contexts_b = backbone(**changed)
        public_a = backbone.fuse_public(contexts_a.public_state, contexts_a.history)
        public_b = backbone.fuse_public(contexts_b.public_state, contexts_b.history)

    assert not torch.allclose(contexts_a.private_state, contexts_b.private_state)
    assert torch.equal(public_a, public_b)
    assert torch.equal(contexts_a.public_state, contexts_b.public_state)
    assert torch.equal(contexts_a.history, contexts_b.history)


def test_public_current_change_does_not_enter_private_encoder():
    backbone = _backbone()
    base = _inputs(batch=1)
    changed = copy.deepcopy(base)
    changed["entities"][:, 6, L.hp] += 0.5

    with torch.no_grad():
        contexts_a = backbone.encode_contexts(**base)
        contexts_b = backbone.encode_contexts(**changed)

    assert not torch.allclose(contexts_a.public_state, contexts_b.public_state)
    assert torch.equal(contexts_a.private_state, contexts_b.private_state)


def test_causal_history_prefix_cannot_see_future_turns():
    backbone = _backbone()
    history = _inputs(batch=1, turns=6)["history"]
    encoder = backbone.turn_history_encoder

    with torch.no_grad():
        prefix, _ = encoder.encode_sequence(
            history[:, :3], backbone.embeds, history_len=torch.tensor([3])
        )
        complete, _ = encoder.encode_sequence(
            history, backbone.embeds, history_len=torch.tensor([6])
        )

    assert torch.allclose(prefix[:, :3], complete[:, :3], atol=1e-6)


def test_belief_private_channel_is_zero_start_then_becomes_sensitive():
    torch.manual_seed(202)
    agent = FogAgent(
        {
            "encoding": {
                "history_encoder": "causal_transformer",
                "dropout": 0.0,
            },
            "model": {},
        }
    ).eval()
    base = _inputs(batch=1, turns=3)
    base["legality_mask"] = torch.ones(1, 14, dtype=torch.bool)
    tokens = torch.full((1, 6, 15), -1, dtype=torch.long)
    tokens[..., 0] = 1
    base["opp_revealed"] = {"tokens": tokens, "masks": {}}
    changed = copy.deepcopy(base)
    changed["entity_ids"][:, :6, 1] = (
        changed["entity_ids"][:, :6, 1] + 1
    ) % 12

    with torch.no_grad():
        output_a = agent(base)
        output_b = agent(changed)

    assert not torch.allclose(output_a.log_probs, output_b.log_probs)
    # A warm-started old checkpoint begins with identical belief behavior.
    for family in output_a.belief.logits:
        assert torch.equal(
            output_a.belief.logits[family], output_b.belief.logits[family]
        )

    # Once trained away from zero, the observer's own set can affect opponent inference.
    with torch.no_grad():
        width = min(
            agent.private_to_belief.weight.shape[0],
            agent.private_to_belief.weight.shape[1],
        )
        agent.private_to_belief.weight[:width, :width].copy_(torch.eye(width))
        output_a = agent(base)
        output_b = agent(changed)

    assert any(
        not torch.allclose(output_a.belief.logits[family], output_b.belief.logits[family])
        for family in output_a.belief.logits
    )
