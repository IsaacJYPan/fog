"""Tests for FogAgent.from_tables() — real-cardinality construction (plan §2a)."""
from __future__ import annotations

import pytest
import torch

from fog.constants.schema import N_EV_UNITS, N_NATURES
from fog.models.agent import FogAgent


class FakeTables:
    """Stands in for constants.tables.Tables with small but non-placeholder cardinalities."""

    n_species = 41
    n_items = 12
    n_abilities = 9
    n_moves = 55


@pytest.fixture()
def agent() -> FogAgent:
    torch.manual_seed(0)
    return FogAgent.from_tables(tables=FakeTables())


def test_embeddings_have_unk_row(agent):
    emb = agent.backbone.embeds
    assert emb.species.num_embeddings == FakeTables.n_species + 1
    assert emb.item.num_embeddings == FakeTables.n_items + 1
    assert emb.ability.num_embeddings == FakeTables.n_abilities + 1
    assert emb.move.num_embeddings == FakeTables.n_moves + 1
    assert emb.nature.num_embeddings == N_NATURES + 1
    assert emb.ev.num_embeddings == N_EV_UNITS + 1


def test_belief_heads_emit_raw_class_counts(agent):
    for net in (agent.belief, agent.self_belief):
        assert net.heads["species"].out_features == FakeTables.n_species
        assert net.heads["move"].out_features == FakeTables.n_moves
        assert net.sizes["nature"] == N_NATURES
        assert net.sizes["ev"] == N_EV_UNITS
        # input value-embeddings allocate their own +1 UNK row
        assert net.value_embeds["species"].num_embeddings == FakeTables.n_species + 1


def test_forward_accepts_unk_ids(agent):
    """entity_ids at exactly the UNK index (== raw table size) must be in range."""
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

    b = 2
    entity_ids = torch.zeros(b, 12, N_ID_FIELDS, dtype=torch.long)
    entity_ids[..., 0] = FakeTables.n_species      # UNK species
    entity_ids[..., 1] = FakeTables.n_items        # UNK item
    entity_ids[..., 2] = FakeTables.n_abilities
    entity_ids[..., 3:8] = FakeTables.n_moves      # 4 move slots + last_used_move
    entity_ids[..., 8] = N_NATURES
    entity_ids[..., 9:15] = N_EV_UNITS
    legality = torch.zeros(b, 14, dtype=torch.bool)
    legality[:, 13] = True
    tokens = torch.full((b, 6, 15), -1, dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, FakeTables.n_species, (b, 6))
    batch = {
        "entities": torch.randn(b, 12, ENTITY_RAW_DIM),
        "entity_ids": entity_ids,
        "field": torch.randn(b, DIMS.field_dim),
        "history": torch.randn(b, 5, DIMS.turn_dim),
        "active_idx": torch.tensor([[0, 6]] * b),
        "legality_mask": legality,
        "opp_revealed": {"tokens": tokens, "masks": None},
    }
    out = agent(batch)
    assert out.log_probs.shape == (b, 14)
    assert out.belief.logits["species"].shape == (b, 6, FakeTables.n_species)


def test_real_generated_tables_if_present():
    """Integration: with the real generated tables, from_tables stays within budget."""
    from fog.constants.tables import TABLES, TablesNotBuilt

    try:
        n = TABLES.n_species
    except TablesNotBuilt:
        pytest.skip("generated tables not built")
    agent = FogAgent.from_tables()
    assert agent.backbone.embeds.species.num_embeddings == n + 1
    agent.assert_within_budget()


def test_compatible_loader_is_explicit_about_every_tensor(agent):
    state = agent.state_dict()
    names = list(state)
    loaded_name, mismatched_name = names[0], names[1]
    checkpoint = {
        loaded_name: torch.full_like(state[loaded_name], 0.125),
        mismatched_name: torch.zeros(1),
        "obsolete.decoder.weight": torch.zeros(2, 2),
    }

    manifest = agent.load_compatible_state_dict(checkpoint)

    assert loaded_name in manifest["loaded"]
    assert torch.equal(agent.state_dict()[loaded_name], checkpoint[loaded_name])
    assert mismatched_name in manifest["mismatched"]
    assert mismatched_name in manifest["initialized"]
    assert manifest["obsolete"] == ["obsolete.decoder.weight"]
