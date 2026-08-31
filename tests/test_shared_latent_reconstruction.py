from types import SimpleNamespace

import torch

from fog.encoding.entities import L
from fog.training.shared_latent_reconstruction import (
    SharedLatentDecoder,
    _checkpoint_selection_score,
    _denoised_batch,
    _flat_direct_input_latent,
    _loss,
    _raw_loss,
    _simple_mse_warmup_loss,
    _weakest_components,
)


def test_checkpoint_selection_uses_loss_during_mse_warmup_then_strict_gate():
    team = {"loss": 0.4}
    battle = {"loss": 0.6}
    phase, score = _checkpoint_selection_score(
        epoch_index=2,
        mse_warmup_epochs=5,
        team=team,
        battle=battle,
        recovery_score=0.0,
    )
    assert phase == "mse_warmup"
    assert score == -0.5

    phase, score = _checkpoint_selection_score(
        epoch_index=5,
        mse_warmup_epochs=5,
        team=team,
        battle=battle,
        recovery_score=0.73,
    )
    assert phase == "strict_recovery"
    assert score == 0.73


def test_linear_flat_identity_control_copies_packed_values():
    from fog.training.shared_latent_reconstruction import SharedLatentDecoder

    decoder = SharedLatentDecoder(
        128,
        [3, 4],
        slot_dim=8,
        raw_dim=2,
        field_dim=3,
        flat=True,
        simple_flat=True,
        linear_flat=True,
        linear_exact_values=True,
    )
    decoder.initialize_linear_flat_identity_control()
    packed_width = 12 * 2 + 12 * 2 + 3 + 14 + 12
    source = torch.randn(2, 128)
    result = decoder(source)
    expected = source[:, :packed_width]
    actual = torch.cat(
        [
            result["raw"].flatten(1),
            result["id_values"].flatten(1),
            result["field"],
            result["legality"],
            result["active_mine"],
            result["active_opp"],
        ],
        dim=-1,
    )
    assert torch.equal(actual, expected)


def test_flat_direct_control_packs_every_exact_current_value_once():
    class Trunk:
        flat_autoencoder = True
        latent_dim = 2048

        @staticmethod
        def normalized_ids(ids, _embeds):
            return ids.float() / 10.0

    agent = SimpleNamespace(
        shared_latent=Trunk(),
        backbone=SimpleNamespace(embeds=object()),
    )
    batch = {
        "entities": torch.randn(2, 12, 140),
        "entity_ids": torch.randint(0, 11, (2, 12, 17)),
        "field": torch.randn(2, 51),
        "active_idx": torch.tensor([[0, 6], [5, 11]]),
        "legality_mask": torch.randint(0, 2, (2, 14)).bool(),
    }
    latent = _flat_direct_input_latent(agent, batch)
    assert latent.shape == (2, 2048)
    expected = torch.cat(
        [
            batch["entities"].flatten(1),
            (batch["entity_ids"].float() / 10.0).flatten(1),
            batch["field"],
            torch.nn.functional.one_hot(batch["active_idx"][:, 0], 12)
            + torch.nn.functional.one_hot(batch["active_idx"][:, 1], 12),
            batch["legality_mask"].float(),
        ],
        dim=-1,
    )
    torch.testing.assert_close(latent[:, : expected.shape[-1]], expected)
    assert torch.count_nonzero(latent[:, expected.shape[-1] :]) == 0


def test_decoder_receives_only_latent_and_recovers_expected_shapes():
    decoder = SharedLatentDecoder(256, [11] * 17, slot_dim=16, latent_entity_dim=16)
    predicted = decoder(torch.randn(3, 256))
    assert predicted["raw"].shape == (3, 12, 140)
    assert len(predicted["ids"]) == 17
    assert predicted["ids"][0].shape == (3, 12, 11)
    assert predicted["id_values"].shape == (3, 12, 17)
    assert predicted["field"].shape == (3, 51)
    assert predicted["legality"].shape == (3, 14)


def test_noncontextual_decoder_creature_outputs_use_only_corresponding_token():
    """The local auxiliary cannot hide a creature's facts in another/global token."""
    decoder = SharedLatentDecoder(
        24 * 16,
        [11] * 17,
        slot_dim=16,
        latent_entity_dim=16,
        contextual=False,
    )
    original = torch.randn(2, 24 * 16)
    changed = original.clone()
    # Change every token except entity slot 3.
    changed[:, : 3 * 16] = torch.randn_like(changed[:, : 3 * 16])
    changed[:, 4 * 16 :] = torch.randn_like(changed[:, 4 * 16 :])
    before = decoder(original)
    after = decoder(changed)
    torch.testing.assert_close(before["raw"][:, 3], after["raw"][:, 3])
    torch.testing.assert_close(before["id_values"][:, 3], after["id_values"][:, 3])
    for before_ids, after_ids in zip(before["ids"], after["ids"]):
        torch.testing.assert_close(before_ids[:, 3], after_ids[:, 3])
    # Global outputs are intentionally allowed to use the complete latent.
    assert not torch.equal(before["field"], after["field"])


def test_reconstruction_objective_is_finite():
    decoder = SharedLatentDecoder(256, [11] * 17, slot_dim=16, latent_entity_dim=16)
    predicted = decoder(torch.randn(2, 256))
    batch = {
        "entities": torch.randn(2, 12, 140),
        "entity_ids": torch.randint(0, 11, (2, 12, 17)),
        "field": torch.randn(2, 51),
        "legality_mask": torch.randint(0, 2, (2, 14)).bool(),
        "active_idx": torch.tensor([[0, 6], [5, 11]]),
    }
    loss = _loss(predicted, batch, id_weight=2.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_code_scaled_identity_loss_rewards_exact_integer_roundtrip():
    decoder = SharedLatentDecoder(256, [11] * 17, slot_dim=16, latent_entity_dim=16)
    batch = {
        "entities": torch.zeros(2, 12, 140),
        "entity_ids": torch.randint(0, 11, (2, 12, 17)),
        "field": torch.zeros(2, 51),
        "legality_mask": torch.zeros(2, 14, dtype=torch.bool),
        "active_idx": torch.tensor([[0, 6], [5, 11]]),
    }
    predicted = decoder(torch.zeros(2, 256))
    exact = dict(predicted)
    exact["id_values"] = batch["entity_ids"].float() / 10.0
    wrong = dict(predicted)
    wrong["id_values"] = torch.zeros_like(exact["id_values"])
    kwargs = {
        "id_weight": 0.0,
        "id_regression_weight": 1.0,
        "raw_weight": 0.0,
        "field_weight": 0.0,
        "linear_id_values": True,
        "code_scaled_identities": True,
    }
    assert _loss(exact, batch, **kwargs) < _loss(wrong, batch, **kwargs)


def test_simple_mse_warmup_rewards_exact_flat_reconstruction():
    decoder = SharedLatentDecoder(256, [11] * 17, slot_dim=16, latent_entity_dim=16)
    batch = {
        "entities": torch.zeros(2, 12, 140),
        "entity_ids": torch.randint(0, 11, (2, 12, 17)),
        "field": torch.zeros(2, 51),
        "legality_mask": torch.zeros(2, 14, dtype=torch.bool),
        "active_idx": torch.tensor([[0, 6], [5, 11]]),
    }
    predicted = decoder(torch.zeros(2, 256))
    exact = dict(predicted)
    exact["raw"] = batch["entities"]
    exact["id_values"] = batch["entity_ids"].float() / 10.0
    exact["field"] = batch["field"]
    wrong = dict(exact)
    wrong["raw"] = torch.ones_like(exact["raw"])
    wrong["id_values"] = torch.zeros_like(exact["id_values"])
    wrong["field"] = torch.ones_like(exact["field"])
    assert _simple_mse_warmup_loss(exact, batch) < _simple_mse_warmup_loss(
        wrong, batch
    )


def test_flat_decoder_reads_every_slot_from_the_complete_latent():
    decoder = SharedLatentDecoder(
        256,
        [11] * 17,
        slot_dim=16,
        latent_entity_dim=16,
        flat=True,
    )
    latent = torch.randn(2, 256, requires_grad=True)
    predicted = decoder(latent)
    assert predicted["raw"].shape == (2, 12, 140)
    predicted["raw"].square().mean().backward()
    assert latent.grad is not None
    assert decoder.flat_slot_projection[0].weight.grad is not None


def test_simple_flat_decoder_reconstructs_one_packed_state_without_tokens():
    decoder = SharedLatentDecoder(
        256,
        [11] * 17,
        slot_dim=16,
        latent_entity_dim=16,
        flat=True,
        simple_flat=True,
        linear_exact_values=True,
    )
    latent = torch.randn(2, 256, requires_grad=True)
    predicted = decoder(latent)
    assert predicted["raw"].shape == (2, 12, 140)
    assert predicted["id_values"].shape == (2, 12, 17)
    assert predicted["field"].shape == (2, 51)
    assert predicted["legality"].shape == (2, 14)
    assert predicted["active_mine"].shape == (2, 6)
    assert predicted["active_opp"].shape == (2, 6)
    predicted["raw"].square().mean().backward()
    assert latent.grad is not None
    assert decoder.simple_flat_projection[0].weight.grad is not None


def test_simple_flat_decoder_can_recover_identities_as_exact_categories():
    decoder = SharedLatentDecoder(
        256,
        [11] * 17,
        slot_dim=16,
        latent_entity_dim=16,
        flat=True,
        simple_flat=True,
        categorical_identity_heads=True,
        linear_exact_values=True,
    )
    latent = torch.randn(2, 256, requires_grad=True)
    predicted = decoder(latent)
    assert len(predicted["ids"]) == 17
    assert predicted["ids"][0].shape == (2, 12, 11)
    predicted["ids"][0].square().mean().backward()
    assert latent.grad is not None
    assert decoder.simple_flat_identity_slots.weight.grad is not None
    assert decoder.simple_flat_ids[0].weight.grad is not None


def test_linear_flat_decoder_is_a_single_global_reconstruction_map():
    decoder = SharedLatentDecoder(
        256,
        [11] * 17,
        slot_dim=16,
        latent_entity_dim=16,
        flat=True,
        simple_flat=True,
        linear_flat=True,
        linear_exact_values=True,
    )
    assert isinstance(decoder.simple_flat_projection, torch.nn.Linear)
    latent = torch.randn(2, 256, requires_grad=True)
    predicted = decoder(latent)
    predicted["raw"].square().mean().backward()
    assert latent.grad is not None
    assert decoder.simple_flat_projection.weight.grad is not None


def test_reconstruction_gate_is_defined_per_input_field():
    # A high aggregate score can hide one erased input column. The production evaluator records
    # and gates the minimum recovery across all 140 creature and 51 field columns.
    source = __import__(
        "inspect"
    ).getsource(__import__(
        "fog.training.shared_latent_reconstruction", fromlist=["run"]
    ).run)
    assert "raw_minimum_field_recovery" not in source  # constructed dynamically
    assert 'report[f"{name}_minimum_field_recovery"]' in source
    assert 'report["legality_row_accuracy"]' in source
    assert 'report["raw_semantic_minimum_recovery"]' in source
    assert 'report["field_semantic_minimum_recovery"]' in source


def test_denoising_changes_inputs_without_mutating_reconstruction_targets():
    batch = {
        "entities": torch.ones(2, 12, 140),
        "entity_ids": torch.ones(2, 12, 17, dtype=torch.long),
        "field": torch.ones(2, 51),
        "legality_mask": torch.ones(2, 14, dtype=torch.bool),
    }
    original = {key: value.clone() for key, value in batch.items()}
    corrupted = _denoised_batch(
        batch,
        fraction=0.5,
        generator=torch.Generator().manual_seed(20261106),
    )

    assert (corrupted["entities"] == 0).any()
    assert (corrupted["entity_ids"] == 0).any()
    assert (corrupted["field"] == 0).any()
    for key, value in batch.items():
        torch.testing.assert_close(value, original[key])
    assert corrupted["legality_mask"] is batch["legality_mask"]


def test_weakest_component_report_names_the_actual_limiters():
    report = {
        "raw_semantic_recovery": {"hp": [0.4], "boosts": [0.9, 0.8]},
        "field_semantic_recovery_by_field": [0.7, 0.6],
        "id_accuracy_by_field": [0.95] * 17,
        "legality_bit_accuracy": 0.99,
        "legality_row_accuracy": 0.5,
        "active_mine_accuracy": 1.0,
        "active_opp_accuracy": 1.0,
    }
    weakest = _weakest_components(report, limit=3)
    assert weakest == [
        {"component": "raw.hp[0]", "recovery": 0.4},
        {"component": "legality.row", "recovery": 0.5},
        {"component": "field[1]", "recovery": 0.6},
    ]


def test_priority_raw_blocks_increase_pressure_on_named_facts():
    logits = torch.zeros(2, 12, 140)
    target = torch.zeros_like(logits)
    # HP is one scalar column; make it the only reconstruction error.
    target[..., L.hp] = 1.0
    ordinary = _raw_loss(logits, target, linear_exact_values=True)
    prioritized = _raw_loss(
        logits,
        target,
        linear_exact_values=True,
        priority_blocks=("hp",),
        priority_weight=10.0,
    )
    assert prioritized > ordinary
