import torch

from fog.scripts.diagnose_joint_gradient_alignment import _cosine, _group


def test_gradient_groups_cover_shared_curriculum_paths():
    assert _group("backbone.embeds.item.weight") == "backbone_embeddings"
    assert _group("backbone.turn_history_encoder.head.0.weight") == "history_encoder"
    assert _group("shared_latent.contextual_encoder.layers.0.linear1.weight") == "shared_transformer"
    assert _group("shared_latent.entity_encoder.0.weight") == "shared_input_routes"
    assert _group("shared_to_belief.output.0.weight") == "belief_reader"
    assert _group("shared_to_value.output.0.weight") == "value_reader"
    assert _group("policy_value.policy.0.weight") is None


def test_gradient_cosine_handles_alignment_conflict_and_zero():
    vector = torch.tensor([1.0, 2.0])
    assert _cosine(vector, vector) == 1.0
    assert _cosine(vector, -vector) == -1.0
    assert _cosine(vector, torch.zeros_like(vector)) == 0.0
