"""Native nature/EV belief-schema regression tests."""
from __future__ import annotations

import numpy as np
import torch

from fog.constants.schema import (
    MAX_TOTAL_EV_UNITS,
    N_EV_UNITS,
    N_NATURES,
    NATURE_INDEX,
    SET_EV_COLS,
    SET_NATURE_COL,
    SET_TERA_COL,
    TYPE_INDEX,
)
from fog.constants.tables import TABLES
from fog.encoding.entities import ENTITY_RAW_DIM, L
from fog.encoding.tracker import BattleTracker
from fog.env.engine_battle import _decode_world_slot
from fog.env.team_truth import truth_tokens_from_paste
from fog.models.agent import FogAgent
from fog.models.belief import BeliefModel
from fog.models.encoder import Backbone
from fog.training.dataset import _normalize_entities, _normalize_entity_ids, _normalize_set_tokens

PASTE = """Great Tusk
Ability: Protosynthesis
Tera Type: Ground
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Earthquake
- Stealth Rock
"""


def test_truth_and_engine_world_round_trip_native_nature_evs():
    truth = truth_tokens_from_paste(PASTE, TABLES)
    row = truth[0]
    assert row[SET_NATURE_COL] == NATURE_INDEX["careful"]
    assert row[list(SET_EV_COLS)].tolist() == [63, 1, 0, 0, 63, 0]
    assert row[SET_TERA_COL] == TYPE_INDEX["ground"]

    decoded = _decode_world_slot(row, 100, TABLES)
    assert decoded.nature == "careful"
    assert decoded.evs == [252, 4, 0, 0, 252, 0]
    assert decoded.tera_type == "ground"


def test_tracker_retains_exact_private_nature_evs_across_clone():
    truth = truth_tokens_from_paste(PASTE, TABLES)
    tracker = BattleTracker("p1")
    tracker.record_exact_private_truth(truth)
    species = int(truth[0, 0])
    assert tracker.exact_private_truth[species][SET_NATURE_COL] == NATURE_INDEX["careful"]
    assert tracker.exact_private_truth[species][list(SET_EV_COLS)].tolist() == [
        63, 1, 0, 0, 63, 0
    ]
    clone = tracker.clone()
    clone.exact_private_truth[species][SET_NATURE_COL] = NATURE_INDEX["timid"]
    assert tracker.exact_private_truth[species][SET_NATURE_COL] == NATURE_INDEX["careful"]


def test_ev_query_masks_values_beyond_remaining_team_sheet_budget():
    model = BeliefModel(
        state_dim=16,
        d=16,
        n_species=3,
        n_items=4,
        n_abilities=3,
        n_moves=8,
        n_natures=N_NATURES,
        n_ev_units=N_EV_UNITS,
    ).eval()
    public = torch.full((1, 6, 15), -1, dtype=torch.long)
    public[:, :, 0] = 0
    context = torch.full_like(public, -1)
    context[0, 0, 8:13] = torch.tensor([63, 63, 1, 0, 0])

    logits = model.query_field(
        torch.zeros(1, 16),
        {"tokens": public, "masks": None},
        slot=0,
        field="ev",
        subslot=5,
        context_values=context,
    )
    assert torch.isfinite(logits[0, 0])
    assert torch.isneginf(logits[0, 1:]).all()


def test_native_sampler_never_exceeds_effective_ev_budget():
    model = BeliefModel(
        state_dim=16,
        d=16,
        n_species=3,
        n_items=4,
        n_abilities=3,
        n_moves=8,
        n_natures=N_NATURES,
        n_ev_units=N_EV_UNITS,
    ).eval()
    public = torch.full((1, 6, 15), -1, dtype=torch.long)
    public[:, :, 0] = 0
    samples = model.sample_team(
        torch.zeros(1, 16),
        {"tokens": public, "masks": None},
        k=16,
        generator=torch.Generator().manual_seed(7),
    )
    evs = samples[..., 8:14]
    assert (evs >= 0).all() and (evs < N_EV_UNITS).all()
    assert (evs.sum(dim=-1) <= MAX_TOTAL_EV_UNITS).all()


def test_legacy_replay_migration_discards_cluster_and_preserves_tera():
    old_tokens = np.full((2, 6, 9), -1, dtype=np.int64)
    old_tokens[..., 0] = 2
    old_tokens[..., 7] = 5
    old_tokens[..., 8] = 7
    migrated = _normalize_set_tokens(old_tokens)
    assert migrated.shape == (2, 6, 15)
    assert (migrated[..., 0] == 2).all()
    assert (migrated[..., 7:14] == -1).all()
    assert (migrated[..., 14] == 7).all()

    old_ids = np.zeros((2, 12, 9), dtype=np.int64)
    old_ids[..., 8] = 5
    migrated_ids = _normalize_entity_ids(old_ids)
    assert migrated_ids.shape == (2, 12, 17)
    assert (migrated_ids[..., 8] == N_NATURES).all()
    assert (migrated_ids[..., 9:15] == N_EV_UNITS).all()


def test_legacy_replay_entity_migration_discards_cluster_flag_and_preserves_tera():
    old_width = ENTITY_RAW_DIM - 6
    old = np.zeros((2, 12, old_width), dtype=np.float32)
    old[..., L.epistemic : L.epistemic + 6] = 1
    old[..., L.epistemic + 6] = 1  # legacy spread cluster: not exact native truth
    old[..., L.epistemic + 7] = 1  # Tera
    old[..., L.epistemic + 8 :] = 0.5
    migrated = _normalize_entities(old)
    assert migrated.shape[-1] == ENTITY_RAW_DIM
    assert (migrated[..., L.epistemic : L.epistemic + 6] == 1).all()
    assert (migrated[..., L.epistemic + 6 : L.epistemic + 13] == 0).all()
    assert (migrated[..., L.epistemic + 13] == 1).all()
    assert (migrated[..., L.epistemic + 14 :] == 0.5).all()


def test_backbone_input_width_stays_compatible_with_cluster_checkpoint():
    backbone = Backbone()
    old_embedding_width = 64 + 8 * 24
    assert backbone.embeds.out_dim == old_embedding_width
    assert backbone.entity_in.in_features == ENTITY_RAW_DIM + old_embedding_width
    assert torch.count_nonzero(backbone.embeds.native_stats[-1].weight) == 0


def test_compatible_loader_inserts_native_flags_without_discarding_board_projection():
    agent = FogAgent()
    current = agent.backbone.entity_in.weight
    old_raw_dim = ENTITY_RAW_DIM - 6
    old = torch.arange(
        current.shape[0] * (old_raw_dim + agent.backbone.embeds.out_dim),
        dtype=current.dtype,
    ).reshape(current.shape[0], -1)
    manifest = agent.load_compatible_state_dict({"backbone.entity_in.weight": old})
    loaded = agent.backbone.entity_in.weight.detach()
    ep = L.epistemic

    assert "backbone.entity_in.weight" in manifest["migrated"]
    assert torch.equal(loaded[:, :ep], old[:, :ep])
    assert torch.equal(loaded[:, ep : ep + 6], old[:, ep : ep + 6])
    assert torch.count_nonzero(loaded[:, ep + 6 : ep + 13]) == 0
    assert torch.equal(loaded[:, ep + 13], old[:, ep + 7])
    assert torch.equal(loaded[:, ep + 14 : ENTITY_RAW_DIM], old[:, ep + 8 : old_raw_dim])
    assert torch.equal(loaded[:, ENTITY_RAW_DIM:], old[:, old_raw_dim:])
