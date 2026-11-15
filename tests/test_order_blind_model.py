"""Permutation checks use noisy weights, real holdout rows and live Showdown decisions.

The large corpus is optional for a fresh clone; the holdout case runs when the brief's fixed
split exists (or FOG_ORDER_BLIND_HOLDOUT names it). The live case needs the pinned Node build.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fog.config import load_config
from fog.encoding.dims import DIMS
from fog.encoding.entities import L
from fog.encoding.events import H_MY_MOVE, H_MY_SWITCH_SPECIES, H_START_MY_SPECIES
from fog.models.agent import FogAgent


def _copy(batch):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _history_actions_by_identity(batch):
    """Remap historical move/switch slots using sidecars, keeping action kind and identity."""
    history = batch["history"]
    ids = batch["history_ids"]
    own = history[..., DIMS.hist_snapshot : DIMS.hist_snapshot + 14]
    for row in range(len(history)):
        species = batch["entity_ids"][row, :6, 0]
        for turn in range(history.shape[1]):
            if not own[row, turn].any():
                continue
            old = int(own[row, turn].argmax())
            active = (species == ids[row, turn, H_START_MY_SPECIES]).nonzero().flatten()
            if len(active) != 1:
                continue
            active = int(active[0])
            if old < 8:
                moves = batch["entity_ids"][row, active, 3:7]
                target = (moves == ids[row, turn, H_MY_MOVE]).nonzero().flatten()
                new = int(target[0]) + (4 if old >= 4 else 0) if len(target) else old
            elif old < 13:
                bench = [i for i in range(6) if i != active]
                target = (species[bench] == ids[row, turn, H_MY_SWITCH_SPECIES]).nonzero().flatten()
                new = 8 + int(target[0]) if len(target) else old
            else:
                new = 13
            own[row, turn] = 0
            own[row, turn, new] = 1


def _permute_moves(batch, slots, generator):
    out = _copy(batch)
    count = len(slots)
    rows = torch.arange(count)
    perm = torch.rand(count, 4, generator=generator).argsort(-1)
    ids = out["entity_ids"][rows, slots].clone()
    ids[:, 3:7] = ids[:, 3:7].gather(1, perm)
    out["entity_ids"][rows, slots] = ids
    raw = out["entities"][rows, slots].clone()
    for offset in (L.move_pp, L.epistemic + 2):
        raw[:, offset : offset + 4] = raw[:, offset : offset + 4].gather(1, perm)
    out["entities"][rows, slots] = raw
    actions = torch.arange(14).expand(count, -1).clone()
    acting = slots == batch["active_idx"][:, 0]
    actions[acting, :4] = perm[acting]
    actions[acting, 4:8] = perm[acting] + 4
    out["legality_mask"] = batch["legality_mask"].gather(1, actions)
    _history_actions_by_identity(out)
    if isinstance(batch.get("action_features"), torch.Tensor):
        width = batch["action_features"].shape[-1]
        out["action_features"] = batch["action_features"].gather(
            1, actions.unsqueeze(-1).expand(-1, -1, width))
    return out, actions


def _permute_team(batch, side, generator):
    out = _copy(batch)
    count = len(batch["entities"])
    perm = torch.rand(count, 6, generator=generator).argsort(-1)
    offset = 6 * side
    for key in ("entities", "entity_ids"):
        old = batch[key][:, offset : offset + 6]
        out[key][:, offset : offset + 6] = old.gather(
            1, perm.unsqueeze(-1).expand(-1, -1, old.shape[-1])
        )
    # Re-encode the arbitrary position tag just as the observation encoder would.
    out["entities"][:, offset : offset + 6, L.party_slot : L.party_slot + 6] = torch.eye(6)
    active = batch["active_idx"][:, side] - offset
    out["active_idx"][:, side] = perm.argsort(1).gather(1, active[:, None])[:, 0] + offset
    actions = torch.arange(14).expand(count, -1).clone()
    if side == 0:
        slots = torch.arange(6).expand(count, -1)
        old_bench = slots[slots != active[:, None]].reshape(count, 5)
        new_active = out["active_idx"][:, 0]
        new_bench = slots[slots != new_active[:, None]].reshape(count, 5)
        targets = perm.gather(1, new_bench)
        actions[:, 8:13] = 8 + (targets[:, :, None] == old_bench[:, None, :]).long().argmax(-1)
        out["legality_mask"] = batch["legality_mask"].gather(1, actions)
        _history_actions_by_identity(out)
    if isinstance(batch.get("action_features"), torch.Tensor):
        width = batch["action_features"].shape[-1]
        out["action_features"] = batch["action_features"].gather(
            1, actions.unsqueeze(-1).expand(-1, -1, width))
    return out, actions


@pytest.fixture(scope="module", params=["synthetic", "holdout", "live"])
def observations(request):
    torch.set_num_threads(2)
    if request.param == "holdout":
        from fog.training.dataset import StreamingReplayDataset

        manifest = Path(
            os.environ.get(
                "FOG_ORDER_BLIND_HOLDOUT", "runs/blueprint/split/holdout_manifest.parquet"
            )
        )
        if not manifest.exists():
            pytest.skip("fixed replay holdout not present; run the brief data preparation")
        batch = next(StreamingReplayDataset(manifest).batches(64, shuffle=True, seed=79))
    elif request.param == "live":
        from fog.eval.showdown_rollout_probe import run_probe
        from fog.training.rollout import _stack_frags

        if not Path("server/pokemon-showdown/dist/sim/battle.js").exists():
            pytest.skip("pinned Showdown build not present")
        result = run_probe(
            "configs/gen9ou_typed_token_selfplay.yaml",
            "data/teams/pool_b",
            6,
            0,
            threads=2,
            batch=6,
        )
        frags = [f for game in result["buffer"]._games.values() for f in game.frags]
        assert len(frags) >= 64
        batch = _stack_frags([frags[i] for i in np.linspace(0, len(frags) - 1, 64, dtype=int)])
    else:
        from fog.constants.tables import TABLES

        g = torch.Generator().manual_seed(481)
        from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

        batch = {
            "entities": torch.rand(64, 12, ENTITY_RAW_DIM, generator=g),
            "entity_ids": torch.zeros(64, 12, N_ID_FIELDS, dtype=torch.long),
            "field": torch.rand(64, DIMS.field_dim, generator=g),
            "history": torch.rand(64, 8, DIMS.turn_dim, generator=g),
            "history_ids": torch.zeros(64, 8, DIMS.hist_id_fields, dtype=torch.long),
            "history_state": torch.rand(64, 8, DIMS.hist_state, generator=g),
            "history_len": torch.full((64,), 8),
            "active_idx": torch.stack(
                [torch.randint(6, (64,), generator=g), torch.randint(6, (64,), generator=g) + 6],
                dim=1,
            ),
            "legality_mask": torch.rand(64, 14, generator=g) > 0.3,
        }
        batch["legality_mask"][:, 0] = True
        batch["entity_ids"][..., 0] = torch.randint(TABLES.n_species, (64, 12), generator=g)
        batch["entity_ids"][..., 3:8] = torch.randint(TABLES.n_moves, (64, 12, 5), generator=g)
        batch["history"][..., 30:44] = torch.nn.functional.one_hot(
            torch.randint(14, (64, 8), generator=g), 14
        ).float()
    assert len(batch["entities"]) == 64
    return batch


def _agent(on, **options):
    cfg = load_config("configs/replay_blueprint_cpu.yaml")
    cfg["model"].update(order_blind_entities=on, **options)
    torch.manual_seed(17)
    agent = FogAgent.from_tables(cfg).eval()
    # Every parameter, including zero-initialized output layers and embeddings, must change.
    with torch.no_grad():
        for parameter in agent.parameters():
            before = parameter.clone()
            parameter.add_(0.02 * torch.randn_like(parameter))
            assert not torch.equal(before, parameter)
    return agent


def _permutations(batch):
    generator = torch.Generator().manual_seed(81)
    yield "acting moves", _permute_moves(batch, batch["active_idx"][:, 0], generator)
    yield "other own moves", _permute_moves(batch, (batch["active_idx"][:, 0] + 1) % 6, generator)
    yield (
        "other opponent moves",
        _permute_moves(
            batch, 6 + torch.randint(6, (len(batch["entities"]),), generator=generator), generator
        ),
    )
    yield "own team", _permute_team(batch, 0, generator)
    yield "opponent team", _permute_team(batch, 1, generator)


def _forward(agent, batch):
    logits = []
    handle = agent.policy_value.action_query_scorer.register_forward_hook(
        lambda module, args, output: logits.append(output.squeeze(-1))
    )
    try:
        result = agent(batch, need_belief=False)
    finally:
        handle.remove()
    return result, logits[0]


def _record(request, on, name, logit_error, probability_error, value_error):
    destination = os.environ.get("FOG_ORDER_BLIND_TEST_REPORT")
    if destination:
        path = Path(destination)
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append(
            dict(
                source=request.node.callspec.params["observations"],
                observations=64,
                all_parameters_perturbed=True,
                key_on=on,
                permutation=name,
                max_logit_error=logit_error,
                max_log_probability_error=probability_error,
                max_value_error=value_error,
            )
        )
        path.write_text(json.dumps(rows, indent=2) + "\n")


@torch.no_grad()
def test_order_blind_model_all_four_permutations(observations, request):
    agent = _agent(True)
    baseline, logits = _forward(agent, observations)
    for name, (changed, actions) in _permutations(observations):
        result, permuted_logits = _forward(agent, changed)
        expected = baseline.log_probs.gather(1, actions)
        finite = torch.isfinite(expected) & torch.isfinite(result.log_probs)
        errors = (
            float((permuted_logits - logits.gather(1, actions)).abs().max()),
            float((result.log_probs[finite] - expected[finite]).abs().max()),
            float((result.value - baseline.value).abs().max()),
        )
        _record(request, True, name, *errors)
        torch.testing.assert_close(
            permuted_logits, logits.gather(1, actions), atol=1e-5, rtol=0, msg=name
        )
        torch.testing.assert_close(result.log_probs, expected, atol=1e-5, rtol=0, msg=name)
        torch.testing.assert_close(result.value, baseline.value, atol=1e-5, rtol=0, msg=name)


@torch.no_grad()
def test_key_off_fails_acting_moves_and_bench(observations, request):
    agent = _agent(False)
    baseline, logits = _forward(agent, observations)
    for name, (changed, actions) in _permutations(observations):
        if name not in ("acting moves", "own team"):
            continue
        result, permuted_logits = _forward(agent, changed)
        expected = baseline.log_probs.gather(1, actions)
        finite = torch.isfinite(expected) & torch.isfinite(result.log_probs)
        errors = (
            float((permuted_logits - logits.gather(1, actions)).abs().max()),
            float((result.log_probs[finite] - expected[finite]).abs().max()),
            float((result.value - baseline.value).abs().max()),
        )
        _record(request, False, name, *errors)
        assert errors[0] > 1e-5 and errors[1] > 1e-5, name


def test_order_blind_rejects_ordered_architectures():
    cfg = load_config("configs/replay_order_blind_cpu.yaml")
    cfg["model"]["structured_memory"] = True
    with pytest.raises(ValueError, match="order_blind_entities requires"):
        FogAgent.from_tables(cfg)


@torch.no_grad()
def test_optional_global_and_dynamic_inputs_are_also_order_blind(observations):
    agent = _agent(True, shared_contextual_data_global=True,
                   shared_contextual_dynamic_residual=True)
    baseline = agent(observations, need_belief=False)
    for name, (changed, actions) in _permutations(observations):
        result = agent(changed, need_belief=False)
        torch.testing.assert_close(result.log_probs, baseline.log_probs.gather(1, actions),
                                   atol=1e-5, rtol=0, msg=name)
        torch.testing.assert_close(result.value, baseline.value, atol=1e-5, rtol=0, msg=name)


@torch.no_grad()
def test_mechanical_model_remains_order_blind(observations):
    agent = _agent(True, mechanical_features=True)
    baseline, logits = _forward(agent, observations)
    for name, (changed, actions) in _permutations(observations):
        result, permuted_logits = _forward(agent, changed)
        torch.testing.assert_close(permuted_logits, logits.gather(1, actions),
                                   atol=1e-5, rtol=0, msg=name)
        torch.testing.assert_close(result.log_probs, baseline.log_probs.gather(1, actions),
                                   atol=1e-5, rtol=0, msg=name)
        torch.testing.assert_close(result.value, baseline.value, atol=1e-5, rtol=0, msg=name)
