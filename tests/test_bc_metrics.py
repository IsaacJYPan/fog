from types import SimpleNamespace

import polars as pl
import pytest
import torch
from torch import nn

from fog.training import bc
from fog.training.bc import (
    _average_metrics,
    _concatenate_batches,
    _equal_norm_pcgrad_pair,
    _pcgrad_many,
    _pcgrad_pair,
)


def test_bc_epoch_metrics_are_row_weighted_and_counts_are_summed():
    result = _average_metrics(
        [
            (2, {"loss": 4.0, "belief_chain_valid_count": 2.0}),
            (6, {"loss": 2.0, "belief_chain_valid_count": 6.0}),
        ]
    )

    assert result["loss"] == pytest.approx(2.5)
    assert result["belief_chain_valid_count"] == 8.0
    assert result["belief_chain_dropped_rate"] == 0.0


def test_bc_epoch_metrics_report_imputation_sensitivity():
    result = _average_metrics(
        [
            (
                4,
                {
                    "imputation_0_valid_count": 4.0,
                    "imputation_0_correct_count": 3.0,
                    "imputation_0_nll_sum_count": 2.0,
                },
            ),
            (
                6,
                {
                    "imputation_1_valid_count": 6.0,
                    "imputation_1_correct_count": 3.0,
                    "imputation_1_nll_sum_count": 6.0,
                },
            ),
        ]
    )

    assert result["imputation_0_top1"] == pytest.approx(0.75)
    assert result["imputation_1_top1"] == pytest.approx(0.5)
    assert result["imputation_top1_range"] == pytest.approx(0.25)
    assert result["imputation_0_nll"] == pytest.approx(0.5)
    assert result["imputation_1_nll"] == pytest.approx(1.0)
    assert result["imputation_nll_range"] == pytest.approx(0.5)


def test_concatenate_batches_recurses_and_preserves_row_order():
    primary = {
        "entities": torch.tensor([[1.0], [2.0]]),
        "belief_valid": {"item": torch.tensor([[True], [False]])},
    }
    auxiliary = {
        "entities": torch.tensor([[3.0]]),
        "belief_valid": {"item": torch.tensor([[True]])},
    }

    combined = _concatenate_batches(primary, auxiliary)

    assert combined["entities"].tolist() == [[1.0], [2.0], [3.0]]
    assert combined["belief_valid"]["item"].tolist() == [[True], [False], [True]]


def test_concatenate_batches_front_pads_variable_history():
    primary = {
        "history": torch.tensor([[[1.0], [2.0]]]),
        "history_len": torch.tensor([2]),
    }
    auxiliary = {
        "history": torch.tensor([[[3.0], [4.0], [5.0]]]),
        "history_len": torch.tensor([3]),
    }

    combined = _concatenate_batches(primary, auxiliary)

    assert combined["history"].tolist() == [
        [[0.0], [1.0], [2.0]],
        [[3.0], [4.0], [5.0]],
    ]
    assert combined["history_len"].tolist() == [2, 3]


def test_concatenate_batches_rejects_schema_or_shape_mismatch():
    with pytest.raises(ValueError, match="mismatched sources"):
        _concatenate_batches({"x": torch.zeros(1, 2)}, {"y": torch.zeros(1, 2)})
    with pytest.raises(ValueError, match="shape mismatch"):
        _concatenate_batches({"x": torch.zeros(1, 2)}, {"x": torch.zeros(1, 3)})


def test_pcgrad_removes_negative_two_task_dot_product():
    parameter = torch.nn.Parameter(torch.zeros(2))
    combined, metrics = _pcgrad_pair(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-1.0, 1.0]),),
        [parameter],
    )
    assert metrics["joint_gradient_conflict"] == 1.0
    assert torch.dot(combined[0], torch.tensor([1.0, 0.0])) >= 0
    assert torch.dot(combined[0], torch.tensor([-1.0, 1.0])) >= 0


def test_pcgrad_averages_nonconflicting_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    combined, metrics = _pcgrad_pair(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([0.0, 1.0]),),
        [parameter],
    )
    assert metrics["joint_gradient_conflict"] == 0.0
    assert torch.equal(combined[0], torch.tensor([0.5, 0.5]))


def test_multi_task_pcgrad_projects_conflicts_and_averages_three_tasks():
    parameter = torch.nn.Parameter(torch.zeros(2))
    combined, metrics = _pcgrad_many(
        [
            (torch.tensor([1.0, 0.0]),),
            (torch.tensor([-1.0, 1.0]),),
            (torch.tensor([0.0, 1.0]),),
        ],
        [parameter],
    )
    assert torch.isfinite(combined[0]).all()
    assert metrics["joint_gradient_conflict"] > 0.0
    assert metrics["task_2_gradient_norm"] == pytest.approx(1.0)


def test_equal_norm_pcgrad_prevents_large_source_from_dominating():
    parameter = torch.nn.Parameter(torch.zeros(2))
    combined, metrics = _equal_norm_pcgrad_pair(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([0.0, 9.0]),),
        [parameter],
    )
    assert torch.allclose(combined[0].abs(), torch.tensor([1.5, 1.5]))
    assert metrics["primary_gradient_norm_before_balance"] == pytest.approx(1.0)
    assert metrics["auxiliary_gradient_norm_before_balance"] == pytest.approx(9.0)


def test_fixed_eval_explicitly_requests_chain_output(monkeypatch):
    class Agent:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    class Dataset:
        def batches(self, *_args, **_kwargs):
            yield {"action": torch.zeros(2, dtype=torch.long)}

    def fake_loss(_agent, batch, _cfg):
        assert batch["compute_belief_chain"] is True
        return torch.tensor(0.0), {"belief_chain": 1.0}

    monkeypatch.setattr(bc, "bc_loss", fake_loss)
    metrics = bc._fixed_eval(Agent(), Dataset(), 2, {}, "cpu", seed=3, max_batches=1)
    assert metrics["belief_chain"] == 1.0


def test_reconstruction_retention_backpropagates_to_trunk_not_frozen_decoder(
    monkeypatch,
):
    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Parameter(torch.tensor([[1.0, -2.0, 0.5]]))

        def forward(self, batch):
            size = batch["action"].shape[0]
            return SimpleNamespace(
                log_probs=torch.log_softmax(torch.zeros(size, 2), dim=-1),
                value=torch.zeros(size),
                belief=None,
                belief_chain=None,
                self_belief=None,
                shared_state=self.shared.expand(size, -1),
            )

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 1, bias=False)

        def forward(self, latent):
            return {"probe": self.projection(latent)}

    monkeypatch.setattr(
        "fog.training.shared_latent_reconstruction._loss",
        lambda decoded, _batch, **_kwargs: decoded["probe"].square().mean(),
    )
    agent = Agent()
    decoder = Decoder().requires_grad_(False)
    batch = {
        "action": torch.zeros(2, dtype=torch.long),
        "outcome": torch.zeros(2),
    }
    total, metrics = bc.bc_loss(
        agent,
        batch,
        {"action_coef": 0.0, "value_coef": 0.0, "belief_coef": 0.0},
        retention_decoder=decoder,
        retention_coef=1.0,
    )
    total.backward()
    assert metrics["reconstruction_retention"] > 0
    assert agent.shared.grad is not None
    assert float(agent.shared.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_replay_manifest_split_is_game_level_stable_and_disjoint(tmp_path):
    manifest = pl.DataFrame(
        {
            "id": [f"battle-{index}" for index in range(10)],
            "n_rows": [index + 1 for index in range(10)],
            "shard": [f"shard-{index}.npz" for index in range(10)],
        }
    )
    source = tmp_path / "manifest.parquet"
    manifest.write_parquet(source)

    train_path, holdout_path = bc._split_replay_manifest(source, tmp_path / "split", 0.2, 7)
    train = pl.read_parquet(train_path)
    holdout = pl.read_parquet(holdout_path)
    assert len(train) == 8
    assert len(holdout) == 2
    assert set(train["id"].to_list()).isdisjoint(holdout["id"].to_list())

    reordered = tmp_path / "reordered.parquet"
    manifest.reverse().write_parquet(reordered)
    _, second_holdout = bc._split_replay_manifest(reordered, tmp_path / "second", 0.2, 7)
    assert set(pl.read_parquet(second_holdout)["id"].to_list()) == set(
        holdout["id"].to_list()
    )
