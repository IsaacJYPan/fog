from __future__ import annotations

import copy
import csv
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from fog.config import load_config_with_args
from fog.training import bc


class TinyAgent(nn.Module):
    """Real stochastic forward/backward updates without the large battle encoder."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.dropout = nn.Dropout(0.3)
        self.policy = nn.Linear(4, 3)
        self.value = nn.Linear(4, 1)

    def forward(self, batch):
        hidden = self.dropout(torch.tanh(self.hidden(batch["features"])))
        # These streams must also survive resume, even though the current model mostly uses
        # torch. Evaluation consumes the same streams in both reference and interrupted runs.
        hidden = hidden + (random.random() + np.random.random()) * 0.001
        return SimpleNamespace(
            log_probs=self.policy(hidden).log_softmax(-1),
            value=self.value(hidden).squeeze(-1).tanh(),
            belief=None,
            belief_chain=None,
            self_belief=None,
        )


class TinyDataset:
    def __init__(self, path, **_kwargs):
        self.table = pl.read_parquet(path)

    def __len__(self):
        return len(self.table)

    def batches(self, batch_size, shuffle=True, seed=0, device=None):
        order = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            rows = self.table[order[start:start + batch_size]]
            yield {
                "features": torch.tensor(rows["features"].to_list(), device=device),
                "action": torch.tensor(rows["action"].to_list(), device=device),
                "outcome": torch.tensor(rows["outcome"].to_list(), device=device),
            }


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    path = tmp_path / "tiny.parquet"
    pl.DataFrame({
        "features": [[0.0, 1.0], [1.0, 0.0], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]],
        "action": [0, 1, 2, 1, 0],
        "outcome": [1.0, -1.0, 1.0, -1.0, 1.0],
    }).write_parquet(path)
    monkeypatch.setattr("fog.models.agent.FogAgent.from_tables", lambda _cfg: TinyAgent())
    monkeypatch.setattr("fog.training.dataset.ReplayDataset", TinyDataset)
    monkeypatch.setattr("fog.training.dataset.StreamingReplayDataset", TinyDataset)
    previous_threads = torch.get_num_threads()
    yield {
        "seed": 17,
        "device": "cpu",
        "preflight": {"required": False},
        "loss": {"belief_coef": 0.0},
        "bc": {
            "manifest": str(path), "batch_size": 2, "epochs": 4,
            "lr": 0.003, "lr_final": 0.0003, "eval_batches": 0,
            "resume": True, "torch_threads": 1,
        },
        "log": {"dir": str(tmp_path / "run")},
    }
    torch.set_num_threads(previous_threads)


def _seed_external():
    random.seed(123)
    np.random.seed(321)


def _history(cfg):
    with (Path(cfg["log"]["dir"]) / "metrics.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def _state(cfg):
    return torch.load(
        Path(cfg["log"]["dir"]) / "checkpoints" / "bc_state.pt", weights_only=False,
    )


def _assert_state_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_state_equal(a, b)
    else:
        assert left == right


def _interrupt_after_epoch(monkeypatch, epoch):
    original = bc._atomic_save

    def save_and_interrupt(state, path):
        original(state, path)
        if state["epoch"] == epoch:
            raise KeyboardInterrupt("simulated process interruption after a completed epoch")

    monkeypatch.setattr(bc, "_atomic_save", save_and_interrupt)


def test_resume_matches_uninterrupted_model_optimizer_schedule_rng_and_metrics(
    tiny_training, monkeypatch,
):
    uninterrupted = copy.deepcopy(tiny_training)
    uninterrupted["log"]["dir"] += "-uninterrupted"
    _seed_external()
    bc.train(uninterrupted)

    interrupted = tiny_training
    _seed_external()
    with monkeypatch.context() as stop:
        _interrupt_after_epoch(stop, 2)
        with pytest.raises(KeyboardInterrupt):
            bc.train(interrupted)
    assert _state(interrupted)["epoch"] == 2
    # Deliberately corrupt a partially written CSV and best model; epoch state is authoritative.
    (Path(interrupted["log"]["dir"]) / "metrics.csv").write_text("partial epoch\n")
    best_path = Path(interrupted["log"]["dir"]) / "checkpoints" / "bc_best.pt"
    torch.save({}, best_path)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    bc.train(interrupted)

    ignored = {"seconds_per_epoch", "train_rows_per_second"}
    first = [{k: v for k, v in row.items() if k not in ignored} for row in _history(uninterrupted)]
    second = [{k: v for k, v in row.items() if k not in ignored} for row in _history(interrupted)]
    assert first == second
    for key in (
        "model", "optimizer", "scheduler", "scaler", "rng", "best_model", "best_epoch",
        "best_eval_loss", "best_meaningful_eval_loss", "epochs_without_improvement", "step",
    ):
        _assert_state_equal(_state(uninterrupted)[key], _state(interrupted)[key])
    assert _state(interrupted)["epoch"] == 4
    for row in _history(interrupted)[1:]:
        assert float(row["seconds_per_epoch"]) > 0
        assert float(row["train_rows_per_second"]) > 0
    # Repeating an already complete command neither starts over nor adds metric rows.
    before = _history(interrupted)
    bc.train(interrupted)
    assert _history(interrupted) == before


@pytest.mark.parametrize("key,value", [
    ("epochs", 5), ("batch_size", 3), ("lr", 0.004), ("lr_final", 0.0004),
    ("selection_metric", "action_ce"),
])
def test_resume_refuses_changed_training_settings(tiny_training, key, value):
    _seed_external()
    bc.train(tiny_training)
    tiny_training["bc"][key] = value
    with pytest.raises(ValueError, match=rf"cannot resume: changed .*bc\.{key}"):
        bc.train(tiny_training)


def test_resume_refuses_changed_manifest_contents(tiny_training):
    _seed_external()
    bc.train(tiny_training)
    source = Path(tiny_training["bc"]["manifest"])
    pl.read_parquet(source).reverse().write_parquet(source)
    with pytest.raises(ValueError, match=r"changed .*bc\.manifest"):
        bc.train(tiny_training)


def test_resume_manifest_signature_survives_moved_rebased_directory(tmp_path):
    for name in ("old", "restored"):
        root = tmp_path / name
        (root / "split").mkdir(parents=True)
        manifest = root / "split" / "train.parquet"
        pl.DataFrame({
            "id": ["replay-a"], "n_rows": [3],
            "shard": [str(root / "decisions" / "a.npz")],
        }).write_parquet(manifest)
    old = tmp_path / "old" / "split" / "train.parquet"
    restored = tmp_path / "restored" / "split" / "train.parquet"
    assert bc._resume_settings({"bc": {"manifest": str(old)}}, old, None) == (
        bc._resume_settings({"bc": {"manifest": str(restored)}}, restored, None)
    )


def test_resume_disabled_by_default_and_does_not_change_updates(tiny_training):
    enabled = copy.deepcopy(tiny_training)
    _seed_external()
    bc.train(enabled)
    disabled = copy.deepcopy(tiny_training)
    del disabled["bc"]["resume"]
    disabled["log"]["dir"] += "-default"
    _seed_external()
    bc.train(disabled)
    checkpoints = Path(disabled["log"]["dir"]) / "checkpoints"
    assert not (checkpoints / "bc_state.pt").exists()
    _assert_state_equal(
        _state(enabled)["model"], torch.load(checkpoints / "bc_final.pt", weights_only=True),
    )
    ignored = {"seconds_per_epoch", "train_rows_per_second"}
    assert [{k: v for k, v in r.items() if k not in ignored} for r in _history(enabled)] == [
        {k: v for k, v in r.items() if k not in ignored} for r in _history(disabled)
    ]


@pytest.mark.parametrize("metric,expected_best,expected_epochs", [
    (None, 1, 2), ("loss", 1, 2), ("action_ce", 2, 3),
])
def test_selection_metric_controls_best_and_early_stopping(
    tiny_training, monkeypatch, metric, expected_best, expected_epochs,
):
    scores = iter([(3.0, 2.0), (2.0, 1.7), (2.4, 1.3), (2.8, 1.5), (3.2, 1.6)])

    def evaluate(*_args, **_kwargs):
        loss, action_ce = next(scores)
        return {
            "loss": loss, "action_ce": action_ce,
            "belief": 0.0, "belief_chain": 0.0, "value": 1.0,
            "belief_chain_dropped_rate": 0.0,
        }

    monkeypatch.setattr(bc, "_fixed_eval", evaluate)
    tiny_training["bc"]["early_stopping_patience"] = 1
    if metric is not None:
        tiny_training["bc"]["selection_metric"] = metric
    bc.train(tiny_training)
    state = _state(tiny_training)
    assert state["best_epoch"] == expected_best
    assert state["epoch"] == expected_epochs
    assert state["early_stopped"] is True
    directory = Path(tiny_training["log"]["dir"]) / "checkpoints"
    _assert_state_equal(
        torch.load(directory / "bc_best.pt", weights_only=True),
        torch.load(directory / f"bc_epoch{expected_best}.pt", weights_only=True),
    )
    # Restarting an early-stopped run must not reset its patience and continue training.
    bc.train(tiny_training)
    assert _state(tiny_training)["epoch"] == expected_epochs


def test_new_options_cross_config_schema_and_reject_bad_values(tiny_training):
    config = load_config_with_args("configs/replay_blueprint_cpu.yaml", [
        "bc.resume=true", "bc.selection_metric=action_ce", "bc.torch_threads=1",
    ])
    assert config["bc"]["resume"] is True
    tiny_training["bc"]["selection_metric"] = "misspelled"
    with pytest.raises(ValueError, match="bc.selection_metric"):
        bc.train(tiny_training)
    tiny_training["bc"]["selection_metric"] = "loss"
    tiny_training["bc"]["torch_threads"] = 0
    with pytest.raises(ValueError, match="bc.torch_threads must be positive"):
        bc.train(tiny_training)


def test_thread_option_leaves_runtime_setting_unchanged_by_default(tiny_training, monkeypatch):
    del tiny_training["bc"]["torch_threads"]
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    bc.train(tiny_training)
    assert calls == []
