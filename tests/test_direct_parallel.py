from __future__ import annotations

import os

import pytest
import torch

from fog.training.direct_parallel import (
    PersistentProcessPool,
    WorkerGame,
    WorkerProcessError,
    partition_games,
    run_workers,
)
from fog.training.rollout import RolloutBuffer


class _PersistentEchoWorker:
    def __init__(self, worker_id: int, prefix: str):
        self.worker_id = worker_id
        self.prefix = prefix
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        if payload == "fail":
            raise RuntimeError("worker failure")
        return self.worker_id, os.getpid(), self.calls, f"{self.prefix}{payload}"


class _SharedTensorWorker:
    def __init__(self, worker_id: int, shared: torch.Tensor):
        self.worker_id = worker_id
        self.shared = shared

    def run(self, payload):
        return self.worker_id, float(self.shared[0]), payload


def _game(worker: int, number: int) -> WorkerGame:
    return WorkerGame(worker, number, 1000 + number, number % 3, number % 5)


def test_partition_is_deterministic_and_tags_are_unique():
    games = [_game(-1, n) for n in range(11)]
    first = partition_games(games, 4)
    second = partition_games(games, 4)
    assert first == second
    assert [g.game_number for part in first for g in part] == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7]
    tagged = [g.battle_tag for part in first for g in part]
    assert len(tagged) == len(set(tagged))


def test_worker_results_preserve_worker_order():
    assignments = partition_games([_game(-1, n) for n in range(8)], 4)
    result = run_workers(assignments, lambda worker, games: (worker, [g.seed for g in games]))
    assert [worker for worker, _ in result] == [0, 1, 2, 3]


def test_worker_exception_propagates_and_executor_shuts_down():
    assignments = partition_games([_game(-1, n) for n in range(4)], 2)

    def fail(worker, games):
        if worker == 1:
            raise RuntimeError("worker failure")
        return games

    with pytest.raises(RuntimeError, match="worker failure"):
        run_workers(assignments, fail)


def test_process_workers_are_persistent_addressable_and_shutdown_cleanly():
    pool = PersistentProcessPool(2, _PersistentEchoWorker, "seen:", timeout=30)
    first = pool.map(["a", "b"])
    second = pool.map(["c", "d"])
    assert [row[0] for row in first] == [0, 1]
    assert [row[1] for row in first] == [row[1] for row in second]
    assert [row[2] for row in first] == [1, 1]
    assert [row[2] for row in second] == [2, 2]
    assert [row[3] for row in second] == ["seen:c", "seen:d"]
    pool.close()
    assert not any(pool.alive)


def test_process_worker_exception_propagates_and_terminates_pool():
    pool = PersistentProcessPool(2, _PersistentEchoWorker, "", timeout=30)
    with pytest.raises(WorkerProcessError, match="worker 1 raised RuntimeError"):
        pool.map(["ok", "fail"])
    assert not any(pool.alive)


def test_process_workers_observe_in_place_shared_weight_refresh():
    shared = torch.zeros(1).share_memory_()
    with PersistentProcessPool(2, _SharedTensorWorker, shared, timeout=30) as pool:
        assert [row[1] for row in pool.map(["before", "before"])] == [0.0, 0.0]
        shared.fill_(3.5)
        assert [row[1] for row in pool.map(["after", "after"])] == [3.5, 3.5]


def test_rollout_merge_preserves_games_decisions_and_labels():
    dst = RolloutBuffer()
    src = RolloutBuffer()
    for outcome in (1.0, -1.0):
        gid = src.start_game()
        src.add_decision(
            gid, {"x": [outcome]}, action=0, log_prob=-0.2, value=0.1
        )
        src.end_game(gid, outcome)
    dst.extend(src)
    assert dst.n_games == 2
    assert dst.n_decisions == 2
    assert src.n_games == src.n_decisions == 0
    batch = dst.to_batch()
    assert batch["actions"].shape[0] == 2


def test_rollout_merge_rejects_unfinished_game():
    dst = RolloutBuffer()
    src = RolloutBuffer()
    src.start_game()
    with pytest.raises(RuntimeError, match="unfinished"):
        dst.extend(src)


def test_rejected_uncommitted_decision_can_be_discarded_before_retry():
    buf = RolloutBuffer()
    gid = buf.start_game()
    buf.add_decision(gid, {"x": [1]}, action=8, log_prob=-0.5, value=0.2)
    buf.discard_last_decision(gid)
    buf.add_decision(gid, {"x": [2]}, action=0, log_prob=-0.1, value=0.3)
    buf.end_game(gid, 1.0)

    batch = buf.to_batch()
    assert batch["actions"].tolist() == [0]
