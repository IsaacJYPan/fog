"""A timing-only probe must not serialize unused, multi-gigabyte rollout fragments."""
from __future__ import annotations

import pickle
from types import SimpleNamespace


def test_pool_probe_counts_without_transferring_observation_buffers(monkeypatch):
    from fog.eval import showdown_rollout_probe as probe
    from fog.training import direct_parallel, ppo

    closed = []

    class UntransferableBuffer:
        def __init__(self, games):
            self._games = {g.game_number: SimpleNamespace(actions=[0, 1, 2]) for g in games}

        def __reduce__(self):
            raise AssertionError("the probe tried to copy the full observation buffer")

    class Worker:
        def __init__(self, worker_id, *_args):
            self.worker_id = worker_id

        def run(self, payload):
            games, _opponent, _bonus = payload
            return UntransferableBuffer(games), len(games)

        def close(self):
            closed.append(self.worker_id)

    class PipePool:
        def __init__(self, count, factory, *args):
            self.workers = [factory(i, *args) for i in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            for worker in self.workers:
                worker.close()

        def map(self, payloads):
            # Model the real pool's IPC boundary. The old probe fails while pickling
            # UntransferableBuffer; the fixed probe sends only small integer counters.
            return [pickle.loads(pickle.dumps(worker.run(payload)))
                    for worker, payload in zip(self.workers, payloads, strict=True)]

    monkeypatch.setattr(ppo, "_ShowdownRolloutWorker", Worker)
    monkeypatch.setattr(direct_parallel, "PersistentProcessPool", PipePool)
    result = probe.run_pool_probe(
        "configs/gen9ou_typed_token_selfplay.yaml", "data/teams/pool_a", 6, 3,
        workers=2, batch=32, opponent="MaxDamage",
    )
    assert result["games"] == 6
    assert result["learner_decisions"] == 18
    assert result["p1_wins"] == 6
    assert result["learner_decisions_per_s"] > 0
    assert closed == [0, 1]
