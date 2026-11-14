"""CPU engineering check of timing hooks; no GPU performance measurement."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fog.training import ppo
from fog.training.direct_parallel import PersistentProcessPool
from fog.training.rollout import RolloutBuffer


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    fixture = runpy.run_path(str(root / "tests/test_ppo_update.py"))
    agent_type = fixture["TinyPolicyAgent"]
    batch_factory = fixture["make_ppo_batch"]

    def updates():
        torch.manual_seed(51)
        agent = agent_type()
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        metrics = [ppo.ppo_update(agent, optimizer, batch_factory(seed=seed),
                                 {"ppo": {"minibatch_size": 32, "epochs": 3}})
                   for seed in range(3)]
        return metrics, {name: tensor.clone() for name, tensor in agent.state_dict().items()}

    torch.set_num_threads(1)
    before_metrics, before_state = updates()
    spec = importlib.util.spec_from_file_location("training_timing", Path(__file__).with_name(
        "training_timing.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = RolloutBuffer()
    gid = buffer.start_game()
    buffer.add_decision(gid, {"x": [1]}, action=0, log_prob=-0.2, value=0.1)
    buffer.end_game(gid, 1.0)
    expected_results = [(buffer, 1)]
    original_update = ppo.ppo_update
    original_phase = ppo._maybe_chunk_update
    try:
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            torch.cuda, "synchronize", return_value=None,
        ), patch.object(torch.cuda, "max_memory_allocated", return_value=0), patch.object(
            PersistentProcessPool, "map", return_value=expected_results,
        ), patch.object(
            ppo, "_maybe_chunk_update", return_value="updated",
        ):
            events = Path(temporary) / "events.jsonl"
            module.install(events)
            # Exercise only the collection wrapper; no fake timing is a GPU result.
            assert PersistentProcessPool.map(object(), []) is expected_results
            after_metrics, after_state = updates()
            setup = SimpleNamespace(buf=SimpleNamespace(n_decisions=64),
                                    batch_decisions=64, total=1000, decisions_seen=0)
            assert ppo._maybe_chunk_update(setup, None, {}) == "updated"
            records = [json.loads(line) for line in events.read_text().splitlines()]
            update_records = [row for row in records if row["kind"] == "update"]
            assert records[0]["rows"] == 1 and records[0]["games"] == 1
            assert [row["rows"] for row in update_records] == [64, 64, 64]
            assert [row["optimized_rows"] for row in update_records] == [192, 192, 192]
            assert all(row["forward_seconds"] <= row["seconds"] for row in update_records)
            assert len([row for row in records if row["kind"] == "update_phase"]) == 1
        assert before_metrics == after_metrics
        assert all(torch.equal(before_state[name], after_state[name]) for name in before_state)
        result = {"label": "CPU engineering smoke; CUDA timing calls mocked",
                  "update_metrics_identical": True, "final_state_tensors_identical": True,
                  "collection_result_identity_preserved": True,
                  "complete_update_phase_return_preserved": True,
                  "row_counts_and_forward_time_partition_valid": True,
                  "gpu_performance_verified": False}
        destination = Path(__file__).with_name("timing-equivalence.json")
        destination.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        ppo.ppo_update = original_update
        ppo._maybe_chunk_update = original_phase


if __name__ == "__main__":
    main()
