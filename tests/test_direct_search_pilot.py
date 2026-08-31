from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from fog.env.player import FogPlayer
from fog.eval.direct_search_pilot import atomic_json, row_payload, wilson95


def test_wilson_and_empty_interval():
    assert wilson95(0, 0) == (0.0, 1.0)
    low, high = wilson95(6, 10)
    assert 0.0 < low < 0.6 < high < 1.0


def test_atomic_progress_and_search_accounting(tmp_path):
    adapter = SimpleNamespace(
        search_stats={
            "decisions": 7,
            "simulation_attempts": 224,
            "simulation_failures": 2,
            "fallbacks": 1,
            "last_error": "example",
        }
    )
    row = row_payload(
        "search vs SimpleHeuristics",
        "SimpleHeuristics",
        True,
        3,
        4,
        2.5,
        adapter,
        losses=1,
    )
    assert row["decisive_win_rate"] == 0.75
    assert row["search_stats"]["fallbacks"] == 1

    path = tmp_path / "results.json"
    atomic_json(path, {"rows": [row]})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["rows"][0]["games"] == 4
    assert not path.with_suffix(".json.tmp").exists()


def test_search_fallback_is_explicitly_counted(monkeypatch):
    class Agent:
        def __call__(self, batch):
            logits = torch.zeros((1, 14))
            logits[:, 2:] = float("-inf")
            return SimpleNamespace(log_probs=torch.log_softmax(logits, dim=-1))

    class FailingStepper:
        def __init__(self, *args, **kwargs):
            self.failures = 3
            self._bases = {}

    def fail_search(*args, **kwargs):
        raise RuntimeError("reconstruction failed")

    monkeypatch.setattr("fog.search.engine_step.ShowdownWorldStepper", FailingStepper)
    monkeypatch.setattr("fog.search.determinized.determinized_action", fail_search)
    adapter = FogPlayer(Agent(), temperature=0.0)
    adapter.search = {"k": 8, "prior_weight": 0.5}
    batch = {"legality_mask": torch.tensor([[True, True] + [False] * 12])}
    action = adapter._search_action(
        SimpleNamespace(player_role="p1"), None, None, batch, torch.device("cpu")
    )

    assert action in (0, 1)
    assert adapter.search_stats["decisions"] == 1
    assert adapter.search_stats["simulation_attempts"] == 16
    assert adapter.search_stats["simulation_failures"] == 3
    assert adapter.search_stats["fallbacks"] == 1
    assert "reconstruction failed" in adapter.search_stats["last_error"]
