import asyncio
import json

from fog.pretrained.live import LiveController


class Worker:
    def __init__(self):
        self.process = None
        self.calls = []
        self.fail = False

    def request(self, message):
        self.calls.append(message)
        if self.fail:
            raise TimeoutError("deadline")
        return {"choice": "move 1", "elapsed_seconds": 0.1}


def test_live_history_is_public_and_isolated_by_battle(tmp_path):
    worker = Worker()
    live = LiveController(worker, tmp_path)
    request = {"side": {"pokemon": []}, "active": [{"moves": [{"id": "tackle", "pp": 1}]}]}
    live.observe("one", ["", "turn", "1"])
    live.observe("one", ["", "request", json.dumps(request)])
    live.observe("two", ["", "turn", "7"])
    assert asyncio.run(live.choose("one", "Player")) == "move 1"
    assert worker.calls[0]["messages"] == ["|turn|1"]
    assert set(worker.calls[0]) == {"op", "battle_tag", "username", "messages", "request", "choose"}
    assert live.room("two")["pending"] == ["|turn|7"]


def test_live_deadline_failure_keeps_legal_fallback_and_does_not_reuse_bad_cache(tmp_path):
    worker = Worker()
    worker.fail = True
    live = LiveController(worker, tmp_path)
    live.observe(
        "one",
        [
            "",
            "request",
            json.dumps(
                {
                    "active": [
                        {"moves": [{"id": "tackle", "disabled": True}, {"id": "protect", "pp": 1}]}
                    ]
                }
            ),
        ],
    )
    assert asyncio.run(live.choose("one", "Player")) == "move 2"
    worker.fail = False
    assert asyncio.run(live.choose("one", "Player")) == "move 2"
    assert len(worker.calls) == 1
    assert live.room("one")["failures"] == 1
    assert live.room("one")["fallbacks"] == 2
