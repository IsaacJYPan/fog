from __future__ import annotations

import importlib.util
import socket

import pytest

from fog.env.runner import _run_random_games


def _localhost_showdown_up() -> bool:
    try:
        with socket.create_connection(("localhost", 8000), timeout=0.25):
            return True
    except OSError:
        return False


@pytest.mark.skipif(importlib.util.find_spec("poke_env") is None, reason="poke-env not installed")
@pytest.mark.skipif(not _localhost_showdown_up(), reason="no Showdown server on localhost:8000")
@pytest.mark.asyncio
async def test_run_random_games_smoke():
    res = await _run_random_games(
        2,
        4,
        "ws://localhost:8000/showdown/websocket",
        "gen9randombattle",
    )
    assert res.num_games == 4
    assert res.num_workers == 2
    assert res.battles_per_sec > 0
