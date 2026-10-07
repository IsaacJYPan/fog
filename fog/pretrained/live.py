"""Public-only live decision history with recorded, persistent battle fallbacks."""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime
import json
from pathlib import Path

from fog.scripts.pretrained_evaluate import fallback_choice


class LiveController:
    def __init__(self, worker, out: Path):
        self.worker = worker
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.rooms = {}

    def room(self, tag):
        return self.rooms.setdefault(
            tag,
            {
                "pending": [],
                "request": None,
                "degraded": False,
                "failures": 0,
                "fallbacks": 0,
                "decisions": 0,
            },
        )

    def record(self, value):
        with (self.out / "decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(value, utc=datetime.now(UTC).isoformat())) + "\n")

    def observe(self, tag, event):
        room = self.room(tag)
        if len(event) < 2:
            return
        if event[1] == "request" and len(event) > 2 and event[2]:
            room["request"] = json.loads(event[2])
        elif event[1] == "showteam":
            raise ValueError(
                "Open opponent team sheets are outside the public-information protocol"
            )
        elif event[1] == "error":
            room["failures"] += 1
            room["degraded"] = True
            self.record({"battle_tag": tag, "server_error": "|".join(event)})
        else:
            room["pending"].append("|".join(event))

    async def choose(self, tag, username):
        room = self.room(tag)
        message = {
            "op": "act",
            "battle_tag": tag,
            "username": username,
            "messages": list(room["pending"]),
            "request": copy.deepcopy(room["request"]),
            "choose": True,
        }
        room["pending"].clear()
        error, response = None, None
        if not room["degraded"]:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.worker.request, message), timeout=5.0
                )
                choice = response["choice"]
            except Exception as caught:
                error = type(caught).__name__ + ": " + str(caught)
                room["failures"] += 1
                room["degraded"] = True
        if room["degraded"]:
            # After a lost/rejected decision the temporal cache no longer matches
            # the played game. Use legal fallbacks for its remainder, then reset.
            choice = fallback_choice(message["request"])
            room["fallbacks"] += 1
        room["decisions"] += 1
        self.record(
            {
                "battle_tag": tag,
                "request_id": (message["request"] or {}).get("rqid"),
                "choice": choice,
                "failure": error,
                "fallback": room["degraded"],
                "seconds": None if response is None else response["elapsed_seconds"],
                "probabilities": None if response is None else response.get("probabilities"),
            }
        )
        return choice

    async def finish(self, battle):
        tag = battle.battle_tag
        room = self.rooms.pop(tag, self.room(tag))
        if self.worker.process is not None:
            try:
                await asyncio.to_thread(self.worker.request, {"op": "close", "battle_tag": tag})
            except Exception:
                await asyncio.to_thread(self.worker.close, force=True)
        result = {
            "battle_tag": tag,
            "won": battle.won,
            "turns": battle.turn,
            "elo": battle.rating,
            "failures": room["failures"],
            "fallbacks": room["fallbacks"],
            "decisions": room["decisions"],
        }
        with (self.out / "games.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        return result


def build_live_player(controller, **kwargs):
    from poke_env.player import Player
    from poke_env.player.battle_order import SingleBattleOrder

    class NativeShowdownPlayer(Player):
        async def _handle_battle_message(self, split_messages):
            tag = split_messages[0][0].lstrip(">")
            # Preserve chronological prefixes: preloading the whole packet before
            # its first request could expose later events to an earlier decision.
            if not hasattr(self, "_native_room_locks"):
                self._native_room_locks = {}
            lock = self._native_room_locks.setdefault(tag, asyncio.Lock())
            async with lock:
                for event in split_messages[1:]:
                    controller.observe(tag, event)
                    await super()._handle_battle_message([split_messages[0], event])

        async def choose_move(self, battle):
            choice = await controller.choose(battle.battle_tag, self.username)
            return SingleBattleOrder("/choose " + choice if choice else "")

        async def teampreview(self, battle):
            choice = await controller.choose(battle.battle_tag, self.username)
            return "/" + choice if choice else ""

    return NativeShowdownPlayer(max_concurrent_battles=1, accept_open_team_sheet=False, **kwargs)
