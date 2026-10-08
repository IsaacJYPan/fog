# SPDX-License-Identifier: GPL-3.0-only
"""Separate Foul Play process bridge. Requires an unmodified upstream installation.

This integration is distributed under GPL-3.0, independently of FOG's MIT modules.
Foul Play: https://github.com/pmariglia/foul-play
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import sys


class Bridge:
    def __init__(self):
        self.battles = {}

    def update(self, message):
        from fp.battle.state import Battle
        from fp.battle.protocol import update_battle
        from fp.modes.standard_battle import StandardBattleMode
        from fp.modes.base import async_pick_move, handle_team_preview
        from fp.teams.team_converter import export_to_dict

        tag = message["battle_tag"]
        if message["op"] == "close":
            self.battles.pop(tag, None)
            return {"closed": True}
        request = deepcopy(message.get("request"))
        if tag not in self.battles:
            random.seed(message.get("seed", 0))
            battle = Battle(tag)
            battle.pokemon_format = "gen9ou"
            battle.generation = battle.format_spec.generation
            battle.battle_type = battle.format_spec.battle_type
            battle.mode = StandardBattleMode()
            battle.user.team_dict = export_to_dict(message["own_team"])
            role = request["side"]["id"]
            battle.user.name, battle.opponent.name = role, "p2" if role == "p1" else "p1"
            battle.user.account_name = message["username"]
            battle.user.initialize_first_turn_user_from_json(request)
            opponent = [
                line.split("|")[3]
                for line in message["messages"]
                if line.startswith("|poke|" + battle.opponent.name + "|")
            ]
            if len(opponent) != 6:
                raise ValueError("Foul Play requires the public opponent preview")
            battle.initialize_team_preview(opponent, "gen9ou")
            names = {p.name for p in battle.user.reserve + battle.opponent.reserve}
            battle.mode.initialize_team_preview_datasets("gen9ou", names, "")
            battle.team_preview = True
            self.battles[tag] = battle
        else:
            battle = self.battles[tag]
        if request is not None:
            request.setdefault("rqid", 1)
            update_battle(
                battle, "\n".join(message["messages"]) + "\n|request|" + json.dumps(request)
            )
        elif message.get("messages"):
            update_battle(battle, "\n".join(message["messages"]))
        if not message.get("choose") or not request or request.get("wait"):
            return {"updated": True}
        if request.get("teamPreview"):

            class PreviewSink:
                def __init__(self):
                    self.choice = None

                async def send_message(self, _tag, choices):
                    self.choice = choices[0].split("|")[0].removeprefix("/")

            sink = PreviewSink()
            asyncio.run(handle_team_preview(battle, sink))
            return {"choice": sink.choice, "preview": True}
        battle.team_preview = False
        choices = asyncio.run(async_pick_move(battle))
        choice = choices[0].removeprefix("/choose ").removeprefix("/")
        if choice.startswith("switch "):
            return {"choice": choice}
        # Replace move names with authoritative numeric slots; the action is unchanged.
        parts = choice.split()
        selected = "".join(c for c in parts[1].lower() if c.isalnum())
        moves = request["active"][0]["moves"]
        matching = [
            index
            for index, move in enumerate(moves)
            if "".join(c for c in move.get("id", move["move"]).lower() if c.isalnum()) == selected
        ]
        if len(matching) != 1:
            raise ValueError("Foul Play decision does not map to one request move")
        parts[1] = str(matching[0] + 1)
        return {"choice": " ".join(parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument(
        "--trace-deadline",
        action="store_true",
        help="Dump a diagnostic stack before the external deadline",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from fp.config import FoulPlayConfig
    from fp.data.mods.apply_mods import apply_mods
    from fp.format_spec import FormatSpec

    FoulPlayConfig.pokemon_format = "gen9ou"
    FoulPlayConfig.smogon_stats = None
    FoulPlayConfig.search_time_ms = args.search_ms
    FoulPlayConfig.parallelism = 1
    FoulPlayConfig.search_threads = 1
    FoulPlayConfig.team_preview_search_parallelism = None
    FoulPlayConfig.team_preview_search_time_ms = None
    apply_mods(FormatSpec.from_format_string("gen9ou"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from integrations.foul_play_sampling import install_bounded_sampling

    sampling = install_bounded_sampling()
    from integrations.foul_play_processes import install_spawn_search

    search_pool = install_spawn_search()
    import atexit

    atexit.register(search_pool.close)
    random.seed(args.seed)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    if args.prewarm:
        from fp.modes.standard_battle import StandardBattleMode
        from fp.data import pokedex

        mode = StandardBattleMode()
        mode.initialize_team_preview_datasets("gen9ou", set(pokedex), "")
    bridge = Bridge()
    print(
        json.dumps(
            {
                "status": "ready",
                "opponent": "Foul Play",
                "search_ms": args.search_ms,
                "sampling": sampling,
                "search_processes": "persistent-spawn-v1",
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        message = json.loads(line)
        try:
            if args.trace_deadline:
                import faulthandler

                faulthandler.dump_traceback_later(4.0, file=sys.stderr)
            response = bridge.update(message)
        except Exception as error:
            response = {"error": type(error).__name__ + ": " + str(error)}
        finally:
            if args.trace_deadline:
                faulthandler.cancel_dump_traceback_later()
        print(json.dumps(dict(response, id=message["id"])), flush=True)


if __name__ == "__main__":
    main()
