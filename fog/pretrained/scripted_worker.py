"""Public-information poke-env heuristic opponents behind the decision deadline."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
import logging
import random
import sys

from fog.env.showdown_local import LocalShowdownBattle
from fog.env.showdown_local_runner import _choice
from fog.pretrained.actions import request_choice
from fog.pretrained.evaluation import SCRIPTED_OPPONENTS
from fog.pretrained.process import validate_player_payload


class ScriptedPlayer:
    def __init__(self, kind):
        if kind not in SCRIPTED_OPPONENTS:
            raise ValueError("Unsupported scripted opponent")
        self.kind = kind
        self.battles = {}

    def update(self, message):
        validate_player_payload(message)
        tag = message["battle_tag"]
        if message["op"] == "close":
            self.battles.pop(tag, None)
            return {"closed": True}
        if message["op"] != "act":
            raise ValueError("Scripted opponents support act and close only")
        if tag not in self.battles:
            from poke_env.battle import Battle

            self.battles[tag] = (
                Battle(tag, message["username"], logging.getLogger(__name__), gen=9),
                random.Random(message["seed"]),
            )
        battle, rng = self.battles[tag]
        request = message.get("request")
        LocalShowdownBattle._apply_view_update(battle, message["messages"], request)
        if not message.get("choose") or not request or request.get("wait"):
            return {"updated": True}
        # Match the fixed neural preview rule so lead selection is not a confound.
        if request.get("teamPreview"):
            return {"choice": "team 123456", "preview": True}
        choice = request_choice(_choice(self.kind, battle, request, rng), request)
        if not choice:
            raise ValueError("Scripted opponent failed to return an owed choice")
        return {"choice": choice}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=SCRIPTED_OPPONENTS, required=True)
    args = parser.parse_args()
    player = ScriptedPlayer(args.kind)
    print(json.dumps({"status": "ready", "opponent": args.kind, "poke_env": version("poke-env")}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        try:
            response = player.update(message)
        except Exception as error:
            response = {"error": type(error).__name__ + ": " + str(error)}
        print(json.dumps(dict(response, id=message["id"])), flush=True)


if __name__ == "__main__":
    main()
