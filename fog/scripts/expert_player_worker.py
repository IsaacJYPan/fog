"""Public-information FOG inference process for bounded, isolated CPU evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


class ExpertPlayer:
    def __init__(self, config, checkpoint, *, temperature=0.0):
        import torch
        from fog.models.agent import FogAgent

        self.agent = FogAgent.from_tables(config).cpu().eval()
        self.agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.temperature = temperature
        self.battles = {}

    def update(self, message):
        from poke_env.battle import Battle
        from fog.constants.tables import TABLES
        from fog.env.player import FogPlayer
        from fog.env.showdown_local import LocalShowdownBattle
        from fog.env.showdown_local_runner import _choice, bind_fog_player
        from fog.env.team_truth import truth_tokens_from_paste
        from fog.pretrained.actions import request_choice

        tag = message["battle_tag"]
        if message["op"] == "close":
            self.battles.pop(tag, None)
            return {"closed": True}
        if tag not in self.battles:
            view = Battle(tag, message["username"], logging.getLogger(__name__), gen=9)
            actor = bind_fog_player(FogPlayer(self.agent, temperature=self.temperature))
            actor.record_mirror = False
            private = truth_tokens_from_paste(message["own_team"], TABLES)
            actor.private_state_truth_provider = lambda _: private.copy()
            self.battles[tag] = (view, actor)
        view, actor = self.battles[tag]
        LocalShowdownBattle._apply_view_update(view, message["messages"], message.get("request"))
        request = message.get("request")
        if not message.get("choose") or not request or request.get("wait"):
            return {"updated": True}
        if request.get("teamPreview"):
            return {"choice": "team 123456", "preview": True}
        choice = _choice(actor, view, request, None)
        if actor.diagnostic_errors:
            raise ValueError("FOG inference reported a diagnostic failure: " + str(actor.diagnostic_errors))
        return {"choice": request_choice(choice, request)}


def main():
    import torch
    from fog.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    player = ExpertPlayer(load_config(args.config), args.checkpoint, temperature=args.temperature)
    print(json.dumps({"status": "ready", "interface": "fog-typed-action-query-public-v1"}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        try:
            response = player.update(message)
        except Exception as error:
            response = {"error": type(error).__name__ + ": " + str(error)}
        print(json.dumps(dict(response, id=message["id"])), flush=True)


if __name__ == "__main__":
    main()
