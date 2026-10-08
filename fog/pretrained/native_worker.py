"""Public-information Kakuna process, used by local collection and live Showdown."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))


class NativePlayer:
    def __init__(self, runtime, *, temperature=1.0, seed=0):
        import torch

        self.runtime = runtime
        self.temperature = temperature
        self.seed = seed
        self.battles = {}
        self.generator = torch.Generator(
            device=next(runtime.model.parameters()).device
        ).manual_seed(seed)

    def update(self, message):
        import numpy as np
        import torch
        from metamon.env.metamon_battle import MetamonBackendBattle
        from metamon.interface import UniversalState
        from fog.pretrained.actions import native_choices

        tag = message["battle_tag"]
        if message["op"] == "close":
            self.battles.pop(tag, None)
            return {"closed": True}
        if tag not in self.battles:
            battle = MetamonBackendBattle(
                tag, message["username"], logging.getLogger(__name__), False, 9
            )
            hidden = self.runtime.model.traj_encoder.init_hidden_state(
                1, next(self.runtime.model.parameters()).device
            )
            space = copy.deepcopy(self.runtime.observation_space)
            space.reset()
            self.battles[tag] = {
                "battle": battle,
                "hidden": hidden,
                "time": 0,
                "space": space,
                "rl2": torch.zeros(1, 1, 14),
                "previous_state": None,
                "generator": torch.Generator(
                    device=next(self.runtime.model.parameters()).device
                ).manual_seed(
                    int.from_bytes(
                        hashlib.sha256(f"{self.seed}:{tag}".encode()).digest()[:8], "big"
                    )
                ),
            }
        record = self.battles[tag]
        if message["op"] == "rollback":
            snapshot = record.pop("decision_snapshot", None)
            if snapshot is None:
                raise ValueError("No pending native decision to roll back")
            record.update(snapshot)
            return {"rolled_back": True}
        if message["op"] == "commit":
            choice = message["choice"].removeprefix("/choose ").removeprefix("/").strip()
            matches = [index for index, value in record["choices"].items() if value == choice]
            if len(matches) != 1:
                raise ValueError("Executed choice does not match one native action")
            action = matches[0]
            record["rl2"].zero_()
            record["rl2"][0, 0, action + 1] = 1
            record["previous_state"] = record["pending_state"]
            record["time"] += 1
            return {"native_action": action}
        battle = record["battle"]
        for line in message["messages"]:
            event = line.split("|")[1] if line.startswith("|") else ""
            if event not in {"request", "t:", "expire", "uhtmlchange"}:
                battle.parse_message(line.split("|"))
        request = message.get("request")
        if request:
            battle.parse_request(request)
        if message["op"] == "terminal":
            state = UniversalState.from_Battle(battle)
            return {"state": state.to_dict()}
        if not message.get("choose") or not request or request.get("wait"):
            return {"updated": True}
        if request.get("teamPreview"):
            # Keep one fixed preview rule across all neural arms. The upstream temporal
            # policy starts at the first actual move; this preserves that convention.
            return {"choice": "team 123456", "preview": True}
        record["decision_snapshot"] = {
            key: copy.deepcopy(record[key])
            for key in ("hidden", "space", "rl2", "previous_state", "time")
        }
        state = UniversalState.from_Battle(battle)
        from poke_env.environment.pokemon import Pokemon
        from metamon.interface import UniversalPokemon
        from fog.pretrained.requests import reconcile_own_roster

        state = reconcile_own_roster(
            state,
            request,
            lambda member: UniversalPokemon.from_Pokemon(Pokemon(gen=9, request_pokemon=member)),
        )
        choices = native_choices(state, request)
        obs = record["space"].state_to_obs(state)
        obs["illegal_actions"] = np.array([a not in choices for a in range(13)], dtype=bool)
        device = next(self.runtime.model.parameters()).device
        tensors = {k: torch.as_tensor(v, device=device)[None, None] for k, v in obs.items()}
        previous = record["previous_state"]
        reward = self.runtime.reward_function(previous, state) if previous is not None else 0.0
        record["rl2"][0, 0, 0] = reward
        record["choices"] = choices
        record["pending_state"] = state
        encoded = {
            "state": state.to_dict(),
            "choices": choices,
            "observation": {k: v.tolist() for k, v in obs.items()},
            "reward": reward,
        }
        if message["op"] == "encode":
            return encoded
        with torch.inference_mode():
            probabilities, _, hidden = self.runtime.distribution(
                tensors,
                record["rl2"].to(device),
                torch.tensor([[[record["time"]]]], device=device),
                record["hidden"],
            )
        probabilities = probabilities[0, 0].float()
        if not torch.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1.0) > 1e-5:
            raise ValueError("Native policy returned invalid probabilities")
        if self.temperature == 0:
            action = int(probabilities.argmax())
        else:
            scaled = probabilities.pow(1 / self.temperature)
            action = int(torch.multinomial(scaled / scaled.sum(), 1, generator=record["generator"]))
        if action not in choices:
            raise ValueError("Native policy selected an illegal semantic action")
        record["hidden"] = hidden
        self.update({"op": "commit", "battle_tag": tag, "choice": choices[action]})
        return dict(
            encoded,
            choice=choices[action],
            native_action=action,
            probabilities=probabilities.cpu().tolist(),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trained", action="store_true")
    args = parser.parse_args()
    import torch
    from fog.pretrained.kakuna import load_kakuna

    torch.set_num_threads(args.threads)
    with contextlib.redirect_stdout(sys.stderr):
        runtime = load_kakuna(args.checkpoint, device=args.device, verify_original=not args.trained)
    player = NativePlayer(runtime, temperature=args.temperature, seed=args.seed)
    print(
        json.dumps(
            {
                "status": "ready",
                "interface": runtime.interface_version,
                "checkpoint_sha256": runtime.source_sha256,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        message = json.loads(line)
        try:
            response = player.update(message)
        except Exception as error:
            response = {"error": type(error).__name__ + ": " + str(error)}
        print(json.dumps(dict(response, id=message["id"])), flush=True)


if __name__ == "__main__":
    main()
