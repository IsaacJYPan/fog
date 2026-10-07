"""Native sequence inputs and separate, provenance-masked auxiliary targets."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F


def read_record(path: Path) -> dict:
    if path.name.endswith(".lz4"):
        import lz4.frame

        return json.loads(lz4.frame.decompress(path.read_bytes()))
    if path.name.endswith(".gz"):
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


class SequenceLoader:
    def __init__(self, runtime, *, sequence_length=128):
        self.runtime = runtime
        self.sequence_length = sequence_length

    def load(self, specification):
        from amago.loading import RLData
        from metamon.interface import UniversalState, UniversalAction

        source, filename, seed = specification
        path = Path(filename)
        # A fresh file contains both perspectives; the sampler seed chooses a POV.
        rng = random.Random(seed)
        data = read_record(path)
        if source == "fresh":
            if not data["training_eligible"] or data.get("failures") or data.get("fallbacks"):
                raise ValueError("Fresh trajectory contains a failure, fallback, or unverified end")
            data = data["perspectives"][rng.choice(["p1", "p2"])]
        states = [UniversalState.from_dict(copy.deepcopy(state)) for state in data["states"]]
        actions = np.asarray(data["actions"][:-1], dtype=np.int64)
        if (
            len(states) != len(actions) + 1
            or len(actions) < 1
            or np.any(actions < -1)
            or np.any(actions >= 13)
        ):
            raise ValueError("Invalid native trajectory/action shape")
        space = copy.deepcopy(self.runtime.observation_space)
        space.reset()
        observations = [space.state_to_obs(state) for state in states]
        illegal = np.ones((len(states), 13), dtype=bool)
        if source == "fresh":
            for index, observation in enumerate(data["observations"]):
                # This mask comes from the actual private request, never a guessed
                # PP/trapping/disabled state from a spectator replay.
                illegal[index] = observation["illegal_actions"]
                for key in ("numbers", "text_tokens"):
                    if not np.array_equal(observations[index][key], observation[key]):
                        raise ValueError(
                            "Stored live and reconstructed native inputs disagree: " + key
                        )
        else:
            for index, state in enumerate(states[:-1]):
                legal = [
                    self.runtime.action_space.action_to_agent_output(state, action)
                    for action in UniversalAction.maybe_valid_actions(state)
                ]
                illegal[index, legal] = False
        missing = np.append(actions < 0, True)
        known = np.flatnonzero(~missing[:-1])
        if np.any(illegal[known, actions[known]]):
            raise ValueError("Recorded native action excluded by its public/request mask")
        if np.any(illegal[:-1].all(-1)):
            raise ValueError("Nonterminal state has no legal native action")
        obs = {
            key: torch.from_numpy(np.stack([row[key] for row in observations]))
            for key in observations[0]
        }
        obs["illegal_actions"] = torch.from_numpy(illegal)
        obs["missing_action_mask"] = torch.from_numpy(missing[:, None])
        rewards = np.asarray(
            [self.runtime.reward_function(left, right) for left, right in zip(states, states[1:])],
            dtype=np.float32,
        )
        dones = np.zeros(len(actions), dtype=bool)
        terminal = states[-1]
        if bool(terminal.battle_won) == bool(terminal.battle_lost):
            raise ValueError("Outcome supervision requires an explicit decisive terminal state")
        dones[-1] = True
        example = RLData(
            obs=obs,
            actions=F.one_hot(torch.from_numpy(actions).clamp(min=0), 13).float(),
            rews=torch.from_numpy(rewards[:, None]),
            dones=torch.from_numpy(dones[:, None]),
            time_idxs=torch.arange(len(states))[:, None],
        )
        # Draw a center uniformly: long games' final decisions stay sampleable.
        length = min(self.sequence_length, len(actions))
        start = min(max(rng.randrange(len(actions)) - length // 2, 0), len(actions) - length)
        stop = start + length
        example.obs = {key: value[start : stop + 1] for key, value in example.obs.items()}
        example.rl2s = example.rl2s[start : stop + 1]
        example.time_idxs = example.time_idxs[start : stop + 1]
        example.actions, example.rews, example.dones = (
            value[start:stop] for value in (example.actions, example.rews, example.dones)
        )
        reveals = np.full((length, 6, 15), -1, dtype=np.int64)
        truth = reveals.copy()
        verified = np.zeros_like(reveals, dtype=bool)
        if source == "fresh":
            labels = data["labels"][start:stop]
            reveals = np.asarray([row["reveals"] for row in labels], dtype=np.int64)
            truth = np.asarray([row["truth"] for row in labels], dtype=np.int64)
            verified = np.asarray([row["verified"] for row in labels], dtype=bool)
            if np.any(verified & (truth < 0)):
                raise ValueError("A missing submitted-set label cannot be verified")
        outcome = 1.0 if terminal.battle_won else -1.0
        targets = {
            "reveals": torch.from_numpy(reveals),
            "truth": torch.from_numpy(truth),
            "verified": torch.from_numpy(verified),
            "outcomes": torch.full((length,), outcome),
            "outcome_valid": torch.ones(length, dtype=torch.bool),
        }
        return example, targets

    def batch(self, specifications, device):
        from amago.loading import RLData_pad_collate

        loaded = [self.load(specification) for specification in specifications]
        batch = RLData_pad_collate([example for example, _ in loaded]).to(device)
        maximum = batch.actions.shape[1]
        targets = {}
        for key in loaded[0][1]:
            values = []
            for _, row in loaded:
                value = row[key]
                padding = maximum - len(value)
                fill = -1 if key in ("reveals", "truth") else 0
                value = torch.cat(
                    [value, torch.full((padding, *value.shape[1:]), fill, dtype=value.dtype)]
                )
                values.append(value)
            targets[key] = torch.stack(values).to(device)
        return batch, targets


def fresh_files(folder: Path, *, minimum_games=1000):
    result, failed, counts = [], [], {}
    for path in sorted(folder.glob("*.json.gz")):
        record = read_record(path)
        if (
            record.get("training_eligible")
            and not record.get("failures")
            and not record.get("fallbacks")
        ):
            result.append(str(path))
            name = record["opponent"]
            counts[name] = counts.get(name, 0) + 1
        else:
            failed.append(str(path))
    if len(result) < minimum_games:
        raise ValueError(
            f"Need {minimum_games} verified fresh games; found {len(result)} ({len(failed)} failures)"
        )
    total = len(result)
    expected = {"foulplay": 0.4, "kakuna": 0.3, "learner": 0.3}
    if minimum_games >= 1000 and any(
        counts.get(name, 0) != round(total * fraction) for name, fraction in expected.items()
    ):
        raise ValueError(
            "Verified fresh opponent mixture is not exactly 40/30/30; replenish failed games"
        )
    return result, {"eligible": total, "failures": failed, "opponent_counts": counts}
