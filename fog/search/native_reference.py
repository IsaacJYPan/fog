"""Known-team engineering reference for native one-ply counterfactual search.

The coordinator sees complete simulator teams. This is a privileged mechanics
diagnostic, not the public-information candidate reconstruction bridge. Native
network requests still contain only their own private request and POV messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time


@dataclass
class NativeReferenceResult:
    choice: str
    policy_choice: str
    complete_grid: bool = False
    simulations: int = 0
    forced_steps: int = 0
    changed: bool = False
    elapsed_seconds: float = 0.0
    scores: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    privileged_known_teams: bool = True


def known_team_action(client, player, *, state, own_tag, opponent_tag, role,
                      own_proposal, opponent_proposal, seed, max_actions=3,
                      samples=2, seconds=4.0, max_forced_steps=8,
                      reply_temperature=0.0, prior_weight=0.0,
                      min_override_margin=0.0, started=None):
    """Mean sampled shaped return with one simultaneous frozen-native reply.

    Every action shares a reply and planning chance seed per draw. A failed or
    incomplete grid retains the policy proposal and records the reason; it never
    favors actions whose bad branches were silently omitted. The caller owns
    the hard process deadline, including root proposals and this soft budget.
    """
    if (role not in {"p1", "p2"} or type(max_actions) is not int or not 1 <= max_actions <= 13
            or type(samples) is not int or not 1 <= samples <= 64 or seconds <= 0
            or type(max_forced_steps) is not int or not 1 <= max_forced_steps <= 32
            or not all(math.isfinite(value) for value in (seconds, reply_temperature, prior_weight, min_override_margin))
            or reply_temperature < 0 or prior_weight < 0
            or min_override_margin < 0):
        raise ValueError("Invalid native reference search settings")
    begin = time.monotonic() if started is None else started
    result = NativeReferenceResult(own_proposal["choice"], own_proposal["choice"])
    own_choices = {int(key): value for key, value in own_proposal["choices"].items()}
    own_probabilities = own_proposal["probabilities"]
    policy_action = own_proposal["native_action"]
    alternatives = sorted(own_choices, key=lambda action: (-own_probabilities[action], action))
    actions = [policy_action, *[action for action in alternatives if action != policy_action]][:max_actions]
    values = {action: [] for action in actions}
    other = "p2" if role == "p1" else "p1"
    rng = random.Random(seed)
    if len(actions) == 1:
        result.complete_grid = True
        result.elapsed_seconds = time.monotonic() - begin
        return result

    def budget():
        if time.monotonic() - begin >= seconds:
            raise TimeoutError("Native reference soft decision budget exhausted")

    def leaf_value(branch, own_branch, opponent_branch):
        for step in range(max_forced_steps + 1):
            budget()
            own_request = branch["requests"][role]
            message = {"battle_tag": own_branch, "messages": branch["player_logs"][role],
                       "request": own_request, "choose": True}
            if branch["ended"]:
                terminal = player.update(dict(message, op="terminal", choose=False))
                return float(terminal["reward"])
            if own_request and not own_request.get("wait"):
                leaf = player.update(dict(message, op="value"))
                return float(leaf["reward"]) + float(leaf["gamma"]) * float(leaf["value"])
            if step == max_forced_steps:
                raise ValueError("Native reference forced replacements did not resolve")
            player.update(dict(message, op="act", choose=False))
            reply = player.update({"op": "act", "battle_tag": opponent_branch,
                                   "messages": branch["player_logs"][other],
                                   "request": branch["requests"][other], "choose": True})
            if not reply.get("choice"):
                raise ValueError("Native reference live branch has no pending player")
            choices = [None, None]
            choices[0 if other == "p1" else 1] = reply["choice"]
            budget()
            branch = client.sample(branch["state"], choices, seed=f"sodium,{rng.getrandbits(256):064x}")
            result.forced_steps += 1
        raise AssertionError("Unreachable replacement loop")

    try:
        for draw in range(samples):
            budget()
            reply_choices = {int(key): value for key, value in opponent_proposal.get("choices", {}).items()}
            if reply_choices:
                probabilities = opponent_proposal["probabilities"]
                if reply_temperature == 0:
                    reply_action = max(reply_choices, key=lambda action: (probabilities[action], -action))
                else:
                    weights = [probabilities[action] ** (1 / reply_temperature) for action in reply_choices]
                    reply_action = rng.choices(list(reply_choices), weights=weights)[0]
                reply_choice = reply_choices[reply_action]
            else:
                reply_choice = None
            chance_seed = f"sodium,{rng.getrandbits(256):064x}"
            for action in actions:
                budget()
                own_branch = f"{own_tag}-reference-{draw}-{action}"
                opponent_branch = f"{opponent_tag}-reference-{draw}-{action}"
                try:
                    player.update({"op": "fork", "battle_tag": own_branch,
                                   "source_tag": own_tag, "choice": own_choices[action]})
                    player.update({"op": "fork", "battle_tag": opponent_branch,
                                   "source_tag": opponent_tag, "choice": reply_choice})
                    choices = [own_choices[action], reply_choice] if role == "p1" else [reply_choice, own_choices[action]]
                    branch = client.sample(state, choices, seed=chance_seed)
                    result.simulations += 1
                    value = leaf_value(branch, own_branch, opponent_branch)
                    budget()
                    if not math.isfinite(value):
                        raise ValueError("Native reference produced a nonfinite shaped return")
                    values[action].append(value)
                finally:
                    player.update({"op": "close", "battle_tag": own_branch})
                    player.update({"op": "close", "battle_tag": opponent_branch})
        scores = {action: sum(values[action]) / samples
                  + prior_weight * math.log(max(own_probabilities[action], 1e-30)) for action in actions}
        candidate = max(actions, key=lambda action: (scores[action], -action))
        if scores[candidate] - scores[policy_action] < min_override_margin:
            candidate = policy_action
        result.choice = own_choices[candidate]
        result.changed = candidate != policy_action
        result.scores = {action: {"mean_return": sum(values[action]) / samples,
                                  "score": scores[action], "draws": values[action]} for action in actions}
        result.complete_grid = True
    except Exception as error:
        result.failures.append({"type": type(error).__name__, "message": str(error)})
    result.elapsed_seconds = time.monotonic() - begin
    return result
