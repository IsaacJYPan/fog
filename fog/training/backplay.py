"""Demonstration-prefix resets with ordinary player-visible observations.

An exact recorded prefix initializes the simulator, reveal ledger and action
history. Only the fresh continuation can supply on-policy reinforcement labels.
Historical winners and the removed demonstration suffix are diagnostic metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from fog.training.teacher_replay import (
    fog_action,
    mapping_state,
    observation_arrays,
    preview_choice,
    public_trace,
)


@dataclass
class PrefixReset:
    local: object
    actors: dict
    role: str
    counters: dict
    observation: dict
    public_lines: int

    def close(self):
        self.local.close()


def reset_from_record(client, record, *, role, remaining, actors, tag, future_seed=None):
    """Replay to ``remaining`` historical own decisions before the demonstration end.

    The caller must hash-bind the source to a complete, verified replay manifest.
    Both actors must be fresh and have only their own submitted private team
    available to ``observe``. No actor forward pass, action sampling or rollout
    row is created by the prefix. Future actions/outcomes are never copied.
    """
    from fog.constants.tables import TABLES
    from fog.encoding.events import record_history_decision
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.teams import to_packed
    from fog.pretrained.actions import native_choices

    roles = ("p1", "p2")
    if role not in roles or type(remaining) is not int or remaining < 1 or set(actors) != set(roles):
        raise ValueError("Require a valid seat, positive remaining decisions and two fresh actors")
    if not record.get("training_eligible") or any(record.get(key) for key in ("failures", "fallbacks", "choice_rejections")):
        raise ValueError("Only complete unrejected exact-team demonstrations support resets")
    names = {}
    for line in record["public_log"]:
        if line.startswith("|player|"):
            parts = line.split("|")
            names.setdefault(parts[2], parts[3])
    if set(names) != set(roles):
        raise ValueError("Recorded player identities are incomplete")
    count = len(record["perspectives"][role]["actions"]) - 1
    if count < 1 or record["perspectives"][role]["actions"][-1] != -1:
        raise ValueError("Recorded decision count lacks a terminal marker")
    target = max(0, count - remaining)
    options = {
        "formatid": "gen9ou", "strictChoices": True,
        "seed": f"sodium,{record['schedule']['showdown_seed']:064x}",
        **{side: {"name": names[side], "team": to_packed(record["submitted_teams"][side])} for side in roles},
    }
    local = LocalShowdownBattle.create(client, options, battle_tag=tag)
    counters = dict.fromkeys(roles, 0)
    expected, actual = public_trace(record["public_log"]), []
    try:
        for _ in range(2000):
            actual.extend(public_trace(local.update.public_messages))
            if actual != expected[:len(actual)]:
                raise ValueError("Recorded prefix public trace drifted")
            if local.ended:
                raise ValueError("Recorded prefix ended before its selected decision")
            root_request = local.update.requests[role]
            if counters[role] == target and root_request and not root_request.get("wait") and not root_request.get("teamPreview"):
                observation = observation_arrays(actors[role].observe(local.views[role]))
                if future_seed is not None:
                    client.reseed_session(local.session_id, seed=future_seed)
                return PrefixReset(local, actors, role, counters, observation, len(actual))
            choices = []
            for side in roles:
                request = local.update.requests[side]
                if not request or request.get("wait"):
                    choices.append(None)
                    continue
                if request.get("teamPreview"):
                    choices.append(preview_choice(record, side, request))
                    continue
                pov, index = record["perspectives"][side], counters[side]
                if index >= len(pov["actions"]) - 1:
                    raise ValueError("Recorded prefix exhausted a perspective")
                mapping = native_choices(mapping_state(pov["states"][index]), request)
                if pov["observations"][index]["illegal_actions"] != [i not in mapping for i in range(13)]:
                    raise ValueError("Recorded prefix native legality drifted")
                choice = mapping[pov["actions"][index]]
                view, actor = local.views[side], actors[side]
                actor.observe(view)
                action = fog_action(view, request, choice)
                _, _, tracker, _ = actor._battle_state(view)
                record_history_decision(tracker, int(view.turn), view, TABLES, action)
                counters[side] += 1
                choices.append(choice)
            update = local.step(*choices, include_state=False)
            if update.choice_errors:
                raise ValueError("Recorded accepted choice was rejected during prefix replay")
        raise ValueError("Recorded prefix exceeded the replay round cap")
    except BaseException:
        local.close()
        raise
