"""Recover FOG supervision by exactly replaying archived authoritative games.

Only a replay with the same public trace, winner, action counts and legal native
choices may produce a shard. Complete submitted opponent teams remain labels;
the encoder sees the censored battle view and the acting player's own team.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fog.pretrained.actions import identifier, native_choices, request_choice


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_record(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def public_trace(lines):
    """Exclude clock timestamps, which are not simulator mechanics or inputs."""
    return [line for line in lines if not line.startswith("|t:|")]


def teacher_roles(record):
    native_role = "p1" if record["schedule"]["p1_is_a"] else "p2"
    opponent_role = "p2" if native_role == "p1" else "p1"
    if record["opponent"] == "kakuna":
        return {"p1": "kakuna34", "p2": "kakuna34"}
    if record["opponent"] == "foulplay":
        return {native_role: "kakuna34", opponent_role: "foulplay"}
    if record["opponent"] == "learner":
        return {native_role: "kakuna34"}
    raise ValueError("Unknown archived teacher identity")


def mapping_state(row):
    """Read only the archived acting player's identities for semantic mapping."""
    active = row["player_active_pokemon"]
    return SimpleNamespace(
        player_active_pokemon=SimpleNamespace(
            moves=[SimpleNamespace(name=move["name"]) for move in active["moves"]]
        ),
        available_switches=[
            SimpleNamespace(name=mon["name"], base_species=mon["base_species"])
            for mon in row["available_switches"]
        ],
    )


def preview_choice(record, role, request):
    """Recover the recorded lead under the archived lead-then-original-order rule.

    The rest of the order is validated by the subsequent complete public trace.
    The recovered preview is replay transport, never a supervised future input.
    """
    active = record["perspectives"][role]["states"][0]["player_active_pokemon"]
    lead_names = {identifier(active["name"]), identifier(active["base_species"])}
    members = request["side"]["pokemon"]
    candidates = [
        index + 1 for index, mon in enumerate(members)
        if identifier(mon["details"].split(",")[0]) in lead_names
    ]
    if len(candidates) != 1 or len(members) != 6:
        raise ValueError("Archived preview lead is ambiguous or the team is incomplete")
    first = candidates[0]
    return "team " + "".join(str(index) for index in [first, *[i for i in range(1, 7) if i != first]])


def fog_action(battle, request, choice):
    """Invert the normal live FOG action mapper against the authoritative request."""
    from fog.env.actions import action_legality_mask, map_action_to_order
    from fog.env.showdown_local_runner import LocalOrderFactory, _choice

    selected = request_choice(choice, request)
    matches = []
    for index, legal in enumerate(action_legality_mask(battle)):
        if not legal:
            continue
        order = map_action_to_order(battle, index, LocalOrderFactory())
        # Numeric switch slots must be normalized through the normal direct runner.
        message = _choice(lambda _: order, battle, request, None)
        if request_choice(message, request) == selected:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"Archived choice maps to {len(matches)} FOG actions: {selected}")
    return matches[0]


def observation_arrays(observation):
    from fog.encoding.public_action_features import public_action_features

    return {
        "entities": observation.entities.copy(),
        "entity_ids": observation.entity_ids.copy(),
        "field": observation.field.copy(),
        "history": observation.history.copy(),
        "history_ids": observation.history_ids.copy(),
        "history_state": observation.history_state.copy(),
        "history_len": np.int64(observation.history_len),
        "active_idx": observation.active_idx.copy(),
        "legality_mask": observation.legality_mask.copy(),
        "action_features": public_action_features(observation.action_features),
        "opp_tokens": observation.opp_revealed["tokens"].copy(),
    }


def shard_arrays(rows):
    from fog.encoding.dims import DIMS

    result = {}
    for key in rows[0]:
        if key in ("history", "history_ids", "history_state"):
            first = rows[0][key]
            values = np.zeros((len(rows), DIMS.history_cap, first.shape[-1]), dtype=first.dtype)
            for index, row in enumerate(rows):
                kept = min(len(row[key]), DIMS.history_cap)
                if kept:
                    values[index, -kept:] = row[key][-kept:]
            result[key] = values
        else:
            result[key] = np.stack([row[key] for row in rows])
    return result


def replay_game(client, record, *, tag="teacher-replay"):
    """Replay and encode a complete game; fail before committing any mismatched shard."""
    from fog.constants.tables import TABLES
    from fog.encoding.events import record_history_decision
    from fog.env.player import FogPlayer
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import to_packed

    if not record["training_eligible"] or record.get("failures") or record.get("fallbacks"):
        raise ValueError("Archived game is not an exact completed trajectory")
    if record.get("choice_rejections"):
        raise ValueError("Request-corrected source requires an explicit replay of its corrections")
    roles = ("p1", "p2")
    teachers = teacher_roles(record)
    names = {}
    for line in record["public_log"]:
        if line.startswith("|player|"):
            parts = line.split("|")
            names.setdefault(parts[2], parts[3])
    if set(names) != set(roles):
        raise ValueError("Archived player identity is missing")
    truths = {role: truth_tokens_from_paste(record["submitted_teams"][role], TABLES) for role in roles}
    for role in roles:
        pov = record["perspectives"][role]
        if len(pov["states"]) != len(pov["actions"]) or pov["actions"][-1] != -1:
            raise ValueError("Archived action/terminal shape mismatch")
    options = {
        "formatid": "gen9ou", "strictChoices": True,
        "seed": f"sodium,{record['schedule']['showdown_seed']:064x}",
        **{role: {"name": names[role], "team": to_packed(record["submitted_teams"][role])} for role in roles},
    }
    counters = dict.fromkeys(roles, 0)
    actors = {role: FogPlayer() for role in roles}
    for role, actor in actors.items():
        actor.private_state_truth_provider = lambda _, side=role: truths[side].copy()
    rows = {role: [] for role in roles}
    expected, actual = public_trace(record["public_log"]), []
    with LocalShowdownBattle.create(client, options, battle_tag=tag) as local:
        for _ in range(1000):
            actual.extend(public_trace(local.update.public_messages))
            if actual != expected[:len(actual)]:
                mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
                raise ValueError(f"Public simulator trace drift at line {mismatch}")
            if local.ended:
                break
            choices = []
            for role in roles:
                request = local.update.requests[role]
                if not request or request.get("wait"):
                    choices.append(None)
                    continue
                if request.get("teamPreview"):
                    choices.append(preview_choice(record, role, request))
                    continue
                pov, index = record["perspectives"][role], counters[role]
                if index >= len(pov["actions"]) - 1:
                    raise ValueError("Simulator owes a decision after archived actions ended")
                action = pov["actions"][index]
                mapping = native_choices(mapping_state(pov["states"][index]), request)
                archived_mask = pov["observations"][index]["illegal_actions"]
                if archived_mask != [a not in mapping for a in range(13)]:
                    raise ValueError("Replayed native legality differs from the recorded authoritative mask")
                if action not in mapping:
                    raise ValueError("Archived action is not legal in the replayed request")
                view, actor = local.views[role], actors[role]
                observation = actor.observe(view)
                fog_index = fog_action(view, request, mapping[action])
                if not observation.legality_mask[fog_index]:
                    raise ValueError("Mapped FOG teacher action is illegal")
                # Labels are assembled only after the public fragment exists.
                fragment = observation_arrays(observation)
                other = "p2" if role == "p1" else "p1"
                fragment.update(
                    action=np.int64(fog_index),
                    belief_targets=truths[other].copy(),
                    self_belief_targets=np.full_like(truths[other], -1),
                    outcome=np.float32(pov["outcome"]),
                    policy_valid=np.bool_(role in teachers),
                    own_set_provenance=np.full((6, 15), 2, dtype=np.uint8),
                    opponent_set_provenance=np.full((6, 15), 2, dtype=np.uint8),
                    legal_provenance=np.full(14, 2, dtype=np.uint8),
                    action_feature_provenance=np.full(14, 2, dtype=np.uint8),
                    imputation_id=np.int16(-1),
                )
                rows[role].append(fragment)
                _, _, tracker, _ = actor._battle_state(view)
                record_history_decision(tracker, int(view.turn), view, TABLES, fog_index)
                choices.append(mapping[action])
                counters[role] += 1
            update = local.step(*choices, include_state=False)
            if update.choice_errors:
                raise ValueError("Archived accepted action was rejected by replay: " + str(update.choice_errors))
        if not local.ended or actual != expected:
            raise ValueError("Public simulator trace did not reach the exact archived end")
        if local.winner != record["winner"] or local.update.turn != record["turns"]:
            raise ValueError("Replayed outcome or turn count differs")
        for role in roles:
            pov = record["perspectives"][role]
            if counters[role] != len(pov["actions"]) - 1 or not counters[role]:
                raise ValueError("Replayed perspective action count differs")
            if pov["outcome"] != (1.0 if local.winner == names[role] else -1.0):
                raise ValueError("Archived perspective outcome differs from the simulator")
    combined = rows["p1"] + rows["p2"]
    arrays = shard_arrays(combined)
    return arrays, {"decisions": counters, "rows": len(combined),
                    "policy_rows": int(arrays["policy_valid"].sum()), "teachers": teachers,
                    "public_lines": len(actual), "exact_public_trace": True,
                    "native_legality_exact": True, "own_private_input": "submitted_own_team",
                    "opponent_truth_is_label_only": True, "teacher_answer_features_zeroed": True}
