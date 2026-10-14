"""Bounded known-team native search engineering checks on authoritative Showdown."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import types

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))


def record_digest(record):
    """Detect parent history mutations, including cache NaN bits and parser state."""
    import torch

    digest = hashlib.sha256()
    digest.update(str(record["time"]).encode())
    for key in ("hidden", "pending_hidden"):
        hidden = record.get(key)
        if hidden is not None:
            for tensor in (hidden.key_cache.data, hidden.val_cache.data, hidden.seq_lens):
                digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    for tensor in (record["rl2"], record["generator"].get_state()):
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    for key in ("space", "previous_state", "pending_state"):
        digest.update(pickle.dumps(record.get(key), protocol=5))
    digest.update(pickle.dumps(record["battle"]._mm_battle, protocol=5))
    return digest.hexdigest()


def model_digest(model):
    """Verify in-memory parameters/buffers, rather than only the checkpoint file."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(checkpoint, pool, reference_source, out, *, games=8, max_rounds=2000, seed=20261014):
    import numpy as np
    import torch

    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.teams import to_packed
    from fog.eval.showdown_direct_selection import matched_schedule
    from fog.pretrained.kakuna import load_kakuna, sha256
    from fog.pretrained.native_worker import NativePlayer
    from fog.search.native_reference import known_team_action

    if out.exists() or games < 2 or (games != 2 and games % 4) or not 2 <= max_rounds <= 2000:
        raise ValueError("Use a new canary identity and a bounded game/round budget")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    before = sha256(checkpoint)
    with contextlib.redirect_stdout(sys.stderr):
        runtime = load_kakuna(checkpoint)
    initial_model_digest = model_digest(runtime.model)
    reference = types.ModuleType("frozen_native_reference")
    reference.__file__ = str(reference_source)
    exec(compile(reference_source.read_text(), str(reference_source), "exec"), reference.__dict__)
    teams = json.loads(pool.read_text())["teams"]
    settings = {
        "stage": "known-team-engineering", "checkpoint_sha256": before,
        "pool_sha256": sha256(pool), "reference_source_sha256": sha256(reference_source),
        "interface": runtime.interface_version, "games_per_arm": games, "max_rounds": max_rounds,
        "reward_function": runtime.reward_function.__class__.__name__,
        "reward_multiplier": float(runtime.model.reward_multiplier),
        "value_units": "unscaled_environment_shaped_return",
        "deployed_gamma": float(runtime.model.gammas[-1]),
        "seed": seed, "temperature": 0.0, "reply_temperature": 0.0,
        "fixed_preview": "team 123456", "max_actions": 3, "chance_samples": 2,
        "decision_seconds": 4.0, "measured_deadline_seconds": 5.0,
        "privileged_known_teams": True, "public_team_inference_tested": False,
        "playing_strength_confirmation": False, "automatic_promotion": False,
        "training_data_exported": False, "gpu_requested": False,
    }
    (out / "PROTOCOL.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.monotonic()
    rows, decisions = [], []
    maximum_error = 0.0
    maximum_seconds = 0.0
    failures = []
    try:
        with ShowdownDirectClient() as client:
            schedule = matched_schedule(len(teams), max(8, games), seed)[:games]
            for arm in ("policy", "known_search"):
                player = NativePlayer(runtime, temperature=0, seed=seed)
                control = reference.NativePlayer(runtime, temperature=0, seed=seed)
                for index, game in enumerate(schedule):
                    names = {"p1": "Candidate" if game.p1_is_a else "Opponent",
                             "p2": "Opponent" if game.p1_is_a else "Candidate"}
                    role = "p1" if game.p1_is_a else "p2"
                    other = "p2" if role == "p1" else "p1"
                    tags = {side: f"battle-gen9ou-canary-{index}-{side}" for side in names}
                    session, update = client.create_session({
                        "formatid": "gen9ou", "strictChoices": True,
                        "seed": f"sodium,{game.showdown_seed:064x}",
                        "p1": {"name": names["p1"], "team": to_packed(teams[game.p1_team])},
                        "p2": {"name": names["p2"], "team": to_packed(teams[game.p2_team])},
                    }, include_state=True)
                    rounds = 0
                    try:
                        while not update.ended and rounds < max_rounds:
                            begin = time.monotonic()
                            responses = {}
                            for side in names:
                                message = {"op": "propose", "battle_tag": tags[side], "username": names[side],
                                           "messages": update.messages[side], "request": update.requests[side], "choose": True}
                                responses[side] = player.update(message)
                                legacy = control.update(dict(message, op="act"))
                                if "probabilities" in responses[side]:
                                    current = responses[side]
                                    error = float(np.max(np.abs(np.asarray(current["probabilities"])
                                                               - np.asarray(legacy["probabilities"]))))
                                    if current["choices"] != legacy["choices"] or error > 1e-3:
                                        raise ValueError("Counterfactual API changed ordinary native inference")
                                    maximum_error = max(maximum_error, error)
                            selected = {side: response.get("choice") for side, response in responses.items()}
                            search = None
                            if arm == "known_search" and "probabilities" in responses[role]:
                                digests = {side: record_digest(player.battles[tags[side]]) for side in names}
                                search = known_team_action(
                                    client, player, state=update.state, own_tag=tags[role], opponent_tag=tags[other],
                                    role=role, own_proposal=responses[role], opponent_proposal=responses[other],
                                    seed=game.controller_seed + rounds, started=begin, seconds=settings["decision_seconds"],
                                    max_actions=settings["max_actions"], samples=settings["chance_samples"],
                                )
                                if any(record_digest(player.battles[tags[side]]) != digests[side] for side in names):
                                    raise ValueError("Counterfactual inference mutated a live parent history")
                                selected[role] = search.choice
                                decisions.append(dict(arm=arm, game=index, round=rounds, role=role, **asdict(search)))
                                if search.failures or not search.complete_grid:
                                    raise ValueError("Known-team engineering grid failed: " + json.dumps(search.failures))
                            for side, response in responses.items():
                                if "probabilities" in response:
                                    action = player.update({"op": "commit", "battle_tag": tags[side],
                                                            "choice": selected[side]})["native_action"]
                                    # The reference must see the actually executed override,
                                    # so the next comparison uses an identical temporal prefix.
                                    control.battles[tags[side]]["rl2"].zero_()
                                    control.battles[tags[side]]["rl2"][0, 0, action + 1] = 1
                            elapsed = time.monotonic() - begin
                            maximum_seconds = max(maximum_seconds, elapsed)
                            if elapsed > settings["measured_deadline_seconds"]:
                                raise TimeoutError("Complete engineering decision exceeded five seconds")
                            update = client.step_session(session, [selected["p1"], selected["p2"]], include_state=True)
                            if update.choice_errors:
                                raise ValueError("Authoritative choice rejected: " + json.dumps(update.choice_errors))
                            rounds += 1
                        row = {"arm": arm, "index": index, "schedule": asdict(game), "rounds": rounds,
                               "turns": update.turn, "complete": update.ended,
                               "winner": update.winner,
                               "score": float(update.winner == "Candidate") if update.ended and update.winner else None}
                        if max_rounds == 2000 and not update.ended:
                            raise ValueError("Known-team canary did not finish within its round cap")
                        rows.append(row)
                        (out / "PROGRESS.json").write_text(json.dumps({"rows": rows, "decisions": decisions}) + "\n")
                        print(json.dumps({"arm": arm, "games": index + 1, "turns": update.turn,
                                          "search_decisions": len(decisions), "maximum_seconds": maximum_seconds}), flush=True)
                    finally:
                        client.close_session(session)
                        for side in names:
                            player.update({"op": "close", "battle_tag": tags[side]})
                            control.update({"op": "close", "battle_tag": tags[side]})
                    if player.battles or control.battles:
                        raise ValueError("Counterfactual records leaked past a completed game")
    except Exception as error:
        failures.append({"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        unchanged = sha256(checkpoint) == before and model_digest(runtime.model) == initial_model_digest
        result = {
            "passed": not failures and len(rows) == 2 * games and unchanged,
            "protocol": settings, "rows": rows, "search_decisions": decisions,
            "ordinary_probability_max_error": maximum_error, "maximum_complete_decision_seconds": maximum_seconds,
            "parent_histories_unchanged": not failures, "record_cleanup_passed": not failures,
            "weights_unchanged": unchanged, "failures": failures,
            "in_memory_model_sha256": initial_model_digest,
            "fallbacks": sum(not decision["complete_grid"] for decision in decisions),
            "elapsed_seconds": time.monotonic() - started,
            "privileged_known_teams": True, "playing_strength_confirmation": False,
            "public_team_inference_tested": False, "gpu_requested": False,
            "interpretation": "Engineering mechanics/cache canary with complete simulator teams. Scores cannot establish improved public-information play.",
        }
        (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError("Native engineering canary did not pass")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--known-teams-diagnostic", action="store_true", required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.pool, args.reference_source, args.out,
        games=args.games, max_rounds=args.max_rounds, seed=args.seed)


if __name__ == "__main__":
    main()
