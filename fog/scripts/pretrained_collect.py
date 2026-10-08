"""Resumable exact-team collection on censored, authoritative Gen9OU games."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time

from fog.env.showdown_direct import ShowdownDirectClient
from fog.env.showdown_local import LocalShowdownBattle
from fog.env.showdown_local_runner import _choice, bind_fog_player
from fog.env.teams import to_packed
from fog.eval.showdown_direct_selection import matched_schedule
from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256
from fog.pretrained.actions import request_choice
from fog.pretrained.labels import exact_team_tokens, submitted_labels
from fog.pretrained.process import DecisionProcess
from fog.pretrained.runtime import interpreter_path, native_environment


def normalize_choice(choice):
    return choice.removeprefix("/choose ").removeprefix("/").strip()


def collect_game(client, native, foul_play, learner, teams, game, *, tag, opponent, max_turns=300):
    from poke_env.battle import Battle
    import logging
    from fog.constants.tables import TABLES

    names = {"p1": "CollectorOne", "p2": "CollectorTwo"}
    pastes = {"p1": teams[game.p1_team], "p2": teams[game.p2_team]}
    truths = {role: exact_team_tokens(paste, TABLES) for role, paste in pastes.items()}
    views = {
        role: Battle(tag + "-" + role, names[role], logging.getLogger(__name__), gen=9)
        for role in names
    }
    session, update = client.create_session(
        {
            "formatid": "gen9ou",
            "strictChoices": True,
            "seed": f"sodium,{game.showdown_seed:064x}",
            **{
                role: {"name": names[role], "team": to_packed(paste)}
                for role, paste in pastes.items()
            },
        },
        include_state=False,
    )
    native_role = "p1" if game.p1_is_a else "p2"
    records = {
        role: {"states": [], "actions": [], "observations": [], "labels": []} for role in names
    }
    failures, public_log, timings, rejections = [], [], [], []
    result, stage = None, "initialization"
    rng = random.Random(game.controller_seed)
    try:
        for _ in range(1000):
            public_log.extend(update.public_messages)
            for role in names:
                LocalShowdownBattle._apply_view_update(
                    views[role], update.messages[role], update.requests[role]
                )
            if update.ended or update.turn >= max_turns:
                break
            choices = []
            for role in names:
                is_native = role == native_role or opponent == "kakuna"
                request = update.requests[role]
                message = {
                    "op": "act" if is_native else "encode",
                    "battle_tag": tag + "-" + role,
                    "username": names[role],
                    "messages": update.messages[role],
                    "request": request,
                    "choose": True,
                }
                stage = "native " + role + " " + message["op"]
                response = native.request(message)
                chosen = response.get("choice") if is_native else None
                needs_choice = request is not None and not request.get("wait")
                if not is_native and opponent == "foulplay":
                    stage = "foulplay " + role
                    foul_response = foul_play.request(
                        dict(message, op="act", own_team=pastes[role], seed=game.controller_seed)
                    )
                    chosen = foul_response.get("choice")
                elif not is_native and needs_choice:
                    import torch

                    # A per-decision stream makes learner sampling independent of
                    # process initialization, earlier games and shard/resume order.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(rng.getrandbits(63))
                        chosen = request_choice(
                            _choice(learner, views[role], request, rng), request
                        )
                if needs_choice and chosen is None:
                    raise ValueError("Player failed to return an owed choice")
                if "state" in response:
                    if not is_native:
                        committed = native.request(
                            {
                                "op": "commit",
                                "battle_tag": tag + "-" + role,
                                "choice": normalize_choice(chosen),
                            }
                        )
                        response["native_action"] = committed["native_action"]
                    record = records[role]
                    record["states"].append(response["state"])
                    record["actions"].append(response["native_action"])
                    record["observations"].append(response["observation"])
                    other = "p2" if role == "p1" else "p1"
                    record["labels"].append(submitted_labels(views[role], truths[other], TABLES))
                    timings.append(
                        foul_response["elapsed_seconds"]
                        if not is_native and opponent == "foulplay"
                        else response["elapsed_seconds"]
                    )
                choices.append(chosen)
            stage = "Showdown choice"
            update = client.step_session(session, choices, include_state=False)
            for retry in range(2):
                if not update.choice_errors:
                    break
                for role, error in update.choice_errors.items():
                    if not (role == native_role or opponent == "kakuna"):
                        raise ValueError("External teacher choice rejected: " + error)
                    if "trapped" not in error.lower():
                        raise ValueError("Unexpected native choice rejection: " + error)
                    rejections.append({"role": role, "turn": update.turn, "error": error})
                    native.request({"op": "rollback", "battle_tag": tag + "-" + role})
                    request = update.requests[role]
                    response = native.request(
                        {
                            "op": "act",
                            "battle_tag": tag + "-" + role,
                            "username": names[role],
                            "messages": update.messages[role],
                            "request": request,
                            "choose": True,
                        }
                    )
                    choices[0 if role == "p1" else 1] = response["choice"]
                    # The rejected switch was never executed. Replace its pending
                    # training row after restoring the temporal policy state.
                    record = records[role]
                    record["states"][-1] = response["state"]
                    record["actions"][-1] = response["native_action"]
                    record["observations"][-1] = response["observation"]
                    timings.append(response["elapsed_seconds"])
                public_log.extend(update.public_messages)
                update = client.step_session(session, choices, include_state=False)
            if update.choice_errors:
                raise ValueError(
                    "Authoritative Showdown rejected choice: " + str(update.choice_errors)
                )
        valid = (
            update.ended
            and update.winner in names.values()
            and all(records[r]["states"] for r in names)
        )
        if valid:
            for role in names:
                terminal = native.request(
                    {
                        "op": "terminal",
                        "battle_tag": tag + "-" + role,
                        "username": names[role],
                        "messages": update.messages[role],
                        "request": update.requests[role],
                        "choose": False,
                    }
                )["state"]
                won = update.winner == names[role]
                if bool(terminal["battle_won"]) != won or bool(terminal["battle_lost"]) == won:
                    raise ValueError(
                        "Native terminal perspective does not agree with Showdown winner"
                    )
                records[role]["states"].append(terminal)
                records[role]["actions"].append(-1)
                records[role]["outcome"] = 1.0 if won else -1.0
        result = {
            "match_id": tag,
            "schedule": asdict(game),
            "opponent": opponent,
            "submitted_teams": pastes,
            "perspectives": records,
            "public_log": public_log,
            "winner": update.winner,
            "turns": update.turn,
            "training_eligible": valid,
            "failures": failures,
            "fallbacks": [],
            "choice_rejections": rejections,
            "decision_seconds": timings,
        }
        return result
    except Exception as error:
        failures.append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "turn": update.turn,
            }
        )
        result = {
            "match_id": tag,
            "schedule": asdict(game),
            "opponent": opponent,
            "submitted_teams": pastes,
            "perspectives": records,
            "public_log": public_log,
            "winner": update.winner,
            "turns": update.turn,
            "training_eligible": False,
            "failures": failures,
            "fallbacks": [],
            "choice_rejections": rejections,
            "decision_seconds": timings,
        }
        return result
    finally:
        client.close_session(session)
        for role in names:
            for label, worker in (("native", native), ("foulplay", foul_play)):
                if worker.process is None or (label == "foulplay" and opponent != "foulplay"):
                    continue
                try:
                    worker.request({"op": "close", "battle_tag": tag + "-" + role})
                except Exception as error:
                    failures.append(
                        {
                            "stage": label + " cleanup",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    worker.close(force=True)
        if result is not None and failures:
            result["training_eligible"] = False
        if hasattr(learner, "_battle_finished"):
            for view in views.values():
                learner._battle_finished(view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "development"], default="train")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--foul-play-python", type=Path, required=True)
    parser.add_argument("--foul-play-source", type=Path, required=True)
    parser.add_argument("--learner", action="append", default=[], help="CONFIG.yaml=CHECKPOINT.pt")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke-opponent", choices=["foulplay", "kakuna", "learner"])
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--slots", type=Path, help="Frozen selected slots for a diagnostic/recovery run"
    )
    parser.add_argument("--attempt-offset", type=int, default=0)
    args = parser.parse_args()
    import torch
    import yaml
    from fog.models.agent import FogAgent
    from fog.env.player import FogPlayer

    torch.set_num_threads(1)
    root = Path.cwd()
    assert sha256(args.checkpoint) == KAKUNA_34_SHA256
    if args.games % 20 and not args.smoke_opponent:
        raise ValueError("Collection counts must be divisible by 20 for seat-balanced 40/30/30")
    teams = json.loads((args.pools / (args.split + ".json")).read_text())["teams"]
    slots = json.loads(args.slots.read_text()) if args.slots else None
    if slots is not None and (
        not slots
        or len(slots) != len(set(slots))
        or any(type(slot) is not int or not 0 <= slot < args.games for slot in slots)
    ):
        raise ValueError("Selected collection slots must be unique indices in the frozen schedule")
    args.out.mkdir(parents=True, exist_ok=True)
    learners = []
    for specification in args.learner:
        cfg_path, weight_path = specification.split("=", 1)
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        model = FogAgent.from_tables(cfg).eval()
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True), strict=True
        )
        learners.append(bind_fog_player(FogPlayer(model, temperature=1.0)))
    if not learners and args.smoke_opponent != "kakuna" and args.smoke_opponent != "foulplay":
        raise ValueError("Fresh collection requires a frozen historical learner")
    descriptor = {
        "version": 6,
        "seed": args.seed,
        "games": args.games,
        "split": args.split,
        "pool_sha256": sha256(args.pools / (args.split + ".json")),
        "native_sha256": sha256(args.checkpoint),
        "temperature": 1.0,
        "opponent_mix": [0.4, 0.3, 0.3],
        "foul_play_search_ms": 500,
        "foul_play_sampling": "bounded-bernoulli-move-proposals-v1",
        "foul_play_processes": "persistent-spawn-v1",
        "slots": slots,
        "attempt_offset": args.attempt_offset,
        "learners": {s: sha256(Path(s.split("=", 1)[1])) for s in args.learner},
        "smoke_opponent": args.smoke_opponent,
        "max_attempts": args.max_attempts,
        "retry_policy": "same team/opponent/seat assignment, new deterministic battle seed; retain failed attempts",
    }
    manifest = args.out / "collection.json"
    if manifest.exists() and json.loads(manifest.read_text()) != descriptor:
        raise ValueError("Collection identity cannot change on resume")
    if not manifest.exists():
        manifest.write_text(json.dumps(descriptor, indent=2) + "\n")
    native = DecisionProcess(
        [
            interpreter_path(args.native_python),
            "-m",
            "fog.pretrained.native_worker",
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
        log=args.out / f"native-{args.shard}.log",
        env=native_environment(root),
    )
    foul = DecisionProcess(
        [
            interpreter_path(args.foul_play_python),
            str(root / "integrations/foul_play_worker.py"),
            "--source",
            str(args.foul_play_source.resolve()),
            "--prewarm",
            "--trace-deadline",
            "--seed",
            str(args.seed + args.shard),
        ],
        cwd=root,
        log=args.out / f"foulplay-{args.shard}.log",
        env=os.environ.copy(),
    )
    schedule = matched_schedule(len(teams), max(4, args.games), args.seed)
    # Pair adjacent games by seat for each opponent independently.
    mix = ["foulplay"] * 4 + ["kakuna"] * 3 + ["learner"] * 3
    random.Random(args.seed).shuffle(mix)
    mix = [opponent for opponent in mix for _ in range(2)]
    counts, started = Counter(), time.monotonic()
    native.start()
    if args.smoke_opponent != "kakuna":
        foul.start()
    try:
        with ShowdownDirectClient() as client:
            for index, game in enumerate(schedule[: args.games]):
                if index % args.shards != args.shard:
                    continue
                if slots is not None and index not in slots:
                    continue
                path = args.out / f"{index:06d}.json.gz"
                if path.exists():
                    import gzip as gzip_reader

                    existing = json.loads(gzip_reader.decompress(path.read_bytes()))
                    if existing["training_eligible"]:
                        continue
                # Each adjacent pair uses opposite seats, equalizing each source.
                if index % 2:
                    game = type(game)(
                        not schedule[index - 1].p1_is_a,
                        game.p1_team,
                        game.p2_team,
                        game.showdown_seed,
                        game.controller_seed,
                    )
                opponent = args.smoke_opponent or mix[index % 20]
                learner = (
                    learners[(index // 20) % len(learners)] if learners else "simpleheuristics"
                )
                for attempt in range(args.attempt_offset, args.attempt_offset + args.max_attempts):
                    tag = f"battle-gen9ou-exact-{args.seed}-{index}-{attempt}"
                    retry_seed = game.showdown_seed ^ int.from_bytes(
                        hashlib.sha256(tag.encode()).digest(), "big"
                    )
                    attempted = type(game)(
                        game.p1_is_a, game.p1_team, game.p2_team, retry_seed, game.controller_seed
                    )
                    try:
                        result = collect_game(
                            client,
                            native,
                            foul,
                            learner,
                            teams,
                            attempted,
                            tag=tag,
                            opponent=opponent,
                        )
                    except Exception as error:
                        result = {
                            "match_id": tag,
                            "opponent": opponent,
                            "training_eligible": False,
                            "failures": [{"type": type(error).__name__, "message": str(error)}],
                            "fallbacks": [],
                            "schedule": asdict(attempted),
                        }
                        native.close(force=True)
                        foul.close(force=True)
                        native.start()
                        if args.smoke_opponent != "kakuna":
                            foul.start()
                    if result["training_eligible"]:
                        break
                    failed = args.out / "failed_attempts" / f"{index:06d}-{attempt}.json.gz"
                    failed.parent.mkdir(exist_ok=True)
                    failed.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
                payload = json.dumps(result, separators=(",", ":")).encode()
                temporary = path.with_suffix(".part")
                temporary.write_bytes(gzip.compress(payload, mtime=0))
                temporary.replace(path)
                counts["eligible" if result["training_eligible"] else "failed"] += 1
                print(
                    json.dumps(
                        {
                            "game": index,
                            "opponent": opponent,
                            "turns": result.get("turns"),
                            "counts": dict(counts),
                            "elapsed_seconds": round(time.monotonic() - started),
                            "failure": result.get("failures"),
                        }
                    ),
                    flush=True,
                )
    finally:
        native.close()
        foul.close()


if __name__ == "__main__":
    main()
