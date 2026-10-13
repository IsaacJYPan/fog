"""Resumable native-agent comparisons using held-out composition/seat-balanced games."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import sys

from fog.env.showdown_direct import ShowdownDirectClient
from fog.env.teams import to_packed
from fog.eval.showdown_direct_selection import matched_schedule
from fog.eval.stats import paired_block_interval
from fog.pretrained.actions import request_choice
from fog.pretrained.kakuna import KAKUNA_34_SHA256, INTERFACE_VERSION, sha256
from fog.pretrained.process import DecisionProcess
from fog.pretrained.evaluation import SCRIPTED_OPPONENTS
from fog.pretrained.runtime import interpreter_path, native_environment
from fog.pretrained.protocol import (
    claim_confirmation,
    claim_strong_evaluation,
    foulplay_fingerprint,
)


def fallback_choice(request):
    if not request or request.get("wait"):
        return None
    if request.get("teamPreview"):
        return "team 123456"
    active = (request.get("active") or [{}])[0]
    if not any(request.get("forceSwitch", [False])):
        for index, move in enumerate(active.get("moves", [])):
            if not move.get("disabled") and move.get("pp", 1) > 0:
                return f"move {index + 1}"
    party = request.get("side", {}).get("pokemon", [])
    reviving = any(mon.get("reviving") for mon in party)
    if any(request.get("forceSwitch", [False])) or not active.get("trapped"):
        for index, mon in enumerate(party):
            fainted = mon.get("condition", "").endswith("fnt")
            if not mon.get("active") and (fainted if reviving else not fainted):
                return f"switch {index + 1}"
    return "default"


def play(client, a, b, teams, game, *, tag, foulplay=False, own_team_players=()):
    roles = {"p1": a if game.p1_is_a else b, "p2": b if game.p1_is_a else a}
    names = {
        "p1": "Candidate" if game.p1_is_a else "Opponent",
        "p2": "Opponent" if game.p1_is_a else "Candidate",
    }
    pastes = {"p1": teams[game.p1_team], "p2": teams[game.p2_team]}
    session, update = client.create_session(
        {
            "formatid": "gen9ou",
            "strictChoices": True,
            "seed": f"sodium,{game.showdown_seed:064x}",
            **{
                role: {"name": names[role], "team": to_packed(team)}
                for role, team in pastes.items()
            },
        },
        include_state=False,
    )
    failures, fallbacks, timings = [], [], []
    try:
        for _ in range(2000):
            if update.ended or update.turn >= 500:
                break
            choices = []
            for role, worker in roles.items():
                request = update.requests[role]
                message = {
                    "op": "act",
                    "battle_tag": tag + "-" + role,
                    "username": names[role],
                    "messages": update.messages[role],
                    "request": request,
                    "choose": True,
                    "seed": game.controller_seed,
                }
                if (foulplay and worker is b) or worker in own_team_players:
                    message["own_team"] = pastes[role]
                try:
                    response = worker.request(message)
                    choice = response.get("choice")
                    timings.append({"role": role, "seconds": response["elapsed_seconds"]})
                except Exception as error:
                    failures.append(
                        {"role": role, "type": type(error).__name__, "message": str(error)}
                    )
                    choice = fallback_choice(request)
                    fallbacks.append({"role": role, "choice": choice})
                    # A worker that lost its history cannot resume this evaluation game.
                    return {
                        "score": None,
                        "failures": failures,
                        "fallbacks": fallbacks,
                        "turns": update.turn,
                        "status": "failed",
                        "decision_seconds": timings,
                    }
                choices.append(request_choice(choice, request) if choice else None)
            update = client.step_session(session, choices, include_state=False)
            if update.choice_errors:
                return {
                    "score": None,
                    "failures": [{"stage": "Showdown choice", "errors": update.choice_errors}],
                    "fallbacks": [],
                    "turns": update.turn,
                    "status": "failed",
                }
        candidate = "Candidate"
        if update.ended and update.winner:
            score = float(update.winner == candidate)
        elif update.ended and any(line.startswith("|tie") for line in update.public_messages):
            score = 0.5
        else:
            score = None
        return {
            "score": score,
            "failures": failures,
            "fallbacks": fallbacks,
            "winner": update.winner,
            "turns": update.turn,
            "status": "complete" if score is not None else "unresolved",
            "decision_seconds": timings,
        }
    finally:
        client.close_session(session)
        for role, worker in roles.items():
            if worker.process is not None:
                try:
                    worker.request({"op": "close", "battle_tag": tag + "-" + role})
                except Exception:
                    worker.close(force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--stage", choices=["development", "confirmation"], default="development")
    parser.add_argument("--attempt", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--confirmation-registry",
        type=Path,
        default=Path("runs/pretrained/confirmation_attempts.json"),
    )
    parser.add_argument(
        "--finalist-sha256", help="Declared finalist for a matched strong-opponent confirmation"
    )
    parser.add_argument("--foul-play-python", type=Path)
    parser.add_argument("--foul-play-source", type=Path)
    parser.add_argument("--scripted-opponent", choices=SCRIPTED_OPPONENTS)
    args = parser.parse_args()
    if args.games < 8 or args.games % 4:
        raise ValueError("Evaluation requires at least eight games in complete crossed blocks")
    if args.scripted_opponent and (args.foul_play_python or args.stage != "development"):
        raise ValueError("Scripted opponents are development diagnostics only")
    if sha256(args.original) != KAKUNA_34_SHA256:
        raise ValueError("Comparison baseline must be unchanged checkpoint 34")
    if args.stage == "confirmation" and args.games < (256 if args.foul_play_python else 1024):
        raise ValueError("Final cohort cannot be spent on a smaller development screen")
    pool = args.pools / ("final.json" if args.stage == "confirmation" else "development.json")
    data = json.loads(pool.read_text())
    teams = data["teams"]
    if args.stage == "confirmation":
        allowed = set(data["confirmation_cohorts"][args.attempt - 1])
        teams = [team for team, key in zip(teams, data["compositions"]) if key in allowed]
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "candidate_sha256": sha256(args.candidate),
        "original_sha256": sha256(args.original),
        "interface": INTERFACE_VERSION,
        "stage": args.stage,
        "attempt": args.attempt,
        "seed": args.seed,
        "games": args.games,
        "pool_sha256": sha256(pool),
        "temperature": 1.0,
        "deadline_seconds": 5.0,
        "foulplay": bool(args.foul_play_python),
        "scripted_opponent": args.scripted_opponent,
        "automatic_promotion": False,
    }
    if args.foul_play_python:
        if not args.foul_play_source:
            raise ValueError("A strong-opponent comparison requires its source and inference priors")
        protocol["foulplay_inputs_sha256"] = foulplay_fingerprint(args.foul_play_source)
        protocol["foulplay_search_ms"] = 500
    from fog.pretrained.protocol import source_fingerprint

    protocol["sources_sha256"] = source_fingerprint(Path.cwd())
    if args.stage == "confirmation":
        if args.foul_play_python:
            claim_strong_evaluation(
                args.confirmation_registry,
                attempt=args.attempt,
                finalist_sha256=args.finalist_sha256 or protocol["candidate_sha256"],
                evaluated_sha256=protocol["candidate_sha256"],
                original_sha256=protocol["original_sha256"],
                pool_sha256=protocol["pool_sha256"],
                seed=args.seed,
                games=args.games,
                sources_sha256=protocol["sources_sha256"],
                opponent_sha256=protocol["foulplay_inputs_sha256"],
            )
        else:
            claim_confirmation(
                args.confirmation_registry,
                attempt=args.attempt,
                candidate_sha256=protocol["candidate_sha256"],
                pool_sha256=protocol["pool_sha256"],
                seed=args.seed,
                games=args.games,
                sources_sha256=protocol["sources_sha256"],
            )
    descriptor = args.out / "protocol.json"
    if descriptor.exists() and json.loads(descriptor.read_text()) != protocol:
        raise ValueError("Cannot change evaluation protocol on resume")
    descriptor.write_text(json.dumps(protocol, indent=2) + "\n")
    root = Path.cwd()

    def native(checkpoint, label):
        command = [
            interpreter_path(args.native_python),
            "-m",
            "fog.pretrained.native_worker",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--seed",
            str(args.seed),
        ]
        if sha256(checkpoint) != KAKUNA_34_SHA256:
            command.append("--trained")
        return DecisionProcess(
            command, cwd=root, log=args.out / (label + ".log"), env=native_environment(root)
        )

    a = native(args.candidate, "candidate")
    if args.foul_play_python:
        b = DecisionProcess(
            [
                interpreter_path(args.foul_play_python),
                str(root / "integrations/foul_play_worker.py"),
                "--source",
                str(args.foul_play_source.resolve()),
                "--prewarm",
            ],
            cwd=root,
            log=args.out / "foulplay.log",
            env=os.environ.copy(),
        )
    elif args.scripted_opponent:
        b = DecisionProcess(
            [sys.executable, "-m", "fog.pretrained.scripted_worker", "--kind", args.scripted_opponent],
            cwd=root,
            log=args.out / (args.scripted_opponent + ".log"),
            env=os.environ.copy(),
        )
    else:
        b = native(args.original, "original")
    a.start()
    b.start()
    rows = []
    try:
        with ShowdownDirectClient() as client:
            for index, game in enumerate(matched_schedule(len(teams), args.games, args.seed)):
                path = args.out / f"game-{index:04d}.json.gz"
                if path.exists():
                    record = json.loads(gzip.decompress(path.read_bytes()))
                else:
                    tag = f"battle-gen9ou-eval-{args.seed}-{index}"
                    record = dict(
                        play(
                            client, a, b, teams, game, tag=tag, foulplay=bool(args.foul_play_python)
                        ),
                        schedule=asdict(game),
                        index=index,
                    )
                    path.write_bytes(gzip.compress(json.dumps(record).encode(), mtime=0))
                    print(
                        json.dumps(
                            {"game": index, "score": record["score"], "status": record["status"]}
                        ),
                        flush=True,
                    )
                rows.append(record)
    finally:
        a.close()
        b.close()
    scores = [row["score"] for row in rows]
    valid = all(score is not None for score in scores)
    interval = (
        paired_block_interval(
            scores, alpha=0.025 if args.stage == "confirmation" else 0.05, seed=args.seed
        )
        if valid
        else None
    )
    report = {
        "protocol": protocol,
        "rows": rows,
        "interval": asdict(interval) if interval else None,
        "failures": sum(bool(row["failures"]) for row in rows),
        "unresolved": sum(score is None for score in scores),
        "promotion": False,
    }
    (args.out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: report[key] for key in ("interval", "failures", "unresolved")}), flush=True
    )


if __name__ == "__main__":
    main()
