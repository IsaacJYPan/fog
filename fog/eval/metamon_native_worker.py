"""Run unmodified Metamon reconstruction in its isolated dependency environment.

Invoke this file directly with Metamon and its supported poke-env on PYTHONPATH. Input JSONL
contains public replays and FOG's public extracted facts, NEVER the submitted truth sheets.
The worker calls the native forward/backward parser and the same sequence checks as save_to_disk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


def seed_for(rid: str, role: str, draw: int, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{rid}:{role}:{draw}".encode()).digest()[:8], "little")


def run(input_path: Path, out: Path, *, draws: int, seed: int, source: Path) -> None:
    from metamon.backend.replay_parser import backward, forward
    from metamon.backend.replay_parser.parse_replays import ReplayParser
    from metamon.backend.team_prediction.predictor import ReplayPredictor
    from metamon.backend.team_prediction.team import PokemonSet, TeamSet
    from metamon.backend.showdown_dex import Dex
    from metamon.backend.team_prediction.usage_stats.stat_reader import (
        EARLIEST_USAGE_STATS_DATE, LATEST_USAGE_STATS_DATE,
    )

    if out.exists():
        raise FileExistsError(out)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("native Metamon source must be unmodified")

    def export(team):
        return [copy.deepcopy(p.to_dict()) for p in team.pokemon]

    class RecordingPredictor(ReplayPredictor):
        def predict(self, team, **kwargs):
            role = "p1" if len(self.calls) == 0 else "p2"
            sample_seed = seed_for(self.rid, role, self.draw, seed)
            random.seed(sample_seed)
            np.random.seed(sample_seed % 2**32)
            prediction = super().predict(team, **kwargs)
            self.calls.append({"revealed": export(team), "predicted": export(prediction)})
            return prediction

    predictor = RecordingPredictor()
    control = ReplayPredictor()
    parser = ReplayParser(team_predictor=predictor, sleep_on_handled_exception=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    with out.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(records):
            raw = record["replay"]
            rid = raw["id"]
            result = {"id": rid, "revision": revision, "poke_env": importlib.metadata.version("poke-env"),
                      "samples": [], "fog_evidence_predictions": [], "errors": []}
            time_played = datetime.fromtimestamp(int(raw["uploadtime"]))
            replay = forward.ParsedReplay(gameid=rid, format=raw["formatid"], time_played=time_played)
            try:
                replay = forward.forward_fill(replay, parser.clean_log(raw))
                result["forward_ok"] = True
                ratings = replay.ratings
                result["native_revealed"] = {
                    role: [PokemonSet.from_ReplayPokemon(p, gen=9).to_dict() for p in team]
                    for role, team in (("p1", replay[-1].pokemon_1), ("p2", replay[-1].pokemon_2))
                }
            except Exception as error:
                result["forward_ok"] = False
                result["errors"].append({"stage": "forward", "type": type(error).__name__,
                                         "message": str(error)[:1200]})
                # Prior/completion control still runs on FOG evidence even if native parsing fails.
                ratings = [None, None]
                for line in raw["log"].splitlines():
                    parts = line.split("|")
                    if len(parts) > 5 and parts[1] == "player" and parts[5].isdigit():
                        ratings[0 if parts[2] == "p1" else 1] = int(parts[5])
            result["usage"] = {}
            for side, role in enumerate(("p1", "p2")):
                stats = control.get_usage_stats(raw["formatid"], time_played.date(),
                                                rating=ratings[side], gameid=rid)
                requested = control.bin_usage_stats_dates(time_played.date())
                effective = [min(max(d, EARLIEST_USAGE_STATS_DATE), LATEST_USAGE_STATS_DATE)
                             for d in requested]
                values, errors = {}, {}
                for row in record["fog_teams"][role]:
                    name = row["name"]
                    try:
                        values[name] = copy.deepcopy(stats[name])
                    except KeyError as error:
                        errors[name] = str(error)
                result["usage"][role] = {
                    "rank": stats.rank, "parsed_rating": ratings[side],
                    "requested_dates": list(map(str, requested)),
                    "effective_primary_dates": list(map(str, effective)),
                    "all_tiers_fallback_dates": [str(EARLIEST_USAGE_STATS_DATE), str(LATEST_USAGE_STATS_DATE)],
                    "species": values, "errors": errors,
                }
            for draw in range(draws):
                fog_prediction = {}
                for side, role in enumerate(("p1", "p2")):
                    try:
                        rows = copy.deepcopy(record["fog_teams"][role])
                        for row in rows:
                            row["moves"] = [Dex.from_gen(9).moves[m]["name"]
                                            if m in Dex.from_gen(9).moves else m for m in row["moves"]]
                        members = [PokemonSet.from_dict(row) for row in rows]
                        team = TeamSet(lead=members[0], reserve=members[1:], format=raw["formatid"])
                        sample_seed = seed_for(rid, role, draw, seed)
                        random.seed(sample_seed)
                        np.random.seed(sample_seed % 2**32)
                        completed = control.predict(team, date=time_played.date(),
                                                    rating=ratings[side], gameid=rid)
                        if not team.is_consistent_with(completed):
                            raise ValueError("native prediction contradicts FOG evidence")
                        fog_prediction[role] = export(completed)
                    except Exception as error:
                        result["errors"].append({"stage": "fog_evidence_prediction", "draw": draw,
                                                 "role": role, "type": type(error).__name__,
                                                 "message": str(error)[:1200]})
                result["fog_evidence_predictions"].append(fog_prediction)
                if result["forward_ok"]:
                    predictor.calls, predictor.rid, predictor.draw = [], rid, draw
                    try:
                        # backward_fill deep-copies its input; never reuse an already-filled state.
                        povs = backward.backward_fill(replay, team_predictor=predictor)
                        lengths = []
                        for pov in povs:
                            states, actions = parser.povreplay_to_seq(pov)
                            lengths.append([len(states), len(actions)])
                        result["samples"].append({"draw": draw, "sides": predictor.calls,
                                                   "sequence_lengths": lengths})
                    except Exception as error:
                        result["errors"].append({"stage": "backward_or_sequence", "draw": draw,
                                                 "type": type(error).__name__,
                                                 "message": str(error)[:1200]})
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(f"native reconstruction {index + 1}/{len(records)}: {rid}, "
                  f"{len(result['samples'])}/{draws} accepted", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args()
    if args.draws < 1:
        ap.error("draws must be positive")
    run(args.input, args.out, draws=args.draws, seed=args.seed, source=args.source)
