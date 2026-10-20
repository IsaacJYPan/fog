"""Fresh disjoint-roster public belief contexts from unchanged Kakuna self-play.

Collection runs in the existing evaluator environment. Encoding runs in the
pinned native environment; neither receives a fitted candidate's actions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
import time


def schedule(groups, seed):
    from fog.eval.showdown_direct_selection import ScheduledGame

    rng = random.Random(seed)
    games = []
    for group in range(groups):
        x, y = group * 2, group * 2 + 1
        for seat, left, right in [(True, x, y), (True, y, x), (False, x, y), (False, y, x)]:
            games.append(
                ScheduledGame(seat, left, right, rng.getrandbits(256), rng.randrange(2**31))
            )
    return games


def collect(config, out, *, native_python):
    import sys
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.pretrained.frozen_belief import digest
    from fog.pretrained.process import DecisionProcess
    from fog.pretrained.runtime import native_environment
    from fog.scripts.pretrained_collect import collect_game

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    teams_path = Path(config["fresh_teams"])
    if (
        digest(teams_path) != config["fresh_teams_sha256"]
        or digest(config["checkpoint"]) != config["native_sha256"]
    ):
        raise ValueError("Fresh cohort or protected native identity changed")
    teams = json.loads(teams_path.read_text())["teams"]
    if len(teams) != 2 * config["fresh_groups"]:
        raise ValueError("Fresh cohort needs two unique rosters per independent group")
    games = schedule(config["fresh_groups"], config["fresh_seed"])
    (out / "collection.json").write_text(
        json.dumps(
            {
                "config": config,
                "schedule": [asdict(g) for g in games],
                "opponent": "unchanged Kakuna",
                "both_teams_known_for_labels_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    worker = DecisionProcess(
        [
            str(Path(native_python).absolute()),
            "-m",
            "fog.pretrained.native_worker",
            "--checkpoint",
            str(Path(config["checkpoint"]).resolve()),
            "--seed",
            str(config["fresh_seed"]),
        ],
        cwd=Path.cwd(),
        log=out / "native.log",
        env=native_environment(Path.cwd()),
        deadline=5.0,
    )
    unused = DecisionProcess([sys.executable], cwd=Path.cwd(), log=out / "unused.log")
    worker.start()
    started = time.monotonic()
    rows = []
    try:
        with ShowdownDirectClient() as client:
            for index, game in enumerate(games):
                tag = f"battle-gen9ou-prior-context-{config['fresh_seed']}-{index}"
                result = collect_game(
                    client,
                    worker,
                    unused,
                    "simpleheuristics",
                    teams,
                    game,
                    tag=tag,
                    opponent="kakuna",
                )
                target = out / f"{index:06d}.json.gz"
                target.write_bytes(gzip.compress(json.dumps(result).encode()))
                rows.append({"path": target.name, "sha256": digest(target), "group": index // 4})
                # Preserve the complete failed record before stopping. Do not
                # replace an inconvenient game or discard a failed trajectory.
                if (
                    not result["training_eligible"]
                    or result["failures"]
                    or result["fallbacks"]
                    or result.get("choice_rejections")
                    or max(result["decision_seconds"], default=0) > 5.0
                ):
                    raise ValueError(
                        "Fresh collection failed its trace/deadline gate: " + target.name
                    )
                if (index + 1) % 4 == 0:
                    print(
                        json.dumps(
                            {
                                "fresh_games": index + 1,
                                "groups": (index + 1) // 4,
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
    finally:
        worker.close(force=True)
        (out / "INVENTORY.json").write_text(
            json.dumps(
                {"records": rows, "expected_games": len(games), "native_policy_changed": False},
                indent=2,
            )
            + "\n"
        )
    return out


def encode(config, source, out):
    import torch
    from fog.pretrained.frozen_belief import DATA_FORMAT, DATA_VERSION, digest
    from fog.pretrained.kakuna import load_kakuna
    from fog.pretrained.latent_cache import tensor_digest
    from fog.pretrained.loading import SequenceLoader, read_record
    from fog.pretrained.state import atomic_save
    from fog.team_identity import team_group_key

    torch.set_num_threads(1)
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if digest(config["checkpoint"]) != config["native_sha256"]:
        raise ValueError("Protected native checkpoint changed")
    if digest(config["fresh_teams"]) != config["fresh_teams_sha256"]:
        raise ValueError("Declared fresh rosters changed")
    teams = json.loads(Path(config["fresh_teams"]).read_text())["teams"]
    fixed = schedule(config["fresh_groups"], config["fresh_seed"])
    runtime = load_kakuna(Path(config["checkpoint"]))
    fingerprint = tensor_digest(runtime.model.state_dict())
    loader = SequenceLoader(runtime, sequence_length=1000)
    inventory = json.loads((source / "INVENTORY.json").read_text())
    if len(inventory["records"]) != config["fresh_groups"] * 4:
        raise ValueError("The new development cohort is incomplete")
    rows, positions = [], 0
    with torch.inference_mode():
        for game, item in enumerate(inventory["records"]):
            path = source / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("Fresh collected trajectory changed")
            record = read_record(path)
            if (
                not record["training_eligible"]
                or record["failures"]
                or record["fallbacks"]
                or record.get("choice_rejections")
            ):
                raise ValueError("Unverified fresh trajectory")
            if record["schedule"] != asdict(fixed[game]):
                raise ValueError("Collected game differs from the fixed schedule")
            for role, index in [("p1", fixed[game].p1_team), ("p2", fixed[game].p2_team)]:
                if record["submitted_teams"][role] != teams[index]:
                    raise ValueError("Collected team differs from its predeclared roster")
            rosters = sorted(
                team_group_key(t, mode="roster") for t in record["submitted_teams"].values()
            )
            for pov in (0, 1):
                example, targets = loader.load(("fresh", str(path), pov))
                role = random.Random(pov).choice(["p1", "p2"])
                if (
                    len(example.actions) != len(record["perspectives"][role]["actions"]) - 1
                    or int(example.time_idxs[0]) != 0
                ):
                    raise ValueError("Fresh encoding cropped a native history")
                points = sorted({0, len(example.actions) // 2, len(example.actions) - 1})
                hidden = runtime.model.traj_encoder.init_hidden_state(1, torch.device("cpu"))
                values = []
                for index in range(len(example.actions)):
                    latent, hidden = runtime.model.get_state_embedding(
                        obs={
                            key: value[None, index : index + 1]
                            for key, value in example.obs.items()
                        },
                        rl2s=example.rl2s[None, index : index + 1],
                        time_idxs=example.time_idxs[None, index : index + 1],
                        hidden_state=hidden,
                    )
                    if index in points:
                        values.append(latent[0, 0].clone())
                name = f"development/{game:06d}-{pov}.pt"
                atomic_save(
                    {
                        "latent": torch.stack(values),
                        "labels": {key: value[points] for key, value in targets.items()},
                    },
                    out / name,
                )
                rows.append(
                    {
                        "path": name,
                        "sha256": digest(out / name),
                        "split": "development",
                        "game": path.name,
                        "pov_seed": pov,
                        "source_sha256": item["sha256"],
                        "roster_groups": rosters,
                        "positions": len(points),
                        "source_decision_indices": points,
                        "source_history_decisions": len(example.actions),
                        "group": item["group"],
                    }
                )
                positions += len(points)
            if (game + 1) % 4 == 0:
                print(json.dumps({"encoded_games": game + 1, "contexts": positions}), flush=True)
    if (
        fingerprint != tensor_digest(runtime.model.state_dict())
        or digest(config["checkpoint"]) != config["native_sha256"]
    ):
        raise ValueError("Native parameters or buffers changed during context extraction")
    manifest = {
        "format": DATA_FORMAT,
        "version": DATA_VERSION,
        "complete": True,
        "local_trial": config.get("local_trial", False),
        "native_sha256": config["native_sha256"],
        "native_dim": 900,
        "both_perspectives": True,
        "complete_chronological_history": True,
        "private_labels_separate_from_latents": True,
        "counts": {
            "development": {
                "games": len(inventory["records"]),
                "perspectives": len(rows),
                "positions": positions,
            }
        },
        "records": rows,
        "source_collection_sha256": digest(source / "collection.json"),
        "native_parameters_and_buffers_unchanged": True,
        "collection_config": config,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["collect", "encode"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--native-python", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "collect":
        collect(config, args.out, native_python=args.native_python)
    else:
        encode(config, args.source, args.out)


if __name__ == "__main__":
    main()
