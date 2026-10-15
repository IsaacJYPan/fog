"""Bounded CPU feasibility screen for verified-prefix curriculum reinforcement learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fog.config import load_config
from fog.training.teacher_replay import digest, read_record, shard_arrays


def probe(config, initial, checkpoint, bank, out, *, records=8, horizons=(4, 16),
          seed=20261015, smoke_update=False):
    from fog.constants.tables import TABLES
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.models.agent import FogAgent
    from fog.training.backplay import reset_from_record
    from fog.training.rollout import RolloutBuffer

    manifest = json.loads(bank.read_text())
    if manifest.get("format") != "fog.backplay-probe-bank" or not manifest.get("source_replays_verified"):
        raise ValueError("Require a hash-bound verified demonstration bank")
    selected = manifest["records"][:records]
    if not selected or records < 1 or any(type(n) is not int or n < 1 for n in horizons):
        raise ValueError("Require records and positive remaining-decision horizons")
    if smoke_update and any(row["split"] != "train" for row in selected):
        raise ValueError("An optimizer smoke check cannot use development data")
    if out.exists():
        raise FileExistsError("Preserve existing probe outputs")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    before = {str(path): digest(path) for path in (initial, checkpoint, bank)}
    agents = {}
    for name, path in (("initial", initial), ("bc1000", checkpoint)):
        agent = FogAgent.from_tables(config).cpu().eval()
        agent.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        agents[name] = agent
    settings = {
        "seed": seed, "horizons": horizons, "records": len(selected), "temperature": 1.0,
        "opponent": "simpleheuristics", "arms": ["initial", "bc1000", "simpleheuristics"],
        "inputs": before, "optimizer_smoke_only": smoke_update,
        "fresh_chance_after_prefix": True, "prefix_actions_in_ppo": False,
        "historical_outcomes_used_as_continuation_labels": False,
        "public_inputs_only": True, "own_submitted_team_is_private_input": True,
        "complete_opponent_team_is_label_only": True, "playing_strength_confirmation": False,
        "five_second_deployment_deadline_tested": False,
    }
    (out / "PROTOCOL.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows, failures, last_batch = [], [], None
    started = time.monotonic()
    try:
        with ShowdownDirectClient() as client:
            for index, item in enumerate(selected):
                source, shard = bank.parent / item["source"], bank.parent / item["shard"]
                if digest(source) != item["source_sha256"] or digest(shard) != item["shard_sha256"]:
                    raise ValueError("A verified bank input changed")
                record = read_record(source)
                with np.load(shard, allow_pickle=False) as saved:
                    reference = {name: saved[name] for name in saved.files}
                truth = {role: truth_tokens_from_paste(record["submitted_teams"][role], TABLES) for role in ("p1", "p2")}
                for role_index, role in enumerate(("p1", "p2")):
                    other = "p2" if role == "p1" else "p1"
                    for horizon in horizons:
                        # Same fresh chance stream across arms; distinct from demonstration future.
                        episode_seed = seed + index * 10000 + role_index * 1000 + horizon
                        for arm in settings["arms"]:
                            torch.manual_seed(episode_seed)
                            buffer = RolloutBuffer(gamma=1.0, lam=0.95)
                            actors = {side: bind_fog_player(FogPlayer(
                                agents.get(arm) if side == role else None, temperature=1.0)) for side in ("p1", "p2")}
                            for side, actor in actors.items():
                                actor.record_mirror = False
                                actor.private_state_truth_provider = lambda _, s=side: truth[s].copy()
                            if arm in agents:
                                actors[role].truth_provider = lambda _: truth[other].copy()
                                actors[role].attach_buffer(buffer)
                            tag = f"battle-gen9ou-backplay-{index}-{role}-{horizon}-{arm}"
                            prefix_started = time.monotonic()
                            reset = reset_from_record(client, record, role=role, remaining=horizon, actors=actors, tag=tag,
                                                      future_seed=f"sodium,{episode_seed:064x}")
                            try:
                                offset = 0 if role == "p1" else item["decisions"]["p1"]
                                ref_index = offset + reset.counters[role]
                                fragment = shard_arrays([reset.observation])
                                if any(not np.array_equal(value[0], reference[name][ref_index]) for name, value in fragment.items()):
                                    raise ValueError("Reset observation/history differs from its verified teacher shard")
                                if buffer.n_decisions or buffer.n_games:
                                    raise ValueError("A demonstration-prefix action reached the on-policy rollout")
                                root_request = reset.local.update.requests[role]
                                root_turn = reset.local.update.turn
                                legal_count = int(reset.observation["legality_mask"].sum())
                                prefix_seconds = time.monotonic() - prefix_started
                                controllers = {side: actors[side] if side == role and arm in agents else "simpleheuristics"
                                               for side in ("p1", "p2")}
                                continued = time.monotonic()
                                result = play_local_battle(client, {}, controllers["p1"], controllers["p2"],
                                                           seed=episode_seed, initial_battle=reset.local, max_choice_rounds=1000)
                                elapsed = time.monotonic() - continued
                                if result.truncated:
                                    raise ValueError("Curriculum continuation was truncated")
                                won = result.p1_won if role == "p1" or result.p1_won is None else not result.p1_won
                                score = 0.5 if won is None else float(won)
                                if arm in agents:
                                    batch = buffer.to_batch("cpu")
                                    expected_outcome = 2.0 * score - 1.0
                                    if not torch.all(batch["game_outcomes"] == expected_outcome):
                                        raise ValueError("Continuation targets disagree with the newly played outcome")
                                    if not batch.get("belief_targets") or not any(bool(x.any()) for x in batch["belief_valid"].values()):
                                        raise ValueError("Continuation lost exact-team belief supervision")
                                    if arm == "initial":
                                        last_batch = batch
                                row = {"source_id": item["id"], "split": item["split"], "role": role,
                                       "remaining_historical_decisions": horizon, "arm": arm,
                                       "prefix_decisions": reset.counters[role], "root_turn": root_turn,
                                       "root_forced_switch": bool(root_request.get("forceSwitch")),
                                       "root_legal_actions": legal_count, "root_matches_verified_shard": True,
                                       "score": score, "archived_outcome": record["perspectives"][role]["outcome"],
                                       "continuation_turns": result.turns - root_turn, "on_policy_decisions": buffer.n_decisions,
                                       "prefix_seconds": prefix_seconds, "continuation_seconds": elapsed}
                                rows.append(row)
                                (out / "PROGRESS.json").write_text(json.dumps(rows) + "\n")
                                print(json.dumps(row), flush=True)
                            finally:
                                reset.close()
        optimizer_check = None
        if smoke_update:
            from fog.training.ppo import ppo_update

            if last_batch is None:
                raise ValueError("No completed train-only continuation for the optimizer check")
            agent = agents["initial"]
            optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-5)
            cfg = {"ppo": {"epochs": 1, "minibatch_size": 64, "rollout_temperature": 1.0,
                           "max_grad_norm": 1.5, "value_target": "outcome"},
                   "loss": {"value_coef": 0.25, "belief_coef": 0.1, "belief_chain_coef": 0.1,
                            "self_belief_coef": 0.0, "self_belief_chain_coef": 0.0, "preview_coef": 0.0}}
            optimizer_check = ppo_update(agent, optimizer, last_batch, cfg)
            if not all(np.isfinite(value) for value in optimizer_check.values()):
                raise ValueError("Optimizer smoke check produced nonfinite metrics")
            optimizer_check["nonzero_shared_gradient_tensors"] = sum(
                p.grad is not None and bool(p.grad.abs().sum()) for name, p in agent.named_parameters()
                if name.startswith(("backbone.", "shared_latent.", "structured_encoder.")))
            if not optimizer_check["nonzero_shared_gradient_tensors"]:
                raise ValueError("Continuation update did not train the shared representation")
    except Exception as error:
        failures.append({"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        unchanged = all(digest(Path(path)) == sha for path, sha in before.items())
        result = {"passed": not failures and len(rows) == len(selected) * 2 * len(horizons) * 3 and unchanged,
                  "protocol": settings, "rows": rows, "failures": failures,
                  "input_checkpoints_unchanged": unchanged, "elapsed_seconds": time.monotonic() - started,
                  "optimizer_check": locals().get("optimizer_check"),
                  "interpretation": "Reset, fresh-continuation and label feasibility only; no curriculum learning or full-game strength claim."}
        (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "initial", "checkpoint", "bank", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--smoke-update", action="store_true")
    args = parser.parse_args()
    probe(load_config(args.config), args.initial, args.checkpoint, args.bank, args.out,
          records=args.records, horizons=tuple(args.horizons), seed=args.seed, smoke_update=args.smoke_update)


if __name__ == "__main__":
    main()
