"""Matched online PPO pilot with optional verified-prefix starting states."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from fog.config import load_config
from fog.training.expert_pilot import atomic_save, probability_contract, restore_rng, rng_state
from fog.training.teacher_replay import digest, read_record


class ReverseSchedule:
    """Train-only competence gate; development games remain outside optimization."""

    def __init__(self, state=None):
        self.state = deepcopy(state) if state is not None else {
            "stage": 0, "last_advance": 0, "winning_scores": [], "reset_episodes": 0,
            "pair_record": None, "full_episodes": 0, "episodes": 0,
        }

    @property
    def horizon(self):
        return (4, 8, 16, 32, 64)[self.state["stage"]]

    def record(self, *, archived_outcome, score):
        if archived_outcome == 1:
            self.state["winning_scores"].append(float(score))
            self.state["winning_scores"] = self.state["winning_scores"][-64:]

    def advance(self, batch):
        values = self.state["winning_scores"]
        n = len(values)
        if self.state["stage"] == 4 or batch - self.state["last_advance"] < 10 or n < 16:
            return False
        mean = sum(values) / n
        z = 1.96
        lower = (mean + z*z/(2*n) - z*math.sqrt(mean*(1-mean)/n + z*z/(4*n*n))) / (1+z*z/n)
        if mean >= 0.7 and lower >= 0.5:
            self.state["stage"] += 1
            self.state["last_advance"] = batch
            self.state["winning_scores"] = []
            return True
        return False


class VerifiedBank:
    def __init__(self, manifest, pool):
        from fog.team_identity import team_group_key

        self.root = manifest.parent.resolve()
        self.sha256 = digest(manifest)
        self.pool_sha256 = digest(pool)
        source = json.loads(manifest.read_text())
        if source.get("format") != "fog.backplay-probe-bank" or not source.get("source_replays_verified") or not source.get("composition_disjoint"):
            raise ValueError("Require verified, composition-disjoint prefix data")
        self.train = [row for row in source["records"] if row["split"] == "train"]
        self.development = [row for row in source["records"] if row["split"] == "development"]
        if not self.train or not self.development:
            raise ValueError("Both train resets and held-out probability-contract inputs are required")
        train_groups = {key for row in self.train for key in row["compositions"]}
        dev_groups = {key for row in self.development for key in row["compositions"]}
        self.teams = json.loads(pool.read_text())["teams"]
        if train_groups & dev_groups or {team_group_key(paste) for paste in self.teams} & dev_groups:
            raise ValueError("Training teams overlap the development compositions")
        if len(self.teams) < 2:
            raise ValueError("Require multiple training teams")
        for row in source["records"]:
            for key in ("source", "shard"):
                path = (self.root / row[key]).resolve()
                if not path.is_relative_to(self.root) or digest(path) != row[key + "_sha256"]:
                    raise ValueError("A verified curriculum input changed or escaped its folder")
        self.cache = OrderedDict()

    def record(self, index):
        if index not in self.cache:
            self.cache[index] = read_record(self.root / self.train[index]["source"])
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def contract_batch(self):
        from fog.constants.tables import TABLES
        from fog.training.dataset import _build_batch

        with np.load(self.root / self.development[0]["shard"], allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        indices = np.linspace(0, len(arrays["action"]) - 1, 8, dtype=int)
        return _build_batch(arrays, indices, TABLES, device="cpu")


class FixedHeuristics:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def choose_teampreview(self, battle):
        return "/team 123456"

    def act(self, battle):
        from fog.env.showdown_local_runner import _scripted_order

        return _scripted_order("simpleheuristics", battle, self.rng)


def collect(client, actor, anchor, bank, sampler, schedule, recipe):
    from fog.constants.tables import TABLES
    from fog.env.player import FogPlayer
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import to_packed
    from fog.training.backplay import reset_from_record
    from fog.training.rollout import RolloutBuffer

    buffer = RolloutBuffer(gamma=1.0, lam=0.95)
    episodes = []
    while buffer.n_decisions < recipe["positions_per_batch"]:
        state = schedule.state
        episode = state["episodes"]
        prefix = recipe["condition"] == "backward" and episode % 4 != 0
        seed = int(sampler.integers(0, 2**63 - 1))
        future_seed = f"sodium,{int(sampler.bit_generator.random_raw()):016x}{int(sampler.bit_generator.random_raw()):016x}{int(sampler.bit_generator.random_raw()):016x}{int(sampler.bit_generator.random_raw()):016x}"
        record = None
        if prefix:
            if state["reset_episodes"] % 2 == 0:
                state["pair_record"] = int(sampler.integers(len(bank.train)))
            record = bank.record(state["pair_record"])
            role = "p1" if state["reset_episodes"] % 2 == 0 else "p2"
            submitted = record["submitted_teams"]
        else:
            role = "p1" if state["full_episodes"] % 2 == 0 else "p2"
            indices = sampler.choice(len(bank.teams), size=2, replace=False)
            submitted = {side: bank.teams[int(i)] for side, i in zip(("p1", "p2"), indices)}
        other = "p2" if role == "p1" else "p1"
        truth = {side: truth_tokens_from_paste(paste, TABLES) for side, paste in submitted.items()}
        learner = bind_fog_player(FogPlayer(actor, temperature=1.0))
        learner.record_mirror = False
        learner.private_state_truth_provider = lambda _: truth[role].copy()
        learner.truth_provider = lambda _: truth[other].copy()
        learner.choose_teampreview = lambda _: "/team 123456"
        local_buffer = RolloutBuffer(gamma=1.0, lam=0.95)
        learner.attach_buffer(local_buffer)
        opponent = FixedHeuristics(seed)
        tag = f"battle-gen9ou-online-{episode}"
        start = time.monotonic()
        if prefix:
            passive = FogPlayer()
            passive.private_state_truth_provider = lambda _: truth[other].copy()
            actors = {role: learner, other: passive}
            reset = reset_from_record(client, record, role=role, remaining=schedule.horizon,
                                      actors=actors, tag=tag, future_seed=future_seed)
            if local_buffer.n_decisions or local_buffer.n_games:
                raise ValueError("Demonstration actions leaked into a PPO batch")
            initial_battle = reset.local
            options = {}
        else:
            initial_battle = None
            options = {"formatid": "gen9ou", "strictChoices": True, "seed": future_seed,
                       **{side: {"name": "Learner" if side == role else "Opponent", "team": to_packed(paste)}
                          for side, paste in submitted.items()}}
        prepared = time.monotonic()
        controllers = {role: learner, other: opponent}
        with torch.inference_mode():
            result = play_local_battle(client, options, controllers["p1"], controllers["p2"], seed=seed,
                                       battle_tag=tag, initial_battle=initial_battle, max_choice_rounds=2000)
        if result.truncated or learner.diagnostic_errors:
            raise ValueError("Online rollout was truncated or produced diagnostic failures")
        won = result.p1_won if role == "p1" or result.p1_won is None else not result.p1_won
        score = 0.5 if won is None else float(won)
        if local_buffer.n_games != 1 or not local_buffer.n_decisions:
            raise ValueError("Online rollout lacks a completed game with decisions")
        episodes.append({"episode": episode, "prefix": prefix, "role": role, "score": score,
                         "source_id": bank.train[state["pair_record"]]["id"] if prefix else None,
                         "horizon": schedule.horizon if prefix else None,
                         "archived_outcome": record["perspectives"][role]["outcome"] if prefix else None,
                         "decisions": local_buffer.n_decisions, "prepare_seconds": prepared-start,
                         "play_seconds": time.monotonic()-prepared})
        buffer.extend(local_buffer)
        state["episodes"] += 1
        if prefix:
            schedule.record(archived_outcome=record["perspectives"][role]["outcome"], score=score)
            state["reset_episodes"] += 1
        else:
            state["full_episodes"] += 1
    batch = buffer.to_batch("cpu")
    selected = torch.as_tensor(sampler.choice(buffer.n_decisions, recipe["positions_per_batch"], replace=False))
    from fog.training.ppo import _index_nested

    batch.pop("preview", None)
    batch = _index_nested(batch, selected)
    # Recompute the frozen anchor on the accepted, sampled public observations only.
    visible_keys = {"entities", "entity_ids", "field", "history", "history_ids", "history_state", "history_len",
                    "active_idx", "legality_mask", "action_features", "opp_revealed"}
    with torch.inference_mode():
        fragments = []
        for offset in range(0, len(selected), recipe["minibatch_size"]):
            visible = {key: value for key, value in batch.items() if key in visible_keys}
            indices = torch.arange(offset, min(offset + recipe["minibatch_size"], len(selected)))
            fragments.append(anchor(_index_nested(visible, indices)).log_probs)
        batch["bc_log_probs"] = torch.cat(fragments)
    if not batch.get("belief_targets") or not any(bool(value.any()) for value in batch["belief_valid"].values()):
        raise ValueError("Online batch lost exact-team belief labels")
    return batch, episodes, buffer.n_decisions


def train(config, recipe, bank, initial, out, *, device="cpu", resume=None, stop_after_batches=None,
          parent=None, anchor_checkpoint=None):
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.models.agent import FogAgent
    from fog.training.ppo import ppo_update

    for key in ("max_batches", "positions_per_batch", "minibatch_size", "epochs", "warmup_batches", "save_every"):
        if type(recipe.get(key)) is not int or recipe[key] < 1:
            raise ValueError("Freeze positive training dimensions: " + key)
    if recipe["positions_per_batch"] % recipe["minibatch_size"] or recipe["condition"] not in ("ordinary", "backward"):
        raise ValueError("Invalid matched pilot dimensions or condition")
    if config["model"].get("oracle") != "none" or not config["model"].get("aux_into_trunk") or not config["model"].get("public_action_features_only"):
        raise ValueError("Require public-information acting and shared auxiliary training")
    if not 0 < recipe["auxiliary_gradient_cap"] <= 0.25 or not 0 < recipe["learning_rate"] <= 1e-4:
        raise ValueError("Freeze bounded learning rate and shared auxiliary gradients")
    if recipe.get("actor_device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("Freeze a supported actor device")
    if bool(recipe.get("parent_state_sha256")) != (parent is not None):
        raise ValueError("The fixed parent full-state input is missing or undeclared")
    if parent is not None and anchor_checkpoint is None:
        raise ValueError("Parent continuation requires an explicit frozen policy anchor")
    torch.set_num_threads(1)
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"] % 2**32)
    sampler = np.random.default_rng(recipe["seed"])
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen GPU pilot has no CUDA accelerator")
    if recipe.get("actor_device") == "cuda" and device.type != "cuda":
        raise ValueError("A CUDA actor requires a CUDA training device")
    agent = FogAgent.from_tables(config).to(device).eval()
    agent.load_state_dict(torch.load(initial, weights_only=True, map_location=device))
    actor_device = device if recipe.get("actor_device") == "cuda" else torch.device("cpu")
    actor, anchor = FogAgent.from_tables(config).to(actor_device).eval(), FogAgent.from_tables(config).cpu().eval()
    anchor.load_state_dict(torch.load(anchor_checkpoint or initial, weights_only=True, map_location="cpu"))
    optimizer = torch.optim.AdamW(agent.parameters(), lr=recipe["learning_rate"], weight_decay=1e-4)
    identity_fields = {"config": config, "recipe": recipe, "bank": bank.sha256, "pool": bank.pool_sha256,
                       "initial": digest(initial)}
    if anchor_checkpoint is not None:
        identity_fields["anchor"] = digest(anchor_checkpoint)
    identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True).encode()).hexdigest()
    schedule, batch_number, history, trained_seconds = ReverseSchedule(), 0, [], 0.0
    lineage = {"inherited_batches": 0, "inherited_optimizer_steps": 0, "inherited_optimized_positions": 0}
    if parent is not None:
        if digest(parent) != recipe["parent_state_sha256"]:
            raise ValueError("Parent full-state hash differs from the fixed continuation")
        previous = torch.load(parent, weights_only=False, map_location="cpu")
        if (previous.get("format") != "fog.curriculum-pilot-state" or previous.get("version") != 1
                or previous["identity"] != recipe.get("parent_identity")
                or previous["batch"] != recipe.get("parent_batches")
                or sum(row["optimizer_steps"] for row in previous["history"]) != recipe.get("parent_optimizer_steps")):
            raise ValueError("Parent training identity or exposure differs")
        weights = torch.load(initial, weights_only=True, map_location="cpu")
        if (weights.keys() != previous["agent"].keys()
                or not all(torch.equal(value, previous["agent"][key]) for key, value in weights.items())):
            raise ValueError("Starting weights differ from the parent full state")
        optimizer.load_state_dict(previous["optimizer"])
        schedule = ReverseSchedule(previous["schedule"])
        older = previous.get("lineage", {})
        lineage = {"inherited_batches": older.get("inherited_batches", 0) + previous["batch"],
                   "inherited_optimizer_steps": older.get("inherited_optimizer_steps", 0) + recipe["parent_optimizer_steps"],
                   "inherited_optimized_positions": older.get("inherited_optimized_positions", 0) + previous["history"][-1]["optimized_positions"],
                   "parent_identity": previous["identity"], "parent_state_sha256": digest(parent)}
        restore_rng(previous["random"], sampler)
    if resume is not None:
        saved = torch.load(resume, weights_only=False, map_location=device)
        if saved.get("format") != "fog.curriculum-pilot-state" or saved.get("version") != 1 or saved["identity"] != identity:
            raise ValueError("Resume identity, version or input hashes differ")
        agent.load_state_dict(saved["agent"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule, batch_number, history = ReverseSchedule(saved["schedule"]), saved["batch"], saved["history"]
        if saved.get("lineage", lineage) != lineage:
            raise ValueError("Resume continuation lineage differs")
        trained_seconds = saved["training_seconds"]
        restore_rng(saved["random"], sampler)
    out.mkdir(parents=True, exist_ok=resume is not None)
    contract = probability_contract(agent, bank.contract_batch())
    begun = time.monotonic()
    cfg = {"ppo": {"epochs": recipe["epochs"], "minibatch_size": recipe["minibatch_size"],
                   "rollout_temperature": 1.0, "max_grad_norm": 1.5, "value_target": "outcome",
                   "auxiliary_gradient_cap": recipe["auxiliary_gradient_cap"], "kl_anchor_coef": 0.02},
           "loss": {"value_coef": 0.25, "belief_coef": 0.01, "belief_chain_coef": 0.1,
                    "self_belief_coef": 0.0, "self_belief_chain_coef": 0.0, "preview_coef": 0.0}}

    def save():
        value = {"format": "fog.curriculum-pilot-state", "version": 1, "identity": identity,
                 "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                 "random": rng_state(sampler), "schedule": deepcopy(schedule.state), "batch": batch_number,
                 "history": history, "training_seconds": trained_seconds + time.monotonic() - begun}
        value["lineage"] = lineage
        if (out / "latest.state.pt").exists():
            (out / "latest.state.pt").replace(out / "previous.state.pt")
        atomic_save(value, out / "latest.state.pt")
        atomic_save(agent.state_dict(), out / "latest.pt")

    reason = "max_batches"
    with ShowdownDirectClient() as client, (out / "metrics.jsonl").open("a") as log:
        save()
        while batch_number < recipe["max_batches"]:
            if stop_after_batches is not None and batch_number >= stop_after_batches:
                reason = "local_trial_stop"
                break
            if trained_seconds + time.monotonic() - begun >= recipe["training_seconds"]:
                reason = "training_time_budget"
                break
            actor.load_state_dict(agent.state_dict())
            tick = time.monotonic()
            cpu_batch, episodes, collected = collect(client, actor, anchor, bank, sampler, schedule, recipe)
            rollout_seconds = time.monotonic() - tick
            progress = max(0.0, (batch_number - recipe["warmup_batches"]) / max(1, recipe["max_batches"] - recipe["warmup_batches"]))
            rate = min(1.0, (batch_number + 1) / recipe["warmup_batches"]) * (0.1 + 0.9*(1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:
                group["lr"] = rate * recipe["learning_rate"]

            def moved(value):
                return {key: moved(item) for key, item in value.items()} if isinstance(value, dict) else value.to(device) if isinstance(value, torch.Tensor) else value

            tick = time.monotonic()
            metrics = ppo_update(agent, optimizer, moved(cpu_batch), cfg)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError("Nonfinite online training diagnostics")
            expected_steps = recipe["epochs"] * recipe["positions_per_batch"] // recipe["minibatch_size"]
            if metrics["optimizer_steps"] != expected_steps or metrics["belief_dropped_rate"] or metrics["capped_auxiliary_primary_ratio"] > 0.25001:
                raise ValueError("Online update lost matched optimization or auxiliary constraints")
            batch_number += 1
            advanced = schedule.advance(batch_number + lineage["inherited_batches"]) if recipe["condition"] == "backward" else False
            row = {"batch": batch_number, "optimized_positions": batch_number * recipe["positions_per_batch"],
                   "collected_positions": collected, "learning_rate": optimizer.param_groups[0]["lr"],
                   "rollout_seconds": rollout_seconds, "update_seconds": time.monotonic()-tick,
                   "episodes": episodes, "horizon_after_update": schedule.horizon, "advanced": advanced, **metrics}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps({key: value for key, value in row.items() if key != "episodes"}), flush=True)
            if batch_number % recipe["save_every"] == 0:
                save()
            if batch_number % 10 == 0:
                atomic_save(agent.state_dict(), out / f"anchor-{batch_number:06d}.pt")
        save()
    final_contract = probability_contract(agent, bank.contract_batch())
    result = {"complete": batch_number == recipe["max_batches"], "stop_reason": reason, "batches": batch_number,
              "optimized_positions": batch_number * recipe["positions_per_batch"],
              "optimizer_steps": sum(row["optimizer_steps"] for row in history), "condition": recipe["condition"],
              "identity": identity, "starting_contract": contract, "final_contract": final_contract,
              "checkpoint_sha256": digest(out / "latest.pt"), "full_state_sha256": digest(out / "latest.state.pt"),
              "training_seconds": trained_seconds + time.monotonic() - begun,
              "schedule": schedule.state, "automatic_promotion": False, "device": str(device)}
    result.update(lineage=lineage, actor_device=str(actor_device), anchor_sha256=digest(anchor_checkpoint or initial))
    result.update(total_optimizer_steps=result["optimizer_steps"] + lineage["inherited_optimizer_steps"],
                  total_optimized_positions=result["optimized_positions"] + lineage["inherited_optimized_positions"])
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "recipe", "bank", "pool", "initial", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--stop-after-batches", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = train(load_config(args.config), json.loads(args.recipe.read_text()), VerifiedBank(args.bank, args.pool),
                   args.initial, args.out, device=args.device, resume=args.resume, stop_after_batches=args.stop_after_batches,
                   parent=args.parent, anchor_checkpoint=args.anchor_checkpoint)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
