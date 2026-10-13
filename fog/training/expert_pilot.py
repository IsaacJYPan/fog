"""Bounded strong-teacher training with shared supervision and complete resume state."""

from __future__ import annotations

import argparse
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
from fog.training.expert_data import ExpertData
from fog.training.teacher_replay import digest


def recipe_digest(recipe, config):
    encoded = json.dumps({"recipe": recipe, "config": config}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_recipe(recipe, config):
    model = config["model"]
    required = ("shared_latent_trunk", "shared_latent_learned_only", "shared_typed_world_tokens",
                "action_query_policy", "use_action_features", "public_action_features_only", "aux_into_trunk")
    if not all(model.get(key) is True for key in required) or model.get("oracle") != "none" or model.get("simple_rule_prior_strength", 0) != 0:
        raise ValueError("Require independent typed/action-query weights, shared supervision and public action features")
    if config.get("bc", {}).get("init_ckpt") or config.get("ppo", {}).get("init_ckpt"):
        raise ValueError("The fallback cannot silently load an inherited warm start")
    for key in ("max_updates", "games_per_batch", "rows_per_game", "microbatch", "warmup_updates", "save_every", "evaluate_every"):
        if type(recipe.get(key)) is not int or recipe[key] < 1:
            raise ValueError("Freeze positive integer training dimensions: " + key)
    if recipe["games_per_batch"] * recipe["rows_per_game"] % recipe["microbatch"]:
        raise ValueError("Effective batch must be a multiple of microbatch")
    for key in ("learning_rate", "gradient_clip", "auxiliary_gradient_cap", "training_seconds"):
        if not math.isfinite(recipe.get(key, 0)) or recipe[key] <= 0:
            raise ValueError("Freeze a finite positive training setting: " + key)
    if recipe["auxiliary_gradient_cap"] > 0.25:
        raise ValueError("Shared auxiliary gradient cap exceeds its approved fraction")
    if any(config["loss"].get(key, 0) <= 0 for key in ("action_coef", "belief_coef", "belief_chain_coef", "value_coef")):
        raise ValueError("Policy, marginal, chain and outcome supervision must all be enabled")


def rng_state(sampler):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": deepcopy(sampler.bit_generator.state)}


def restore_rng(state, sampler):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA training state requires CUDA for exact resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    sampler.bit_generator.state = deepcopy(state["sampler"])


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def shared_parameters(agent):
    prefixes = ("backbone.", "shared_latent.", "structured_encoder.")
    return [parameter for name, parameter in agent.named_parameters() if name.startswith(prefixes) and parameter.requires_grad]


def gradient_norm(values):
    present = [value for value in values if value is not None]
    if not present:
        return 0.0
    return float(torch.sqrt(sum(value.float().square().sum() for value in present)))


def capped_update(agent, optimizer, data, specifications, config, recipe, device):
    """Cap the summed auxiliary gradient on shared parameters; retain full head gradients."""
    from fog.training.bc import bc_loss

    optimizer.zero_grad(set_to_none=True)
    shared = shared_parameters(agent)
    primary_gradients = [None] * len(shared)
    auxiliary_gradients = [None] * len(shared)
    metrics = {}
    microbatch = recipe["microbatch"]
    accumulation = len(specifications) // microbatch
    for start in range(0, len(specifications), microbatch):
        batch = data.batch(specifications[start:start + microbatch], device=device)
        total, observed, components = bc_loss(agent, batch, config["loss"], return_components=True)
        if observed["policy_valid_count"] != microbatch or observed["belief_dropped_count"] or observed["belief_chain_dropped_count"]:
            raise ValueError("Exact teacher batch lost valid policy or belief supervision")
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite expert training loss")
        primary = components["action"] / accumulation
        auxiliary = sum(value for name, value in components.items() if name != "action") / accumulation
        for destination, objective in ((primary_gradients, primary), (auxiliary_gradients, auxiliary)):
            gradients = torch.autograd.grad(objective, shared, retain_graph=True, allow_unused=True)
            for index, value in enumerate(gradients):
                if value is not None:
                    destination[index] = value.detach().clone() if destination[index] is None else destination[index].add_(value.detach())
        (total / accumulation).backward()
        for key, value in observed.items():
            metrics[key] = metrics.get(key, 0.0) + (value if key.endswith("_count") else value / accumulation)
    primary_norm, auxiliary_norm = gradient_norm(primary_gradients), gradient_norm(auxiliary_gradients)
    if not math.isfinite(primary_norm + auxiliary_norm):
        raise FloatingPointError("Nonfinite shared gradient")
    scale = min(1.0, recipe["auxiliary_gradient_cap"] * primary_norm / max(auxiliary_norm, 1e-20))
    for parameter, auxiliary in zip(shared, auxiliary_gradients):
        if auxiliary is not None:
            parameter.grad.add_(auxiliary, alpha=scale - 1.0)
    norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), recipe["gradient_clip"]))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite expert update gradient")
    optimizer.step()
    metrics.update(shared_primary_norm=primary_norm, shared_auxiliary_norm=auxiliary_norm,
                   shared_auxiliary_scale=scale, raw_auxiliary_primary_ratio=auxiliary_norm / max(primary_norm, 1e-20),
                   capped_auxiliary_primary_ratio=scale * auxiliary_norm / max(primary_norm, 1e-20), gradient_norm=norm)
    return metrics


@torch.no_grad()
def evaluate(agent, data, config, microbatch, device):
    from fog.training.bc import bc_loss

    specifications = data.development()
    metrics = {}
    was_training = agent.training
    agent.eval()
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261013)
            for start in range(0, len(specifications), microbatch):
                selected = specifications[start:start + microbatch]
                batch = data.batch(selected, device=device)
                batch["compute_belief_chain"] = True
                _, values = bc_loss(agent, batch, config["loss"])
                if values["policy_valid_count"] != len(selected) or values["belief_dropped_count"] or values["belief_chain_dropped_count"]:
                    raise ValueError("Development lost exact supervision")
                for key, value in values.items():
                    if not math.isfinite(value):
                        raise FloatingPointError("Nonfinite development metric")
                    metrics[key] = metrics.get(key, 0.0) + (value if key.endswith("_count") else value * len(selected) / len(specifications))
    finally:
        agent.train(was_training)
    return {"contexts": len(specifications), **metrics}


@torch.no_grad()
def probability_contract(agent, batch, *, tolerance=1e-3):
    """Compare ordinary CPU/CUDA policy probabilities on fixed player-visible inputs."""
    def moved(value, destination):
        if isinstance(value, dict):
            return {key: moved(item, destination) for key, item in value.items()}
        return value.to(destination) if isinstance(value, torch.Tensor) else value

    original = next(agent.parameters()).device
    was_training = agent.training
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    agent.eval()
    player_keys = {"entities", "entity_ids", "field", "history", "history_ids", "history_state", "history_len",
                   "active_idx", "legality_mask", "action_features", "opp_revealed"}
    visible = {key: value for key, value in batch.items() if key in player_keys}
    try:
        cpu = agent.to("cpu")(moved(visible, "cpu")).log_probs.exp()
        if not torch.isfinite(cpu).all() or not torch.allclose(cpu.sum(-1), torch.ones(len(cpu)), atol=1e-5):
            raise ValueError("Ordinary policy returned invalid CPU probabilities")
        illegal = ~visible["legality_mask"].cpu()
        if bool(cpu[illegal].any()):
            raise ValueError("Ordinary policy assigns mass to illegal actions")
        error = None
        if original.type == "cuda":
            gpu = agent.to(original)(moved(visible, original)).log_probs.exp().cpu()
            error = float((cpu - gpu).abs().max())
            if not math.isfinite(error) or error > tolerance:
                raise ValueError("CPU/CUDA policy probability contract failed")
        return {"passed": True, "contexts": len(cpu), "cpu_cuda_max_error": error,
                "cpu_cuda_verified": original.type == "cuda", "tolerance": tolerance,
                "player_visible_inputs_only": True}
    finally:
        agent.to(original).train(was_training)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)


def train(recipe, config, data, initial, out, *, device="cpu", resume=None, stop_after_updates=None):
    from fog.models.agent import FogAgent

    validate_recipe(recipe, config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The frozen GPU pilot has no CUDA device")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    agent = FogAgent.from_tables(config).to(device)
    agent.load_state_dict(torch.load(initial, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(agent.parameters(), lr=recipe["learning_rate"], weight_decay=1e-4)
    sampler = np.random.default_rng(seed)
    identity = recipe_digest(recipe, config)
    update, history, baseline, trained_seconds = 0, [], None, 0.0
    out.mkdir(parents=True, exist_ok=resume is not None)
    if resume is not None:
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved.get("format") != "fog.expert-pilot-state" or saved.get("version") != 1 or saved["recipe_sha256"] != identity:
            raise ValueError("Resume state recipe/version differs")
        if saved["data_sha256"] != data.source_sha256 or saved["initial_sha256"] != digest(initial):
            raise ValueError("Resume state data or initialization differs")
        agent.load_state_dict(saved["agent"])
        optimizer.load_state_dict(saved["optimizer"])
        update, history, baseline = saved["update"], saved["history"], saved["baseline"]
        trained_seconds = saved["training_seconds"]
        restore_rng(saved["random"], sampler)
    fixed = data.batch(data.development()[:8], device="cpu")
    contract = probability_contract(agent, fixed)
    if baseline is None:
        baseline = evaluate(agent, data, config, recipe["microbatch"], device)
        history.append({"update": 0, "development": baseline})
    agent.train()
    begun = time.monotonic()
    reason = "max_updates"

    def save():
        saved = {"format": "fog.expert-pilot-state", "version": 1, "recipe_sha256": identity,
                 "data_sha256": data.source_sha256, "initial_sha256": digest(initial),
                 "agent": agent.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                 "random": rng_state(sampler), "history": history, "baseline": baseline,
                 "training_seconds": trained_seconds + time.monotonic() - begun}
        current = out / "latest.state.pt"
        if current.exists():
            current.replace(out / "previous.state.pt")
        atomic_save(saved, current)
        atomic_save(agent.state_dict(), out / "latest.pt")

    with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while update < recipe["max_updates"]:
            if trained_seconds + time.monotonic() - begun >= recipe["training_seconds"]:
                reason = "training_time_budget"
                break
            if stop_after_updates is not None and update >= stop_after_updates:
                reason = "local_trial_stop"
                break
            progress = max(0.0, (update - recipe["warmup_updates"]) / max(1, recipe["max_updates"] - recipe["warmup_updates"]))
            rate = min(1.0, (update + 1) / recipe["warmup_updates"]) * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = recipe["learning_rate"] * rate
            specifications = data.sample(sampler, games=recipe["games_per_batch"], rows_per_game=recipe["rows_per_game"])
            started = time.monotonic()
            metrics = capped_update(agent, optimizer, data, specifications, config, recipe, device)
            update += 1
            record = {"update": update, "sampled_positions": update * len(specifications),
                      "sample_schedule_sha256": hashlib.sha256(json.dumps([(row["id"], i) for row, i in specifications]).encode()).hexdigest(),
                      "learning_rate": optimizer.param_groups[0]["lr"], "update_seconds": time.monotonic() - started, **metrics}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if update <= 3 or update % 50 == 0:
                print(json.dumps(record), flush=True)
            if update % recipe["evaluate_every"] == 0:
                observed = evaluate(agent, data, config, recipe["microbatch"], device)
                history.append({"update": update, "development": observed})
                print(json.dumps(history[-1]), flush=True)
            if update % recipe["save_every"] == 0:
                save()
            if update % 1000 == 0:
                atomic_save(agent.state_dict(), out / f"anchor-{update:06d}.pt")
        if history[-1]["update"] != update:
            history.append({"update": update, "development": evaluate(agent, data, config, recipe["microbatch"], device)})
        save()
    final_contract = probability_contract(agent, fixed)
    result = {"complete": True, "stop_reason": reason, "updates": update,
              "recipe_sha256": identity, "data_sha256": data.source_sha256,
              "initial_sha256": digest(initial), "checkpoint_sha256": digest(out / "latest.pt"),
              "full_state_sha256": digest(out / "latest.state.pt"), "device": str(device),
              "torch": torch.__version__, "history": history, "baseline": baseline,
              "starting_contract": contract, "final_contract": final_contract,
              "training_seconds": trained_seconds + time.monotonic() - begun,
              "sampled_positions": update * recipe["games_per_batch"] * recipe["rows_per_game"],
              "sampled_positions_are_unique": False, "automatic_promotion": False}
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    result = train(json.loads(args.recipe.read_text()), load_config(args.config), ExpertData(args.manifest),
                   args.initial, args.out, device=args.device, resume=args.resume, stop_after_updates=args.stop_after_updates)
    print(json.dumps({"complete": result["complete"], "updates": result["updates"], "stop_reason": result["stop_reason"]}), flush=True)


if __name__ == "__main__":
    main()
