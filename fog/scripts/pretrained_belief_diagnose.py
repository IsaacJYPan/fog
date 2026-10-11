"""Compare public marginals and true-prefix conditionals across saved checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))

import torch

from fog.constants.tables import TABLES
from fog.encoding.reveals import masks_for_tokens
from fog.models.belief import BeliefModel
from fog.pretrained.belief_diagnostics import DistributionMetrics
from fog.pretrained.checkpoints import load_auxiliary
from fog.pretrained.joint import JointKakuna
from fog.pretrained.kakuna import load_kakuna, sha256
from fog.pretrained.loading import read_record, SequenceLoader


def validate_diagnostic(config):
    if type(config.get("seed")) is not int or type(config.get("queries_per_context")) is not int or not 1 <= config["queries_per_context"] <= 256:
        raise ValueError("Freeze a seed and 1-256 true-prefix queries per context")
    if set(config.get("cohorts", {})) != {"train", "development"}:
        raise ValueError("Require separate training and development diagnostic cohorts")
    for row in config["cohorts"].values():
        if not row.get("folder") or type(row.get("games")) is not int or not 1 <= row["games"] <= 64:
            raise ValueError("Freeze bounded complete diagnostic cohorts")
    checkpoints = config.get("checkpoints", [])
    if not checkpoints or any(
        not re.fullmatch(r"[a-z0-9-]+", row.get("label", ""))
        or not row.get("native") or not row.get("auxiliary")
        or not re.fullmatch(r"[a-f0-9]{64}", row.get("native_sha256", ""))
        or type(row.get("update")) is not int or row["update"] < 1
        for row in checkpoints
    ):
        raise ValueError("Require hash-bound diagnostic checkpoint anchors")
    if len({row["label"] for row in checkpoints}) != len(checkpoints) or [row["update"] for row in checkpoints] != sorted({row["update"] for row in checkpoints}):
        raise ValueError("Checkpoint labels and increasing update anchors must be unique")


def score_checkpoint(config, checkpoint, pools, out):
    native, auxiliary = Path(checkpoint["native"]), Path(checkpoint["auxiliary"])
    if sha256(native) != checkpoint["native_sha256"]:
        raise ValueError("Diagnostic native checkpoint hash changed")
    auxiliary_sha = sha256(auxiliary)
    runtime = load_kakuna(native, verify_original=False)
    model = JointKakuna(runtime.model).eval()
    state = load_auxiliary(model, auxiliary, native)
    if state["update"] != checkpoint["update"]:
        raise ValueError("Diagnostic checkpoint update does not match its auxiliary head")
    loader = SequenceLoader(runtime, sequence_length=1000)
    split_reports = {}
    for split, cohort in config["cohorts"].items():
        folder = Path(cohort["folder"])
        collection = json.loads((folder / "collection.json").read_text())
        if collection["split"] != split or collection["pool_sha256"] != sha256(pools / (split + ".json")):
            raise ValueError("Diagnostic cohort pool/split mismatch")
        paths = sorted(folder.glob("*.json.gz"))
        if len(paths) != cohort["games"]:
            raise ValueError("Diagnostic cohort is incomplete")
        metrics = {name: DistributionMetrics() for name in ("public_marginal", "marginal_on_chain_queries", "true_prefix_chain")}
        generator = torch.Generator().manual_seed(config["seed"])
        contexts, outcome_errors, by_game = 0, [], {}
        query_schedule = hashlib.sha256()
        with torch.inference_mode():
            for path in paths:
                record = read_record(path)
                if not record["training_eligible"] or record.get("failures") or record.get("fallbacks"):
                    raise ValueError("Failed diagnostic game requires investigation")
                if any(len(pov["actions"]) - 1 > loader.sequence_length for pov in record["perspectives"].values()):
                    raise ValueError("A diagnostic cannot silently crop battle history")
                game_metrics = {name: DistributionMetrics() for name in metrics}
                game_errors, game_contexts = [], 0
                for pov_seed in (0, 1):
                    example, targets = loader.load(("fresh", str(path), pov_seed))
                    points = {0, len(example.actions) // 2, len(example.actions) - 1}
                    hidden = runtime.model.traj_encoder.init_hidden_state(1, torch.device("cpu"))
                    for index in range(len(example.actions)):
                        latent, hidden = runtime.model.get_state_embedding(
                            obs={key: value[None, index:index + 1] for key, value in example.obs.items()},
                            rl2s=example.rl2s[None, index:index + 1],
                            time_idxs=example.time_idxs[None, index:index + 1], hidden_state=hidden,
                        )
                        if index not in points:
                            continue
                        public = targets["reveals"][index:index + 1]
                        verified = targets["verified"][index:index + 1]
                        safe_truth = torch.where(verified, targets["truth"][index:index + 1], -1)
                        masks = {key: torch.from_numpy(value[None]) for key, value in masks_for_tokens(public[0].numpy(), TABLES).items()}
                        payload = {"tokens": public, "masks": masks}
                        context = model.context_projection(latent[:, 0])
                        marginal = model.belief(context, payload)
                        hidden_fields = verified & ~model.belief._revealed_mask(public).reshape_as(verified)
                        game_metrics["public_marginal"].update(marginal.logits, BeliefModel._split_flat(safe_truth), BeliefModel._split_flat(hidden_fields))
                        q = config["queries_per_context"]
                        chain = model.belief(
                            context.expand(q, -1),
                            {"tokens": public.expand(q, -1, -1), "masks": {key: value.expand(q, *value.shape[1:]) for key, value in masks.items()}},
                            teacher_values=safe_truth.expand(q, -1, -1), generator=generator,
                        )
                        selected = {key: value & (chain.loss_targets[key] >= 0) for key, value in chain.loss_valid.items()}
                        game_metrics["true_prefix_chain"].update(chain.logits, chain.loss_targets, selected)
                        game_metrics["marginal_on_chain_queries"].update(
                            {key: value.expand(q, *value.shape[1:]) for key, value in marginal.logits.items()}, chain.loss_targets, selected
                        )
                        query_schedule.update(json.dumps([path.name, pov_seed, index, chain.order.tolist(), chain.context_count.tolist()], separators=(",", ":")).encode())
                        prediction = float(model.outcome(latent[:, 0]).tanh().item())
                        if not math.isfinite(prediction):
                            raise FloatingPointError("Nonfinite outcome diagnostic prediction")
                        game_errors.append((prediction - float(targets["outcomes"][index])) ** 2)
                        contexts += 1
                        game_contexts += 1
                for name, metric in game_metrics.items():
                    metrics[name].merge(metric)
                outcome_errors.extend(game_errors)
                by_game[path.name] = {
                    "contexts": game_contexts,
                    "metrics": {name: metric.report() for name, metric in game_metrics.items()},
                    "outcome_mse": sum(game_errors) / len(game_errors),
                }
                print(json.dumps({"checkpoint": checkpoint["label"], "split": split, "game": path.name, "contexts": contexts}), flush=True)
        split_reports[split] = {
            "games": len(paths), "contexts": contexts,
            "input_games": {path.name: sha256(path) for path in paths},
            "pool_sha256": collection["pool_sha256"],
            "metrics": {name: metric.report() for name, metric in metrics.items()},
            "by_game": by_game,
            "query_schedule_sha256": query_schedule.hexdigest(),
            "outcome_mse": sum(outcome_errors) / len(outcome_errors),
        }
    if sha256(native) != checkpoint["native_sha256"] or sha256(auxiliary) != auxiliary_sha:
        raise ValueError("Diagnostic modified checkpoint weights")
    report = {
        **checkpoint, "auxiliary_sha256": auxiliary_sha, "splits": split_reports,
        "weights_unchanged": True, "training_performed": False,
        "interpretation": "Marginal and true-prefix chain query scores use identical sampled next-field targets. Public marginal scores include all verified hidden fields. Distribution averages weight eligible fields; by-game summaries allow game-level resampling. Move top1/NLL refer to an exchangeable next move label, not exact moveset accuracy. Queries and contexts within a game are correlated, and games may share teams. Teacher prefixes use exact labels offline and never enter policy play. Outcome MSE is against observed behavior outcomes, not a native-critic comparison.",
    }
    (out / (checkpoint["label"] + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    del model, runtime
    gc.collect()
    return report


def diagnose(config, pools, out):
    validate_diagnostic(config)
    if out.exists():
        raise FileExistsError("Choose a new diagnostic identity")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    reports = [score_checkpoint(config, checkpoint, pools, out) for checkpoint in config["checkpoints"]]
    for split in config["cohorts"]:
        for report in reports[1:]:
            for key in ("input_games", "query_schedule_sha256", "contexts", "pool_sha256"):
                if report["splits"][split][key] != reports[0]["splits"][split][key]:
                    raise ValueError("Checkpoint diagnostic input/query schedules differ")
    ancestral = None
    if config.get("ancestral_report"):
        path = Path(config["ancestral_report"])
        ancestral = json.loads(path.read_text())
        if ancestral["native_sha256"] != reports[-1]["native_sha256"] or ancestral["input_games"] != reports[-1]["splits"]["development"]["input_games"]:
            raise ValueError("Ancestral benchmark uses different weights or development games")
        if not ancestral["sampling_trace_finite"] or ancestral["search_connected"]:
            raise ValueError("Require finite offline ancestral benchmark")
    result = {"complete": True, "config": config, "checkpoints": reports, "ancestral_reference": ancestral, "gpu_requested": False, "training_performed": False, "automatic_promotion": False}
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(json.loads(args.config.read_text()), args.pools, args.out)


if __name__ == "__main__":
    main()
