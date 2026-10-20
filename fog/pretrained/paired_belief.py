"""Audit and join independently executed, identically exposed belief conditions."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

from fog.scripts.kakuna_latent_resume_check import identical


def merge_conditions(config, out):
    """Keep genuine per-arm Adam/RNG states; never invent a combined optimizer."""
    out = Path(out)
    if config.get("execution") != "parallel_conditions":
        raise ValueError("An explicit parallel-condition recipe is required")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    reports, states, heads = {}, {}, {}
    for mode in ("facts", "native"):
        folder = out / mode
        report = json.loads((folder / "RESULTS.json").read_text())
        state = torch.load(folder / f"{mode}_full.state.pt", map_location="cpu", weights_only=False)
        head = torch.load(folder / f"{mode}_heads_final.pt", map_location="cpu", weights_only=True)
        if (
            (folder / "recipe.json").read_text() != canonical
            or not report["complete"]
            or report["condition"] != mode
            or report["steps"] != config["updates"]
            or report["native_policy_changed"]
            or report["native_optimizer_present"]
            or state["version"] != 2
            or state["condition"] != mode
            or state["phase"] != 0
            or state["steps"] != config["updates"]
            or state["completed"] != config["updates"]
            or state["config"] != config
            or state["identity"] != report["identity"]
            or len(state["histories"]) != config["updates"]
            or not state["optimizer"]["state"]
            or not identical(state["model"], head["heads"])
            or any(key.startswith("native.") for key in state["model"])
        ):
            raise ValueError("Independent condition failed its full-state/exposure audit: " + mode)
        if any(row["mode"] != mode for row in state["histories"]):
            raise ValueError("A condition mixed optimizer histories")
        if any(
            not 0 < float(value["step"]) <= config["updates"]
            for value in state["optimizer"]["state"].values()
        ):
            raise ValueError("A condition has inconsistent Adam exposure")
        if any(not torch.isfinite(value).all() for value in state["model"].values()):
            raise ValueError("Nonfinite condition weights")
        reports[mode], states[mode], heads[mode] = report, state, head
    if reports["facts"]["identity"] != reports["native"]["identity"] or not identical(
        states["facts"]["initial"], states["native"]["initial"]
    ):
        raise ValueError("Independent conditions used different initializations or recipes")
    for left, right in zip(states["facts"]["histories"], states["native"]["histories"]):
        for key in (
            "update", "query_sha256", "route_before_sha256", "route_after_sha256",
            "AR_queries", "AR_query_draws", "lr",
        ):
            if left[key] != right[key]:
                raise ValueError("Independent conditions consumed different matched queries: " + key)
    if not all(
        torch.equal(value, heads["native"]["heads"][key])
        for key, value in heads["facts"]["heads"].items() if key.startswith("outcome.")
    ):
        raise ValueError("Outcome heads differ across independent belief conditions")
    priors = [
        torch.load(out / mode / "prior_only.pt", map_location="cpu", weights_only=True)
        for mode in ("facts", "native")
    ]
    if not identical(priors[0], priors[1]):
        raise ValueError("Independent conditions did not start at the same prior-only model")
    for mode in ("facts", "native"):
        for filename in (f"{mode}_heads_final.pt", f"{mode}_full.state.pt"):
            shutil.copy2(out / mode / filename, out / filename)
    shutil.copy2(out / "facts/prior_only.pt", out / "prior_only.pt")
    (out / "recipe.json").write_text(canonical)
    result = {
        "complete": True,
        "condition": None,
        "execution": "parallel_conditions",
        "identity": reports["facts"]["identity"],
        "steps": config["updates"] * 2,
        "arm_updates": {mode: reports[mode]["steps"] for mode in reports},
        "updates_per_condition": config["updates"],
        "AR_queries_per_context": config["chain_queries"],
        "matched_query_routes": True,
        "outcome_heads_identical": True,
        "native_policy_changed": False,
        "native_optimizer_present": False,
        "native_sha256": config["native_sha256"],
        "prior_sha256": config["prior_sha256"],
        "evaluations": {
            "prior_only": reports["facts"]["evaluations"]["prior_only"],
            **{mode: reports[mode]["evaluations"][mode] for mode in reports},
        },
        "full_state_files": {mode: f"{mode}_full.state.pt" for mode in reports},
        "elapsed_seconds": max(report["elapsed_seconds"] for report in reports.values()),
        "elapsed_scope": "Maximum arm wall-time; worker reports whole-job time separately.",
        "automatic_promotion": False,
        "search_connected": False,
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
