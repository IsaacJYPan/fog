"""Separate CPU reconstruction comparison for the sealed frozen-policy head pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fog.models.donor_prior import DonorPrior
from fog.pretrained.frozen_belief import FrozenBeliefData, digest, load_heads
from fog.pretrained.joint import BeliefHeads
from fog.pretrained.prior_head_metrics import evaluation_queries, score_distributions, score_samples
from fog.scripts.pretrained_prior_audit import compare


def contexts(data, specifications):
    rows = {r["path"]: r for values in data.rows.values() for r in values}
    output = []
    for name, index in specifications:
        labels = data.record(name)["labels"]
        record = rows[name]
        output.append(
            {
                "game": record["game"],
                "reveals": labels["reveals"][index].tolist(),
                "truth": labels["truth"][index].tolist(),
                "verified": labels["verified"][index].tolist(),
                "team_rosters": record["roster_groups"],
                "source_sha256": record["source_sha256"],
            }
        )
    return output


def run(config, out):
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = Path(config["training"])
    result = json.loads((training / "RESULTS.json").read_text())
    recipe = json.loads((training / "recipe.json").read_text())
    if (
        not result["complete"]
        or not result["matched_query_routes"]
        or not result["outcome_heads_identical"]
        or result["native_policy_changed"]
        or result["native_optimizer_present"]
    ):
        raise ValueError("The fixed matched training result failed its audit")
    if digest(config["prior"]) != recipe["prior_sha256"]:
        raise ValueError("Evaluation prior differs from training")
    prior = DonorPrior(
        Path(config["prior"]), max_donors=recipe["max_donors"], smoothing=recipe["smoothing"]
    )
    reports = {}
    old_manifest = json.loads((Path(config["data"]) / "manifest.json").read_text())
    old_rosters = {group for row in old_manifest["records"] for group in row["roster_groups"]}
    for cohort, folder in [
        ("reused_development", config["data"]),
        ("fresh_development", config["fresh_data"]),
    ]:
        manifest_hash = digest(Path(folder) / "manifest.json")
        if cohort == "reused_development" and manifest_hash != recipe["data_sha256"]:
            raise ValueError("Reused evaluation data differ from the training freeze")
        data = FrozenBeliefData(
            folder,
            manifest_hash,
            required_splits=("development",),
            allow_trial=config.get("local_trial", False),
        )
        if data.manifest["native_sha256"] != recipe["native_sha256"]:
            raise ValueError("Evaluation native features belong to another policy")
        if cohort == "fresh_development" and not config.get("local_trial"):
            new_rosters = {
                group for row in data.manifest["records"] for group in row["roster_groups"]
            }
            if new_rosters & old_rosters or len(new_rosters) != config["fresh_groups"] * 2:
                raise ValueError(
                    "Fresh roster groups overlap old evidence or lack the declared coverage"
                )
        specifications = evaluation_queries(data)
        rows = contexts(data, specifications)
        methods = {}
        for label, filename, mode in [
            ("prior_only", "prior_only.pt", "native"),
            ("facts_residual", "facts_heads_final.pt", "facts"),
            ("native_residual", "native_heads_final.pt", "native"),
            ("previous_neural", None, "native"),
        ]:
            model = BeliefHeads().eval()
            if filename:
                state = load_heads(
                    model,
                    training / filename,
                    native_sha256=recipe["native_sha256"],
                    prior_sha256=recipe["prior_sha256"],
                )
                checkpoint_hash = digest(training / filename)
            else:
                if digest(config["initial_heads"]) != recipe["initial_heads_sha256"]:
                    raise ValueError("Preceding decoder changed")
                state = torch.load(config["initial_heads"], map_location="cpu", weights_only=True)
                if state["native_sha256"] != recipe["native_sha256"]:
                    raise ValueError("Preceding heads belong to another native policy")
                model.load_state_dict(state["heads"], strict=True)
                checkpoint_hash = digest(config["initial_heads"])
            use_prior = label != "previous_neural"
            methods[label] = {
                "head_sha256": checkpoint_hash,
                "distributions": score_distributions(
                    model,
                    data,
                    prior,
                    specifications,
                    mode=mode,
                    seed=config["seed"],
                    use_prior=use_prior,
                ),
                "ancestral": score_samples(
                    model,
                    data,
                    prior,
                    specifications,
                    mode=mode,
                    draws=config["draws"],
                    seed=config["seed"],
                    use_prior=use_prior,
                    progress=lambda n: print(
                        json.dumps({"cohort": cohort, "method": label, "sampled_contexts": n}),
                        flush=True,
                    ),
                ),
            }
            if label == "native_residual":
                methods[label]["zero_context"] = score_distributions(
                    model, data, prior, specifications, mode="facts", seed=config["seed"]
                )
                methods[label]["permuted_context"] = score_distributions(
                    model,
                    data,
                    prior,
                    specifications,
                    mode="native",
                    seed=config["seed"],
                    permute_latent=True,
                )
            print(
                json.dumps(
                    {
                        "cohort": cohort,
                        "method": label,
                        "contexts": len(specifications),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            # Keep each completed method even if a later comparison fails.
            (out / f"{cohort}-{label}.json").write_text(json.dumps(methods[label], indent=2) + "\n")
        donor = compare(
            rows, {"expanded_prior": prior.bank}, draws=config["draws"], seed=config["seed"]
        )
        if (
            cohort == "reused_development"
            and donor["methods"]["expanded_prior"]["metrics"] != config["golden_expanded_prior"]
        ):
            raise ValueError("The existing expanded donor baseline did not reproduce")
        methods["expanded_existing_prior"] = donor["methods"]["expanded_prior"]
        reports[cohort] = {
            "methods": methods,
            "data_sha256": manifest_hash,
            "contexts": len(specifications),
            "games": len({row["game"] for row in rows}),
        }
        (out / f"{cohort}.json").write_text(json.dumps(reports[cohort], indent=2) + "\n")
    result = {
        "complete": True,
        "cohorts": reports,
        "draws": config["draws"],
        "seed": config["seed"],
        "native_policy_changed": False,
        "search_connected": False,
        "automatic_promotion": False,
        "elapsed_seconds": time.monotonic() - started,
        "training_identity": result["identity"],
        "limitations": "Whole roster groups, repeated contexts/draws and finite k=8 require clustered uncertainty. Reused development is diagnostic; fresh rosters were selected before head results. No playing-strength gain is measured.",
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
