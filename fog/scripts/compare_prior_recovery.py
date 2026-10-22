"""Matched CPU comparison of larger donor support and fallback-only residuals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fog.models.donor_prior import DonorPrior
from fog.pretrained.frozen_belief import FrozenBeliefData, digest, load_heads
from fog.pretrained.grouped_belief_review import density_comparison, sampler_comparisons
from fog.pretrained.joint import BeliefHeads
from fog.pretrained.prior_head_metrics import evaluation_queries, score_distributions, score_samples
from fog.scripts.evaluate_prior_heads import contexts
from fog.scripts.pretrained_prior_audit import compare


def probability_probe(config):
    """Frozen public-only CPU probe across local and Kaggle interpreter versions."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    training = Path(config["training"])
    recipe = json.loads((training / "recipe.json").read_text())
    specification = next(iter(config["cohorts"].values()))
    data = FrozenBeliefData(specification["data"], specification["manifest_sha256"],
                            required_splits=("development",))
    selected = evaluation_queries(data)[:4]
    batch = data.batch(selected)
    from fog.pretrained.frozen_belief import public_payload

    result = {}
    with torch.no_grad():
        for condition in config["conditions"]:
            model = BeliefHeads().eval()
            load_heads(model, training / condition["checkpoint"],
                       native_sha256=recipe["native_sha256"], prior_sha256=recipe["prior_sha256"])
            prior = DonorPrior(Path(config["prior"]), max_donors=condition["max_donors"],
                               smoothing=recipe["smoothing"], residual_mode=condition["residual_mode"])
            latent = batch["latent"]
            if condition["context_mode"] == "facts":
                latent = torch.zeros_like(latent)
            output = model.belief(model.context_projection(latent),
                                  public_payload(model, batch["labels"]["reveals"], prior))
            result[condition["name"]] = {
                family: values.softmax(-1).tolist() for family, values in output.logits.items()
            }
    return {"torch": torch.__version__, "contexts": len(selected), "probabilities": result}


def compare_probability_probes(reference, probe, *, tolerance=0.001):
    if not 0 < tolerance <= 0.001 or reference["contexts"] != probe["contexts"]:
        raise ValueError("Probability probe coverage or tolerance changed")
    expected, actual = reference["probabilities"], probe["probabilities"]
    if not expected or expected.keys() != actual.keys():
        raise ValueError("Probability probe conditions changed")
    errors = {}
    for name, fields in expected.items():
        if not fields or fields.keys() != actual[name].keys():
            raise ValueError("Probability probe field coverage changed")
        errors[name] = {}
        for family, values in fields.items():
            left, right = torch.tensor(values), torch.tensor(actual[name][family])
            if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError("Probability probe has invalid shape or nonfinite values")
            error = float((left - right).abs().max())
            if not math.isfinite(error) or error > tolerance:
                raise ValueError("Local/Kaggle CPU probability agreement failed")
            errors[name][family] = error
    return {"passed": True, "local_torch": reference["torch"], "kaggle_torch": probe["torch"],
            "contexts": probe["contexts"], "maximum_errors": errors, "tolerance": tolerance,
            "actual_CUDA_test": False}


def run(config, out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    training = Path(config["training"])
    recipe = json.loads((training / "recipe.json").read_text())
    if digest(config["prior"]) != recipe["prior_sha256"]:
        raise ValueError("Comparison bank differs from the saved head identity")
    reports = {}
    for cohort, specification in config["cohorts"].items():
        data = FrozenBeliefData(
            specification["data"], specification["manifest_sha256"],
            required_splits=("development",), allow_trial=config.get("local_trial", False),
        )
        if data.manifest["native_sha256"] != recipe["native_sha256"]:
            raise ValueError("Latents belong to another frozen native model")
        selected = evaluation_queries(data, maximum_games=specification.get("maximum_games"))
        methods = {}
        for condition in config["conditions"]:
            name = condition["name"]
            if name in methods:
                raise ValueError("Duplicate condition")
            model = BeliefHeads().eval()
            filename = training / condition["checkpoint"]
            load_heads(model, filename, native_sha256=recipe["native_sha256"],
                       prior_sha256=recipe["prior_sha256"])
            prior = DonorPrior(
                Path(config["prior"]), max_donors=condition["max_donors"],
                smoothing=recipe["smoothing"], residual_mode=condition["residual_mode"],
            )
            methods[name] = {
                "condition": condition, "head_sha256": digest(filename),
                "distributions": score_distributions(
                    model, data, prior, selected, mode=condition["context_mode"], seed=config["seed"],
                ),
                "ancestral": score_samples(
                    model, data, prior, selected, mode=condition["context_mode"],
                    seed=config["seed"], draws=config["draws"],
                    progress=lambda n: print(json.dumps({"cohort": cohort, "method": name,
                                                        "contexts": n}), flush=True),
                ),
            }
            golden = specification.get("golden", {}).get(name)
            if golden is not None and methods[name]["ancestral"]["metrics"] != golden:
                raise ValueError("Historical control sampling did not reproduce: " + name)
            golden_density = specification.get("golden_distributions", {}).get(name)
            if golden_density is not None:
                current = methods[name]["distributions"]
                for route in ["public_marginal", "correct_prefix"]:
                    for family, fields in golden_density[route].items():
                        for key, value in fields.items():
                            if not math.isclose(current[route][family][key], value,
                                                rel_tol=0, abs_tol=1e-5):
                                raise ValueError("Historical probability control changed: " + name)
            (out / f"{cohort}-{name}.json").write_text(json.dumps(methods[name], indent=2) + "\n")
            print(json.dumps({"cohort": cohort, "method": name, "complete": True,
                              "seconds": time.monotonic() - started}), flush=True)
        donor = compare(contexts(data, selected), {"expanded_existing_prior": prior.bank},
                        draws=config["draws"], seed=config["seed"])["methods"]["expanded_existing_prior"]
        golden = specification.get("golden", {}).get("expanded_existing_prior")
        if golden is not None and donor["metrics"] != golden:
            raise ValueError("Historical expanded donor control did not reproduce")
        sampling = {name: value["ancestral"] for name, value in methods.items()}
        sampling["expanded_existing_prior"] = donor
        pairs = [tuple(pair) for pair in config["pairs"]]
        statistics = sampler_comparisons(sampling, pairs=pairs, replicates=config["replicates"])
        density = {
            right + "_minus_" + left: density_comparison(
                methods[left]["distributions"], methods[right]["distributions"],
                replicates=config["replicates"],
            )
            for left, right in pairs if left in methods and right in methods
        }
        reports[cohort] = {
            "contexts": len(selected), "statistics": statistics, "density": density,
            "methods": methods, "expanded_existing_prior": donor,
            "data_sha256": specification["manifest_sha256"],
            "historical_control_checks": sorted(specification.get("golden", {})),
        }
        (out / f"{cohort}.json").write_text(json.dumps(reports[cohort], indent=2) + "\n")
    result = {
        "complete": True, "cohorts": reports, "draws": config["draws"], "seed": config["seed"],
        "elapsed_seconds": time.monotonic() - started, "native_policy_changed": False,
        "training_performed": False, "GPU_requested": False, "search_connected": False,
        "automatic_promotion": False, "automatic_rollback": False,
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "threads": torch.get_num_threads(),
                    "deterministic": torch.are_deterministic_algorithms_enabled()},
        "limitations": "Consumed development evidence, exploratory whole connected-roster groups and finite Monte Carlo draws. Larger support is not verified truth. Gating cannot correct a supported wrong prior. No new playing-strength or integrated deadline comparison.",
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-reference", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.probe_reference:
        reference = json.loads(args.probe_reference.read_text())
        probe = probability_probe(config)
        report = compare_probability_probes(reference, probe)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    else:
        run(config, args.out)


if __name__ == "__main__":
    main()
