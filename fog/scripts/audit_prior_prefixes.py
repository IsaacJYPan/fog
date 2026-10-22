"""Diagnose frozen AR support on public, correct and generated prefixes."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import torch

from fog.models.donor_prior import DonorPrior
from fog.pretrained.frozen_belief import FrozenBeliefData, digest, load_heads, public_payload
from fog.pretrained.joint import BeliefHeads
from fog.pretrained.prefix_audit import ObservedPriorBatch, PrefixObservations, compatible_sets, summarize
from fog.pretrained.prior_head_metrics import evaluation_queries


def public_support(uncapped, payload, truth, verified, metadata):
    rows = []
    public = payload["tokens"].clone()
    public = torch.where(public >= 0, public, -1)
    # The prepared prior already reflects the model's submitted-item sentinel handling.
    for i, team in enumerate(public.cpu()):
        roster = tuple(sorted(int(v) for v in team[:, 0]))
        for slot, facts in enumerate(team):
            capped = payload["prior"].tokens[i, slot]
            weights = payload["prior"].weights[i, slot]
            all_sets, _ = uncapped.retrieve(roster, tuple(int(v) for v in facts))
            all_sets = torch.from_numpy(all_sets)
            eligible = bool(verified[i, slot].all())
            target = truth[i, slot]
            before = eligible and bool(compatible_sets(all_sets, target).any())
            after = eligible and bool((compatible_sets(capped, target) & (weights > 0)).any())
            at_512 = eligible and bool(compatible_sets(all_sets[:512], target).any())
            rows.append({**metadata[i], "slot": slot, "complete_verified_set": eligible,
                         "donors_before_cap": len(all_sets), "donors_after_cap": int((weights > 0).sum()),
                         "truth_set_before_cap": before, "truth_set_after_cap": after,
                         "truth_set_at_512": at_512,
                         "truth_set_lost_by_cap": before and not after})
    return rows


@torch.no_grad()
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
        raise ValueError("Prior differs from the sealed training recipe")
    prior = DonorPrior(Path(config["prior"]), max_donors=recipe["max_donors"],
                       smoothing=recipe["smoothing"])
    uncapped = DonorPrior(prior.bank, max_donors=len(prior.bank.tokens),
                         smoothing=recipe["smoothing"])
    raw_path = out / "observations.jsonl.gz"
    result = {"complete": False, "cohorts": {}, "native_policy_changed": False,
              "training_performed": False, "search_connected": False,
              "config_sha256": digest(out / "config.json"), "head_hashes": {},
              "limitations": "Descriptive query counts, not independent trials. Move truth mass means another remaining correct move, not ordered-move NLL. Generated prefixes may contradict truth: do not use such labels as valid conditional supervision."}
    with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
        for cohort, specification in config["cohorts"].items():
            folder = Path(specification["data"])
            data = FrozenBeliefData(folder, specification["manifest_sha256"],
                                    required_splits=(specification["split"],))
            if data.manifest["native_sha256"] != recipe["native_sha256"]:
                raise ValueError("Frozen native feature identity differs")
            specs = evaluation_queries(data, split=specification["split"],
                                       maximum_games=specification.get("maximum_games"))
            records = {r["path"]: r for r in data.rows[specification["split"]]}
            reports = {}
            for method, filename, mode in [
                ("prior_only", "prior_only.pt", "native"),
                ("facts", "facts_heads_final.pt", "facts"),
                ("native", "native_heads_final.pt", "native"),
            ]:
                model = BeliefHeads().eval()
                load_heads(model, training / filename, native_sha256=recipe["native_sha256"],
                           prior_sha256=recipe["prior_sha256"])
                fingerprint = digest(training / filename)
                result["head_hashes"][method] = fingerprint
                source_rows = {source: [] for source in ["public", "correct_prefix", "generated"]}
                support = []
                for context_index, spec in enumerate(specs):
                    if time.monotonic() - started > config.get("maximum_seconds", 600):
                        raise TimeoutError("Preserve the partial audit; exceeded the declared local cap")
                    name, index = spec
                    record = records[name]
                    phase = "early" if index == 0 else (
                        "late" if index == record["positions"] - 1 else "middle"
                    )
                    meta = [{"game": record["game"], "phase": phase, "perspective": name,
                             "position": index, "roster_groups": record["roster_groups"]}]
                    batch = data.batch([spec])
                    labels = batch["labels"]
                    truth = torch.where(labels["verified"], labels["truth"], -1)
                    payload = public_payload(model, labels["reveals"], prior)
                    normalized_public = torch.where(
                        model.belief._revealed_mask(labels["reveals"]).reshape_as(truth),
                        labels["reveals"], -1,
                    )
                    if method == "prior_only":
                        support.extend(public_support(
                            uncapped, {**payload, "tokens": normalized_public},
                            truth, labels["verified"], meta,
                        ))
                    latent = batch["latent"] if mode == "native" else torch.zeros_like(batch["latent"])
                    context = model.context_projection(latent)
                    for source in source_rows:
                        observations = PrefixObservations(
                            truth, labels["verified"], normalized_public, meta, source=source
                        )
                        wrapped = {**payload, "prior": ObservedPriorBatch(payload["prior"], observations)}
                        generator = torch.Generator().manual_seed(config["seed"] + context_index)
                        if source == "public":
                            model.belief(context, wrapped)
                        elif source == "correct_prefix":
                            q = config["queries"]
                            wrapped = {"tokens": wrapped["tokens"].repeat_interleave(q, 0),
                                       "masks": {k: v.repeat_interleave(q, 0) for k, v in wrapped["masks"].items()},
                                       "prior": wrapped["prior"].repeat_interleave(q)}
                            model.belief(context.repeat_interleave(q, 0), wrapped,
                                         teacher_values=truth.repeat_interleave(q, 0),
                                         generator=generator, selected_only=True)
                        else:
                            sampled = model.belief.sample_team(
                                context, wrapped, k=config["draws"], generator=generator,
                                return_trace=True,
                            )
                            # One real production-path transparency check per method/cohort.
                            if context_index == 0:
                                plain = model.belief.sample_team(
                                    context, payload, k=config["draws"],
                                    generator=torch.Generator().manual_seed(config["seed"]),
                                    return_trace=True,
                                )
                                for field in ["teams", "log_value_prob", "log_route_prob"]:
                                    if not torch.equal(getattr(sampled, field), getattr(plain, field)):
                                        raise ValueError("Offline observer changed production sampling")
                        source_rows[source].extend(observations.rows)
                        for row in observations.rows:
                            stream.write(json.dumps({"cohort": cohort, "method": method, **row}) + "\n")
                reports[method] = {key: summarize(rows) for key, rows in source_rows.items()}
                if support:
                    eligible = [r for r in support if r["complete_verified_set"]]
                    reports["public_support"] = {
                        "set_contexts": len(eligible),
                        "supported_before_cap": sum(r["truth_set_before_cap"] for r in eligible),
                        "supported_after_cap": sum(r["truth_set_after_cap"] for r in eligible),
                        "supported_at_512": sum(r["truth_set_at_512"] for r in eligible),
                        "truth_set_lost_by_cap": sum(r["truth_set_lost_by_cap"] for r in eligible),
                        "no_public_donors": sum(r["donors_after_cap"] == 0 for r in eligible),
                        "by_context": support,
                    }
                print(json.dumps({"cohort": cohort, "method": method, "contexts": len(specs),
                                  "seconds": time.monotonic() - started}), flush=True)
                (out / f"{cohort}-{method}.json").write_text(json.dumps(reports[method], indent=2) + "\n")
            result["cohorts"][cohort] = {"contexts": len(specs), "methods": reports,
                                        "games": len({records[name]["game"] for name, _ in specs})}
    result.update(complete=True, raw_observations_sha256=digest(raw_path),
                  elapsed_seconds=time.monotonic() - started, production_trace_transparency_passed=True)
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
