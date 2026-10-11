"""Combine locked native/strong-opponent confirmation reports into a promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fog.pretrained.evaluation import confirmation_gate
from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256


def gate(primary: Path, strong_candidate: Path, strong_original: Path, probability_contract: Path):
    reports = [
        json.loads(path.read_text()) for path in (primary, strong_candidate, strong_original)
    ]
    main, candidate, original = reports
    numerical = json.loads(probability_contract.read_text())
    if (
        not numerical.get("passed")
        or not numerical.get("cpu_cuda_verified")
        or numerical.get("checkpoint_sha256") != main["protocol"]["candidate_sha256"]
        or numerical.get("interface") != main["protocol"]["interface"]
        or numerical.get("tolerance", float("inf")) > 1e-3
    ):
        raise ValueError("The selected finalist needs its own passed CPU/CUDA probability contract")
    protocols = [report["protocol"] for report in reports]
    for protocol in protocols:
        if protocol["stage"] != "confirmation" or protocol["original_sha256"] != KAKUNA_34_SHA256 or protocol.get("scripted_opponent"):
            raise ValueError(
                "Only final, unchanged-baseline comparisons can establish a strength gain"
            )
        for key in (
            "attempt",
            "pool_sha256",
            "interface",
            "sources_sha256",
            "temperature",
            "deadline_seconds",
        ):
            if protocol[key] != protocols[0][key]:
                raise ValueError("Confirmation source/protocol mismatch: " + key)
    if (
        main["protocol"]["foulplay"]
        or not candidate["protocol"]["foulplay"]
        or not original["protocol"]["foulplay"]
    ):
        raise ValueError("Require one native and two matched strong-opponent comparisons")
    if candidate["protocol"]["candidate_sha256"] != main["protocol"]["candidate_sha256"]:
        raise ValueError("Strong-opponent finalist differs from the primary model")
    if original["protocol"]["candidate_sha256"] != KAKUNA_34_SHA256:
        raise ValueError("Strong-opponent control must use unchanged Kakuna")
    if candidate["protocol"]["seed"] != original["protocol"]["seed"]:
        raise ValueError("Strong-opponent random seeds differ")
    for key in ("foulplay_inputs_sha256", "foulplay_search_ms"):
        if not candidate["protocol"].get(key) or (
            candidate["protocol"][key] != original["protocol"].get(key)
        ):
            raise ValueError("Strong-opponent inference inputs differ: " + key)
    if len(candidate["rows"]) != len(original["rows"]):
        raise ValueError("Strong-opponent update/game budgets differ")
    for a, b in zip(candidate["rows"], original["rows"]):
        if a["index"] != b["index"] or a["schedule"] != b["schedule"]:
            raise ValueError("Strong-opponent game schedules differ")
    failures = sum(report["failures"] for report in reports)
    unresolved = sum(report["unresolved"] for report in reports)
    if unresolved or failures:
        result = {
            "passed": False,
            "failures": failures,
            "unresolved": unresolved,
            "reason": "Final comparisons contain recorded failures or incomplete outcomes.",
        }
    else:
        scores = [row["score"] for row in main["rows"]]
        differences = [a["score"] - b["score"] for a, b in zip(candidate["rows"], original["rows"])]
        result = confirmation_gate(
            scores,
            attempt=main["protocol"]["attempt"],
            failures=failures,
            unresolved=unresolved,
            strong_differences=differences,
        )
    return dict(
        result,
        candidate_sha256=main["protocol"]["candidate_sha256"],
        evidence={
            str(path): sha256(path)
            for path in (primary, strong_candidate, strong_original, probability_contract)
        },
        automatic_promotion=False,
        attribution="This gate establishes play against unchanged Kakuna. Auxiliary contribution requires a separate matched ordinary/joint comparison.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--strong-candidate", type=Path, required=True)
    parser.add_argument("--strong-original", type=Path, required=True)
    parser.add_argument("--probability-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = gate(
        args.primary, args.strong_candidate, args.strong_original, args.probability_contract
    )
    if args.out.exists():
        raise FileExistsError("Do not overwrite confirmation evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
