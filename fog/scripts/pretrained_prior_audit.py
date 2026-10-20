"""Compare a larger training-only set prior on fixed public-information contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fog.constants.schema import NO_ITEM, N_EV_UNITS, N_NATURES, TYPE_INDEX
from fog.constants.tables import TABLES
from fog.pretrained.belief_metrics import SampleMetrics
from fog.pretrained.prior_data import build_prior, digest
from fog.training.replay_candidates import ReplayCandidateBank
from fog.training.replay_policy_data import complete_private_team, replay_set_priors


def known_fields(tokens, tables=TABLES):
    sizes = np.array(
        [
            tables.n_species,
            tables.n_items,
            tables.n_abilities,
            *([tables.n_moves] * 4),
            N_NATURES,
            *([N_EV_UNITS] * 6),
            len(TYPE_INDEX),
        ]
    )
    known = (tokens >= 0) & (tokens < sizes)
    known[:, 1] &= tokens[:, 1] != tables.item_index(NO_ITEM)
    return known


def compare(contexts, banks, *, draws=8, seed=20261016):
    """Truth and provenance are scoring-only inputs, never proposal inputs."""
    if draws < 2 or not contexts:
        raise ValueError("Require fixed contexts and at least two matched draws")
    reports = {}
    priors = replay_set_priors()
    for name, bank in banks.items():
        rng = np.random.default_rng(seed)
        total, games = SampleMetrics(), {}
        samples_hash = hashlib.sha256()
        for row in contexts:
            public = np.asarray(row["reveals"], dtype=np.int64)
            truth = np.asarray(row["truth"], dtype=np.int64)
            verified = np.asarray(row["verified"], dtype=bool)
            if (
                public.shape != (6, 15)
                or truth.shape != public.shape
                or verified.shape != public.shape
            ):
                raise ValueError("Context does not have the modeled team shape")
            known = known_fields(public)
            facts = np.where(known, public, -1)
            samples = np.stack(
                [
                    complete_private_team(
                        facts,
                        seed=int(rng.integers(2**32)),
                        candidate_bank=bank,
                        timestamp=2,
                        usage_priors=priors,
                    )[0]
                    for _ in range(draws)
                ]
            )
            samples_hash.update(samples.astype("<i8").tobytes())
            total.update(samples, truth, verified, known)
            game = games.setdefault(
                row["game"],
                {
                    "metrics": SampleMetrics(),
                    "contexts": 0,
                    "team_rosters": row["team_rosters"],
                    "source_sha256": row["source_sha256"],
                },
            )
            if (
                game["team_rosters"] != row["team_rosters"]
                or game["source_sha256"] != row["source_sha256"]
            ):
                raise ValueError("A game changed identity across its perspectives")
            game["metrics"].update(samples, truth, verified, known)
            game["contexts"] += 1
        by_game = {}
        for game, value in games.items():
            m = value.pop("metrics")
            by_game[game] = {
                **value,
                "methods": m.report(),
                "resampling_data": {
                    "fields": m.fields,
                    "joint_count": m.joint_count,
                    "joint_correct": m.joint_correct,
                    "recall_count": m.recall_count,
                    "recall_correct": m.recall_correct,
                },
            }
        reports[name] = {
            "metrics": total.report(),
            "by_game": by_game,
            "sample_sha256": samples_hash.hexdigest(),
            "bank_sha256": bank.sha256,
        }
        print(
            json.dumps({"prior": name, "games": len(games), "contexts": len(contexts)}), flush=True
        )
    return {
        "complete": True,
        "contexts": len(contexts),
        "draws": draws,
        "seed": seed,
        "methods": reports,
        "training_performed": False,
        "policy_play_evaluated": False,
        "limitations": "Reused development games. Compare at matched k, resample connected game/roster groups, and do not interpret reconstructed prior evidence as verified historical team labels.",
    }


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if digest(config["contexts"]) != config["contexts_sha256"]:
        raise ValueError("Frozen context labels changed")
    contexts = json.loads(Path(config["contexts"]).read_text())
    metadata = build_prior(
        config["archive"], config["pools"], out / "prior", config["showdown"], **config["build"]
    )
    bank = ReplayCandidateBank(out / "prior/bank.npz")
    with np.load(out / "prior/bank.npz", allow_pickle=False) as source:
        if len(source["composition_ids"]) != len(bank.tokens) or set(
            source["composition_ids"]
        ) & set(metadata["excluded_compositions"]):
            raise ValueError("The new prior includes a held-out roster or lacks provenance")
    report = compare(
        contexts,
        {"baseline_prior": ReplayCandidateBank(config["baseline_bank"]), "expanded_prior": bank},
        draws=config["draws"],
        seed=config["seed"],
    )
    if report["methods"]["baseline_prior"]["metrics"] != config["golden_baseline_metrics"]:
        raise ValueError("Fixed baseline did not reproduce the previous matched-k benchmark")
    report.update(
        prior_manifest=metadata,
        baseline_reproduced=True,
        contexts_sha256=config["contexts_sha256"],
        context_games=len({r["game"] for r in contexts}),
        heldout_rosters_excluded=True,
        automatic_promotion=False,
    )
    (out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(json.loads(args.config.read_text()), args.out)
    print(json.dumps({k: v for k, v in report.items() if k not in ("methods", "prior_manifest")}))


if __name__ == "__main__":
    main()
