"""Compare native learned sampling with FOG set inference on fixed exact-team development games."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))

import numpy as np
import torch

from fog.constants.tables import TABLES
from fog.encoding.reveals import masks_for_tokens
from fog.pretrained.belief_metrics import SampleMetrics
from fog.pretrained.checkpoints import load_auxiliary
from fog.pretrained.data import assert_disjoint
from fog.pretrained.joint import JointKakuna
from fog.pretrained.kakuna import load_kakuna, sha256
from fog.pretrained.labels import exact_team_tokens
from fog.pretrained.loading import read_record, SequenceLoader
from fog.training.replay_candidates import canonical_row, ReplayCandidateBank, vocabulary_digest
from fog.training.replay_policy_data import complete_private_team, replay_set_priors


def training_bank(pools, out):
    teams = pools["train"]["teams"]
    tokens, rosters, ids, hashes = [], [], [], []
    for index, team in enumerate(teams):
        truth = exact_team_tokens(team, TABLES)
        digest = hashlib.sha256(team.encode()).hexdigest()
        for row in truth:
            tokens.append(canonical_row(row))
            rosters.append(truth[:, 0])
            ids.append("exact-train-pool-" + str(index))
            hashes.append(digest)
    metadata = {
        "schema_version": 1,
        "format": "gen9ou",
        "vocabulary_sha256": vocabulary_digest(),
        "private_truth": True,
        "method": "existing FOG donor inference using submitted training-pool sets only; no held-out donors",
        "donor_rows": len(tokens),
        "timestamp_scope": "synthetic availability ordering, not historical replay dates",
    }
    np.savez_compressed(
        out,
        tokens=np.asarray(tokens, dtype=np.int32),
        rosters=np.asarray(rosters, dtype=np.int32),
        replay_ids=np.asarray(ids),
        log_hashes=np.asarray(hashes),
        timestamps=np.ones(len(tokens), dtype=np.int64),
        ratings=np.zeros(len(tokens), dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return ReplayCandidateBank(out)


def benchmark(native, auxiliary, fresh, pools_path, out, *, games=32, draws=32, seed=20261009):
    if out.exists() or draws < 2 or games < 1:
        raise ValueError("Choose a new benchmark folder and at least two samples per context")
    pools = {
        split: json.loads((pools_path / (split + ".json")).read_text())
        for split in ("train", "development", "final")
    }
    assert_disjoint(pools)
    collection = json.loads((fresh / "collection.json").read_text())
    if collection["split"] != "development" or collection["pool_sha256"] != sha256(
        pools_path / "development.json"
    ):
        raise ValueError(
            "Belief benchmark requires composition-disjoint exact-team development games"
        )
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    runtime = load_kakuna(native, verify_original=False)
    model = JointKakuna(runtime.model).eval()
    state = load_auxiliary(model, auxiliary, native)
    bank = training_bank(pools, out / "training-bank.npz")
    priors = replay_set_priors()
    loader = SequenceLoader(runtime, sequence_length=1000)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    metrics = {name: SampleMetrics() for name in ("learned", "existing_set_inference")}
    files = sorted(fresh.glob("*.json.gz"))[:games]
    if len(files) != games:
        raise ValueError("The fixed benchmark cohort is incomplete")
    contexts, trace_finite, outcome_errors = 0, True, []
    with torch.inference_mode():
        for game, path in enumerate(files):
            record = read_record(path)
            if not record["training_eligible"] or record.get("failures") or record.get("fallbacks"):
                raise ValueError("Failed development games must be diagnosed before benchmarking")
            for pov_seed in (0, 1):
                example, targets = loader.load(("fresh", str(path), pov_seed))
                points = {0, len(example.actions) // 2, len(example.actions) - 1}
                hidden = runtime.model.traj_encoder.init_hidden_state(1, torch.device("cpu"))
                for index in range(len(example.actions)):
                    latent, hidden = runtime.model.get_state_embedding(
                        obs={
                            key: value[None, index : index + 1]
                            for key, value in example.obs.items()
                        },
                        rl2s=example.rl2s[None, index : index + 1],
                        time_idxs=example.time_idxs[None, index : index + 1],
                        hidden_state=hidden,
                    )
                    if index not in points:
                        continue
                    revealed = targets["reveals"][index].numpy()
                    masks = {
                        key: torch.from_numpy(value[None])
                        for key, value in masks_for_tokens(revealed, TABLES).items()
                    }
                    payload = {"tokens": targets["reveals"][index : index + 1], "masks": masks}
                    trace = model.sample_team(
                        latent[:, 0], payload, k=draws, generator=generator, return_trace=True
                    )
                    trace_finite &= bool(
                        torch.isfinite(trace.log_value_prob).all()
                        and torch.isfinite(trace.log_route_prob).all()
                    )
                    learned = trace.teams[0].numpy()
                    known = model.belief._revealed_mask(payload["tokens"]).reshape(6, 15).numpy()
                    public_facts = np.where(known, revealed, -1)
                    baseline = []
                    for _ in range(draws):
                        baseline.append(
                            complete_private_team(
                                public_facts,
                                seed=int(rng.integers(2**32)),
                                candidate_bank=bank,
                                timestamp=2,
                                usage_priors=priors,
                            )[0]
                        )
                    for name, samples in (
                        ("learned", learned),
                        ("existing_set_inference", baseline),
                    ):
                        metrics[name].update(
                            samples,
                            targets["truth"][index].numpy(),
                            targets["verified"][index].numpy(),
                            known,
                        )
                    prediction = float(model.outcome(latent[:, 0]).tanh().item())
                    outcome_errors.append((prediction - float(targets["outcomes"][index])) ** 2)
                    contexts += 1
            print(json.dumps({"games": game + 1, "contexts": contexts}), flush=True)
    report = {
        "native_sha256": sha256(native),
        "auxiliary_sha256": sha256(auxiliary),
        "completed_updates": state["update"],
        "training_bank_sha256": bank.sha256,
        "development_pool_sha256": sha256(pools_path / "development.json"),
        "input_games": {path.name: sha256(path) for path in files},
        "games": games,
        "contexts": contexts,
        "draws_per_context": draws,
        "seed": seed,
        "sampling_trace_finite": trace_finite,
        "methods": {name: value.report() for name, value in metrics.items()},
        "outcome_mse": float(np.mean(outcome_errors)),
        "search_connected": False,
        "automatic_promotion": False,
        "limitations": "Monte Carlo calibration depends on sample count. Set metrics cover encoded categories, excluding IVs and distinguishing only modeled item buckets. This does not measure downstream search play; that requires fixed-search matched games before connecting the learned sampler.",
    }
    (out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--draws", type=int, default=32)
    args = parser.parse_args()
    benchmark(
        args.native,
        args.auxiliary,
        args.fresh,
        args.pools,
        args.out,
        games=args.games,
        draws=args.draws,
    )


if __name__ == "__main__":
    main()
