# Candidate-imputed training data for Kaggle

The selected method is **replay candidates with fixed June 2026 usage/spread priors**.
It had the highest exact full-schema set accuracy in the known-team comparison (0.86%
per Pokemon per draw). The native-prior hybrid had better move and partial-set scores,
but lower full-set recovery. This is a choice based on that benchmark, not a guarantee
that hidden sheets are accurate or that the resulting policy will be stronger.

## Training workflow

Use [the immutable experiment workflow](kaggle-workflow.md) after preparing and validating the
data below. `configs/automated_gen9ou_candidates_kaggle.yaml` is a bounded 24,000-decision
baseline with candidate-imputed replays as an imitation regularizer. It does not run a separate
behavior-cloning stage or automatically promote a model. Direct and replay audits are required.

The recipe uses `typed_dagger_2048.pt`, 512-row PPO minibatches, 256-row replay batches, and two
CPU rollout workers feeding one centralized GPU learner. Multiple allocated GPUs do not imply
distributed training. `notebooks/kaggle_candidates.ipynb` remains a manual bundle alternative.

A completed execution baseline processed 24,079 decisions and exported its final checkpoint in
about 23 minutes. This establishes that the pipeline runs. Subsequent pilot strength and retention
results are recorded separately in [research status](research-status.md).

## Dataset and split

The frozen selection contains the latest 2,000 unique Gen9OU public logs rated at
least 1650 from the already imported Metamon shard. There were 4,546 eligible unique
games in that local source. Dates span April 16, 2025 through May 18, 2026.
The chronological validation boundary is March 20, 2026 at 20:04:15 UTC.

| Partition | Games | Accepted decision versions |
|---|---:|---:|
| Training | 1,800 | 297,404 |
| Validation | 200 | 33,048 |
| Total | 2,000 | 330,452 |

No games were quarantined. Of 331,083 candidate decision versions, 631 (0.19%) were
excluded for called/unaligned moves or forced/unaligned switches. Donors supplied
fields for 70,971 of 72,000 sampled Pokemon slots; the remainder used the fallback.

The split is frozen **before** reconstruction. Every version of a battle remains in
one partition. The bank excludes all 645 source replays at or after the boundary,
including lower-rated donor-only games, as well as duplicate logs of excluded games.
It contains 77,608 public set fragments from 11,903 earlier games. Per-query filtering
also excludes the query battle, duplicate logs, and donors newer than the query.
Three sampled sheets are uncertainty augmentation, not three independent battles.

Raw data, all manifests and compressed shards, the bank, its exclusions, selection
hashes, quarantine/quality reports, and validation reports are under
`runs/gen9ou_candidates_kaggle_v1` and included in the archive. The manifests have
portable repository-relative shard paths. `provenance.json` records source revision,
bank hash, fixed-prior hashes, dates, and the serial/parallel comparison.

Limitations remain explicit:

- Inferred own sheets are never promoted to exact truth. Opponent observations use
  only the chronological public prefix; inferred opponent targets are excluded.
- The fixed June 2026 fallback snapshot postdates these battles. The donor split is
  chronological, but this is not a strictly historical forecasting experiment.
- A game-disjoint split does not guarantee disjoint players or hidden team identities.
  The existing checkpoint's historical exposure cannot be established.
- This is one cached Metamon raw-data shard, not a representative sample of all play.
- This production bank excludes held-out donors and differs from the benchmark bank;
  the benchmark accuracy numbers are not measured ground truth for this corpus.

## Rebuild

Use a new output directory if an existing conversion or split is complete, and update
the job paths accordingly. Run these commands in the FOG environment; native Metamon
dependencies are not required for this selected imputer.

```sh
python -m fog.scripts.prepare_candidate_training prepare --raw data/metamon_gen9ou_candidates --out runs/gen9ou_candidates_kaggle_v1 --games 2000 --min-rating 1650
python -m fog.scripts.build_replay_candidates --raw data/metamon_gen9ou_candidates --out runs/gen9ou_candidates_kaggle_v1/bank.npz --exclude-ids runs/gen9ou_candidates_kaggle_v1/donor_excluded_ids.json
python -m fog.scripts.convert_replay_decisions --raw runs/gen9ou_candidates_kaggle_v1/raw --out runs/gen9ou_candidates_kaggle_v1 --min-rating 1650 --imputations 3 --seed 20260920 --candidate-bank runs/gen9ou_candidates_kaggle_v1/bank.npz --workers 8
python -m fog.scripts.prepare_candidate_training finalize --out runs/gen9ou_candidates_kaggle_v1
python -m fog.scripts.validate_replay_decisions --manifest runs/gen9ou_candidates_kaggle_v1/manifest.parquet --out runs/gen9ou_candidates_kaggle_v1/validation_report.json
python -m fog.scripts.automated_training bundle-kaggle --job configs/automated_gen9ou_candidates_kaggle.yaml --out artifacts/fog-kaggle-candidates.zip
```

Worker count affects throughput, not imputation seeds. Use fewer workers on a machine
with less memory or fewer cores. Completed conversions are not overwritten. Retain the
quality report and investigate quarantines rather than relaxing data checks to fill a quota.

## Local verification

The focused suite passed 41 tests, including chronological split isolation and
required provenance bundling. Ruff passed on the changed Python modules and tests.
A serial conversion and the eight-worker conversion produced identical values in
all 21 arrays for the checked replay. The initial checkpoint loaded strictly and
completed an eight-row forward pass with finite action loss and zero optimizer steps.
The automated job also passed command-generation dry-run validation.
All 2,000 shards and 330,452 rows passed validation: zero chosen-illegal actions,
action-feature/legality mismatches, nonfinite action features, invalid retained policy
rows, or inferred opponent fields. The complete bundle inventory has no missing inputs.
