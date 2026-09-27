# Replay-candidate imputation

Implemented 20 September 2026. This is an explicit alternative to the species-marginal usage
baseline, not a claim that reconstructed sheets are true or that policy strength has improved.

## Design and review decisions

`fog.training.replay_candidates` extracts original submitted-set facts from public logs using the
same transfer, Transform and Illusion handling as replay supervision. It does not parse Metamon's
imputed sheets or feed generated completions back into its evidence. A donor contains one
Pokemon's observed item, ability, moves and Tera type, plus the source roster, rating, timestamp,
replay ID and public-log hash. Exact EVs, natures and IVs are not recoverable this way.

For each target Pokemon, the imputer filters donors by species, scalar consistency and the
four-move union constraint. It rejects the target replay, identical public logs, and donors dated
at or after the target. With no target date it uses the usage fallback. Identical set fragments
are grouped; their weights sum actual donor observations rather than counts of incomplete games
that merely *could* have contained the set. Sampling uses up to 32 candidate patterns, weighted
by recency (180-day half-life), rating proximity (400-point scale), teammate overlap and additional
observed information. These constants are fixed heuristics, not tuned on the benchmark test set.

One selected donor supplies its jointly observed fields. Remaining holes use the existing usage
fallback. Revealed target facts remain exact; **every donor-supplied value remains inferred**.
Separate random streams per Pokemon and attribute keep unrelated fallback EV/nature draws paired
between the baseline and candidate method. The implementation also fixes nonpacked move-hole
filling and supports Stellar when installing a completed sheet in poke-env.

The design review deliberately rejected three shortcuts:

- Using a target's complete hidden sheet to build its own candidate support, or allowing later
  replay evidence in a historical donor query.
- Counting each imputation as an independent battle in evaluation.
- Treating the more sophisticated Gen1–4 Metamon predictor as its Gen9 production method.
  [Pinned upstream predictor](https://github.com/UT-Austin-RPL/metamon/blob/0a00a759c9a4382a2877088d828302ec294a05a5/metamon/backend/team_prediction/predictor.py)
  falls back to the usage predictor outside Gen1–4 OU.

This implementation is related to Metamon's replay-candidate idea, but does not reproduce its
maximal-set graph or full-roster generation. Gen9 preview already supplies the roster. Partial
donors may still need marginal fallback, so it is not a completely joint team posterior.

## Data and conversion

The revision-pinned first raw shard provided 12,551 finished Gen9OU games rated at least 1500.
Extraction retained 12,547 games and 81,844 informative donor fragments; four games lacked the
required six-member previews. Slots with fewer than two observed moves or invalid species/moves/
abilities were excluded. Source revision: `792a86b5cdbcab9696032e947483b218453a0d4f`.
The bank contains 423 species and 27,783 distinct observed fragments; 11,304 donor records reveal
all four moves. It occupies approximately 1.8 MB compressed. These are observed fragments, not
81,844 verified complete submitted sets.

```sh
python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou_candidates --revision 792a86b5cdbcab9696032e947483b218453a0d4f --max-shards 1 --max-games 20000 --min-rating 1500
python -m fog.scripts.build_replay_candidates --raw data/metamon_gen9ou_candidates --out data/metamon_gen9ou_candidates/bank.npz
python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_probe --out runs/replay_imputation/converted_final --limit 25 --imputations 3 --candidate-bank data/metamon_gen9ou_candidates/bank.npz
python -m fog.scripts.validate_replay_decisions --manifest runs/replay_imputation/converted_final/manifest.parquet --out runs/replay_imputation/conversion_validation.json
```

Raising `--max-games` cannot produce more records from an exhausted shard. Raise `--max-shards`
and the download budget for a larger corpus. The raw importer remains resumable. Bank building is
an explicit rebuild and refuses to overwrite an existing bank. Bank vocabulary and content hashes
are checked/recorded; conversion reports donor coverage and fallback counts.

**Split real-replay training and evaluation before building the bank.** Either supply a
training-only raw directory, or supply `build_replay_candidates --exclude-ids holdout_ids.json`
with a JSON list of all held-out replay IDs. Known duplicate logs of those IDs are also excluded.
Leave-one-replay-out filtering alone does not prevent *other* held-out games contributing donors.
Do not build from all raw games and subsequently claim an independent random replay holdout.
The generated known-sheet benchmark below uses an entirely separate simulator corpus.

The converter defaults to the usage baseline unless `--candidate-bank` is supplied. This keeps a
reproducible control and avoids silently changing existing research runs before a strength benefit
has been established. Candidate conversion does not change the policy/AR-decoder interface.

## Known-sheet reconstruction and downstream pilot

`fog.eval.replay_imputation` generates local authoritative Showdown games with exact submitted
teams, stores only public logs as imputer inputs, and keeps truth sheets for scoring. It splits
deduplicated team compositions between policy training and test games. Public facts must agree
with the submitted sheet; contradictions are reported as failed games rather than quietly
rewritten using oracle truth.

The Metamon comparison runs its actual pinned move-sampling, scalar-sampling and set-merging
functions, extracted from checksum-verified source definitions. It supplies **the same FOG usage
priors** to isolate algorithm differences. It does not benchmark Metamon's full parser, historical
statistics selection, learned experimental predictor, or published reconstructed dataset.
Source downloads are explicit; no Metamon installation or replacement of poke-env is required.

```sh
python -m fog.eval.metamon_imputation_baseline --out runs/replay_imputation/upstream
# Set PYTHONHASHSEED=0 in the shell before launching Python: upstream merges moves using sets.
python -m fog.eval.replay_imputation --bank data/metamon_gen9ou_candidates/bank.npz --corpus runs/replay_imputation/corpus96.jsonl --generate-pool data/teams/pool_b --games 96 --samples 16 --metamon-source runs/replay_imputation/upstream --policy-steps 100 --out runs/replay_imputation/final_report.json
```

Omit `--generate-pool` to reuse the identical corpus. Use a fresh output corpus path to generate
another one. The pilot fits a small actual FOG policy for equal optimizer updates from matched
initializations (three seeds), on the common accepted decisions across methods. Held-out policy
inputs use exact own sheets with the same public replay parser. This is a controlled imitation
pilot, not a full production-architecture training run or a win-rate evaluation. Replay legality
remains reconstructed even with an exact sheet.

Metrics report hidden scalar accuracy, hidden-move recall, exact item+ability+moves+Tera sets,
exact full schema sets, and full-set recall among K draws. Moves are compared as sets, not ordered
lists. Joint scores are **per Pokemon**, not exact recovery of all six Pokemon simultaneously.
Bootstrap intervals resample whole test games and are conditional on the fixed pool and scripted
controllers. Sixteen samples are sixteen alternatives, not sixteen independent observations.

The completed run used 96 games, including 32 test games on the held-out half of the 16-team pool,
16 draws per perspective, and three matched training seeds. No games failed or truncated. The
policy pilot used 2,439 common training decisions, 1,231 test decisions and 100 updates per seed.

| Test metric | Usage baseline | Replay candidates | Metamon core, shared priors |
|---|---:|---:|---:|
| Hidden-move recall | 50.60% | 57.50% | 51.72% |
| Hidden-item accuracy | 49.53% | 46.31% | 50.25% |
| Exact item/ability/moves/Tera set | 2.02% | 3.48% | 2.29% |
| Exact full-schema set, per draw | 0.67% | 0.86% | 0.73% |
| Full-schema set recall among 16 draws | 9.38% | 10.94% | 8.07% |
| Policy action top-1, all three seeds | 65.48% | 65.48% | 65.48% |

The candidate-minus-usage hidden-move delta has an exploratory paired-game 95% interval of
+5.98 to +7.83 percentage points. The hidden-item delta is negative: -6.03 to -0.20 points.
The exact full-schema set delta interval includes zero (-0.11 to +0.52 points). Nature and EV
scores are identical between FOG's two arms because their fallback draws are now correctly paired.
Mean policy NLL is approximately 1.1032 / 1.1020 / 1.1024 respectively: essentially tied for this
small pilot. These results support better move reconstruction, not a general superiority claim.

End-to-end validation converted 25 real high-rated Metamon games into 4,083 decision versions,
with zero quarantines. Donors contributed to 888 of 900 sampled Pokemon slots; the other 12 used
the fallback. Shard validation found zero chosen-illegal actions, nonfinite action features,
action-feature/mask mismatches, invalid retained policy rows, or inferred opponent fields.
The focused regression suite passed 60 tests, with two existing tests skipped.

## Remaining limits and promotion criteria

The subsequent [native Metamon comparison](metamon-native-comparison.md) runs the
full upstream reconstruction pipeline with its own historical statistics, alongside
FOG arms using those same statistics. The shared-prior results above remain a
sampling control and should not be presented as results for native Metamon.

Observed donors have selection bias: moves and items that reveal themselves are overrepresented,
and longer games expose more complete sets. More replay volume does not itself correct this bias.
Recent/rating/roster weighting is heuristic and does not establish calibration. Fallback usage
priors remain the pinned June 2026 snapshot; historical donor exclusion does **not** make the
whole reconstruction strictly historically isolated. Historical usage snapshots are a separate
needed improvement for temporal experiments. IVs are not represented in FOG's set token schema.

The first controlled results show a move-reconstruction benefit but an item-accuracy regression;
the policy pilot does not establish a practical strength gain. Do not silently promote this to the
only training path. A follow-up should use a separate development corpus to correct reveal bias,
then evaluate on untouched diverse known-sheet games and a matched authoritative win-rate test.
Results and reproduction hashes are in `runs/replay_imputation/final_report.json`.
