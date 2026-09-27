# Native Metamon reconstruction comparison

This experiment runs Metamon's actual forward parser, backward reconstruction,
`ReplayPredictor`, and both perspective-to-sequence validation paths. It replaces the
earlier shared-prior sampling comparison as the test of native reconstruction.
Metamon source is unmodified; a predictor subclass records calls and fixes random seeds.
Native dependencies are isolated from FOG's poke-env installation.

## Design

The existing 96-game known-team corpus has 64 training games and 32 test games from
disjoint halves of a 16-team pool. Submitted sheets are removed from worker inputs;
they are used only for scoring and oracle-sheet policy test observations. Sixteen
reconstruction draws are evaluated for each perspective. All methods are scored on
the same fields hidden according to FOG's public parser, with an additional audit of
each parser's revealed facts against submitted sheets.

Six arms distinguish changes in statistics, completion, and parsing:

1. FOG usage completion with its fixed June 2026 priors.
2. FOG replay candidates with the same fixed fallback priors.
3. FOG usage completion with Metamon-selected native priors and spreads.
4. FOG replay candidates with those native fallback priors and spreads.
5. Metamon's native predictor on FOG-extracted public evidence.
6. Metamon's full native reconstruction pipeline.

Native output is translated into FOG's schema without repairing wrong or missing
predictions from FOG's evidence. The report separates common-success scores from
available-output scores and records failures. The policy pilot uses matched FOG
models, three initialization seeds, 100 optimizer updates, and common accepted
training decisions. It does not train Metamon's agent or measure battle win rate.

## Results

Scores below use all 32 held-out games; there is no difference between common-success
and available-output subsets in this run. Set metrics are per draw and per Pokemon.

| Method | Hidden moves | Hidden items | Exact item/ability/moves/Tera | Exact full schema | Mean policy NLL |
|---|---:|---:|---:|---:|---:|
| FOG usage, fixed priors | 50.60% | 49.53% | 2.02% | 0.67% | 1.10317 |
| FOG candidates, fixed priors | 57.50% | 46.31% | 3.48% | 0.86% | 1.10198 |
| FOG usage, native priors | 52.58% | 49.68% | 2.83% | 0.57% | 1.10301 |
| FOG candidates, native priors | 57.90% | 47.00% | 4.38% | 0.70% | 1.10184 |
| Metamon predictor, FOG evidence | 51.87% | 48.96% | 2.62% | 0.59% | 1.10259 |
| Native Metamon end to end | 51.35% | 51.23% | 1.86% | 0.29% | 1.10254 |

Every method and seed reached 65.48% policy top-1 on 1,231 test decisions, after
training on the same 2,439 decisions. The near-identical losses and top-1 scores
provide no evidence of a useful downstream gain in this pilot; they do not prove
equivalence under longer training or on human decisions.

Native-minus-FOG-usage exploratory paired-game 95% bootstrap intervals include zero
for hidden moves (-1.02 to +2.59 percentage points) and hidden items (-3.19 to +6.80).
Exact full-schema accuracy is lower (-0.72 to -0.05 points). These intervals resample
games, not individual draws, and are conditional on the fixed pool and controllers;
they do not account for team-population uncertainty or multiple comparisons.

Native priors help FOG candidates on the item/ability/moves/Tera joint score
(+0.90 points; exploratory interval +0.42 to +1.38), but the full-schema score drops
from 0.86% to 0.70%. Thus this is a promising hybrid experiment, not an unconditional
improvement. It also remains below the usage baseline on hidden items.

The parser audit found 461 correct revealed moves in both methods, 186 versus 161
correct item facts (FOG versus native), and 79 versus 205 correct ability facts.
Neither parser had incorrect revealed facts in these games. More native ability
facts and broader species support do not offset every other difference: native
end-to-end full-schema accuracy is also lower than its predictor on FOG evidence.

**Decision:** retain the existing production default. Keep native priors injectable
and both reconstruction paths available for evaluation. A useful next experiment
would use a separate, more diverse known-sheet development set to investigate item
fact retention and rare-species handling, followed by untouched evaluation games
and a longer policy/win-rate comparison. Do not tune on this test pool and then
claim the resulting numbers as independent validation.

## Coverage and limitations

All 96 known-team games passed native reconstruction and sequence checks for all
16 draws. All 25 additional historical human replay probes passed native reconstruction
for one draw. Historical probes have no submitted truth sheets and establish coverage,
not imputation accuracy.

One historical control failed: FOG exports Sawsbuck as `<RARE_SPECIES>`, which cannot
be looked up by the native predictor. The native parser retained Sawsbuck and completed
the same replay. This is a concrete FOG vocabulary limitation, not evidence that every
native prediction in that battle was correct.

Native statistics are not automatically current. At the pinned source revision the
latest supported month is April 2026. The synthetic September 2026 games therefore
use April 2026 primary statistics at rank 1500 (their player protocol says Unrated).
Historical probes exercise 2024 half-year windows and ranks 1500, 1695, and 1825.
Native lower-rank and all-tier fallbacks remain enabled. Half-year windows and
all-time fallbacks are not strictly causal historical estimates.

The small, scripted team pool limits external validity. Sixteen draws from one game
are correlated alternatives, not independent games. Joint accuracy is per Pokemon;
full-schema accuracy covers FOG's item, ability, moves, nature, EVs and Tera fields,
and excludes IVs. Gen9 native prediction uses Metamon's usage-based fallback, not
its experimental learned joint predictor. Improvements in plausible sheets alone
do not establish stronger policies or calibrated beliefs.

## Provenance and reproduction

- Metamon: `https://github.com/UT-Austin-RPL/metamon`, commit
  `0a00a759c9a4382a2877088d828302ec294a05a5`.
- Supported poke-env fork: `https://github.com/UT-Austin-RPL/poke-env`, commit
  `e1268d270c3f2bd32c7ff5713e01062302020579`, version 0.8.3.3.
- Usage dataset: `jakegrigsby/metamon-usage-stats`, v5 revision
  `0c2939074d16fcf28bf10d8eacc5c323fba51aaf`.
- Native Gen9 checks archive SHA256:
  `8efc65e99b60485d6d28832909018a3d35c7aef86a749eebe732fd408cf281bd`.
- Native Gen9 movesets archive SHA256:
  `4568e382ce332427e4fb776a701514d7ba8661eaecedce3bd71b8340c6198633`.
- Corpus SHA256:
  `b8293ab91f8d1da2fe59457128b380eda6ad44fbb40f8b77079409b63fcef393`.
- Candidate bank SHA256:
  `7e75eb8fee23c4031e8123724416d0421a15cf8c175a8d19eabf253b78eb47e8`.

Local artifacts live under `runs/metamon_native/`: public queries, native results,
historical probe queries/results, dataset metadata, isolated source/runtime/cache,
and `report.json`. The report also hashes the native output. Large artifacts are not
committed. Keep both Python environments separate; invoke the native worker by file
path so it does not import FOG.

Validation: 39 focused regression tests passed; Ruff passed for the comparison,
worker, modified imputer, and new tests. The production change is limited to optional
usage/spread injection in `complete_private_team`; omitted arguments preserve the
existing path. The native runner and six-arm comparison are evaluation utilities.

```sh
# FOG environment; output paths must be fresh for prepare and worker.
python -m fog.eval.metamon_native_comparison prepare --corpus runs/replay_imputation/corpus96.jsonl --out runs/metamon_native/public_queries.jsonl

# Native environment: pinned Metamon and supported poke-env installed, native usage
# archives extracted under METAMON_CACHE_DIR/usage-stats/{checks_data,movesets_data}/gen9.
# Set PYTHONHASHSEED=0 before starting Python and METAMON_CACHE_DIR to the native cache.
python fog/eval/metamon_native_worker.py --input runs/metamon_native/public_queries.jsonl --out runs/metamon_native/native_results.jsonl --source runs/metamon_native/source --draws 16 --seed 19

# FOG environment again.
python -m fog.eval.metamon_native_comparison score --corpus runs/replay_imputation/corpus96.jsonl --native runs/metamon_native/native_results.jsonl --bank data/metamon_gen9ou_candidates/bank.npz --policy-steps 100 --out runs/metamon_native/report.json
```
