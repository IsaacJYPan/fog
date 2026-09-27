Metamon Gen9OU integration, verified 20 September 2026

The optional [replay-candidate imputer and known-sheet benchmark](replay-candidate-imputation.md)
now expand the raw-data route beyond independent species priors. Read its split requirements
before building a donor bank for any held-out replay experiment.

Metamon does support Gen9OU. Its raw release backfilled 1.2 million Gen9OU games through 16 June 2025; the v6 release extends the overall corpus through 19 May 2026. The current raw metadata lists 2,681,841 records across formats, distributed over 46 Parquet shards, with approximately 5.70 GB of compressed downloads. That total is not a measured count of usable, high-rated Gen9OU games. [Raw dataset and version history](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays).

The parsed release contains reconstructed RL trajectories and explicitly incomplete revealed teams. Its v3 update added opponent team preview for Gen9. Missing private information is filled through team prediction. Those reconstructions cannot serve as exact FOG private-state labels. [Parsed dataset documentation](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays/blob/main/README.md).

FOG imports the original public protocol logs through `fog.scripts.import_metamon_replays`, then uses its existing chronological parser and uncertainty-aware converter. This retains FOG's observation, action, and provenance conventions. It does not require installing or executing Metamon's model code. Ordinary policy inference still uses the shared representation; the autoregressive decoder remains an auxiliary learning objective and a source of discrete search candidates.

The importer resolves a dataset commit, verifies downloaded shard hashes, and records the source commit, shard hash, and row number in every accepted replay. A manifest stores filters and the next unconsumed row. Resuming with changed filters, revision, shard content, or conflicting replay IDs fails explicitly. Raising `--max-games` continues a partial shard. Raising `--max-shards` allows subsequent shards. Download limits count network bytes in the current invocation; existing cached shards are verified and reused. `--parquet` accepts local files without a network request and requires a source revision label.

For a bounded initial import:

```sh
python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou --revision 792a86b5cdbcab9696032e947483b218453a0d4f --max-shards 1 --max-download-mb 256 --max-games 1000 --min-rating 1650
python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou --out data/metamon_gen9ou_decisions --limit 1000 --min-rating 1650 --imputations 3
python -m fog.scripts.split_replay_manifest --manifest data/metamon_gen9ou_decisions/manifest.parquet --out data/metamon_gen9ou_splits --holdout-fraction 0.1
```

The importer additionally accepts `--after YYYY-MM-DD` (inclusive) and `--before YYYY-MM-DD` (exclusive), interpreted as UTC upload dates. Use a new output directory when changing filters. Select dates according to the experiment's declared rules era. Upload date is only a proxy: a replay can have been uploaded later, and a date range alone does not certify compatibility with the pinned simulator. Preserve the full raw log and validate supported species, moves, mechanics, and rules before treating any corpus as a current-rules benchmark.

The conversion manifest retains source provider, revision, shard hash, and upload time, with `private_truth=false`. Inferred and unknown labels keep their existing masks. Multiple imputations are alternate reconstructions of the same game, not independent observations; keep them in the same split. Public replay policy imitation remains an approximation because the original player had private information absent from the log. A valid action mask and agreement across three reconstructions do not establish that the original private state has been recovered.

The actual smoke import used the pinned revision above and one 120,230,600-byte shard. It retained 100 Gen9OU games rated 1650–2000 after inspecting 1,206 records. Filters rejected 1,039 below the rating threshold, 58 with invalid or unavailable metadata, and 9 from other formats. The retained games span 20 March 2024 through 18 May 2026, so this is a compatibility sample spanning multiple eras. Ten games converted successfully into 1,926 decision versions across three imputations, with zero quarantined games. This is neither a representative corpus audit nor evidence of stronger play.

The local evidence is in `runs/research_improvements/metamon_conversion/quality_report.json` and `data/metamon_gen9ou_probe/.metamon/import.json`. Those generated directories are ignored by Git. No large training run consumes the smoke data automatically.

For a substantive experiment, freeze a rules-era corpus and its revision, reserve temporal and player-disjoint tests, audit reconstruction accuracy on private-truth self-play logs, then compare replay initialization against the existing baseline under the same training compute. Metamon's persistent pseudonyms can support player grouping, but public upload selection still biases the sample. Reserve exact, simulator-generated private labels for evaluating hidden-state calibration and candidate coverage.
