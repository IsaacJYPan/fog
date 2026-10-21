# Metamon Gen9OU import

FOG imports public logs from Metamon, then runs its chronological parser and replay
converter. Metamon's model code is not needed. For donor-based reconstruction, see
[replay-candidate imputation](replay-candidate-imputation.md), including its split rules.

The September 2026 dataset check found 2,681,841 raw records across formats in 46
Parquet shards, about 5.70 GB compressed. The initial release included 1.2 million
Gen9OU games through 16 June 2025; v6 extends the overall corpus through 19 May 2026.
These totals do not count usable, high-rated Gen9OU games separately.
[Raw dataset and version history](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays).

Use the raw logs rather than treating the parsed release as private-state truth.
The parsed release fills missing information with team prediction; its v3 update
added Gen9 opponent preview but does not recover exact hidden sheets.
[Parsed dataset documentation](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays/blob/main/README.md).

## Import and convert

The importer resolves a dataset commit, verifies downloaded shard hashes, and records the source commit, shard hash, and row number in every accepted replay. A manifest stores filters and the next unconsumed row. Resuming with changed filters, revision, shard content, or conflicting replay IDs fails explicitly. Raising `--max-games` continues a partial shard. Raising `--max-shards` allows subsequent shards. Download limits count network bytes in the current invocation; existing cached shards are verified and reused. `--parquet` accepts local files without a network request and requires a source revision label.

For a bounded initial import:

```sh
python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou --revision 792a86b5cdbcab9696032e947483b218453a0d4f --max-shards 1 --max-download-mb 256 --max-games 1000 --min-rating 1650
python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou --out data/metamon_gen9ou_decisions --limit 1000 --min-rating 1650 --imputations 3
python -m fog.scripts.split_replay_manifest --manifest data/metamon_gen9ou_decisions/manifest.parquet --out data/metamon_gen9ou_splits --holdout-fraction 0.1
```

The importer additionally accepts `--after YYYY-MM-DD` (inclusive) and `--before YYYY-MM-DD` (exclusive), interpreted as UTC upload dates. Use a new output directory when changing filters. Select dates according to the experiment's declared rules era. Upload date is only a proxy: a replay can have been uploaded later, and a date range alone does not certify compatibility with the pinned simulator. Preserve the full raw log and validate supported species, moves, mechanics, and rules before treating any corpus as a current-rules benchmark.

The conversion manifest retains source provider, revision, shard hash, and upload time, with `private_truth=false`. Inferred and unknown labels keep their existing masks. Multiple imputations are alternate reconstructions of the same game, not independent observations; keep them in the same split. Public replay policy imitation remains an approximation because the original player had private information absent from the log. A valid action mask and agreement across three reconstructions do not establish that the original private state has been recovered.

## Smoke run

The pinned revision above was checked in September 2026 using one 120,230,600-byte
shard. It retained 100 Gen9OU games rated 1650–2000 after inspecting 1,206 records:
1,039 fell below the rating threshold, 58 had invalid or unavailable metadata, and 9
used other formats. Ten games converted to 1,926 decision versions across three
imputations, with zero quarantines. The retained games span 20 March 2024 through
18 May 2026, so this checks compatibility across several eras, not corpus quality
for a current-rules experiment.

The reports are `runs/research_improvements/metamon_conversion/quality_report.json`
and `data/metamon_gen9ou_probe/.metamon/import.json`; both directories are ignored by Git.

For a substantive experiment, freeze a rules-era corpus and its revision, reserve temporal and player-disjoint tests, audit reconstruction accuracy on private-truth self-play logs, then compare replay initialization against the existing baseline under the same training compute. Metamon's persistent pseudonyms can support player grouping, but public upload selection still biases the sample. Reserve exact, simulator-generated private labels for evaluating hidden-state calibration and candidate coverage.
