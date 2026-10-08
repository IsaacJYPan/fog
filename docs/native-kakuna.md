# Native Kakuna experiments

This optional path starts from Metamon's Kakuna checkpoint 34. FOG's earlier models and checkpoints remain independent baselines. No pretrained checkpoint is committed to Git.

The integration preserves Kakuna's native observation encoder, temporal policy, actor, critic, and weights. A parameter-free SDPA implementation preserves the native 96-step causal lookback (97 positions including the current step) on CPU and T4. A common deployment adapter reapplies authoritative legal-action masks because the upstream actor's probability floor gives illegal actions a small mass. This adapter is identical for unchanged Kakuna and both trained conditions. Team preview uses a fixed order in all neural comparisons.

Joint training projects the 900-dimensional shared latent into FOG's fact-set autoregressive belief decoder. Ordinary policy inference reads the native latent directly. Belief sampling retains constraints and probability traces, but is not connected to search until an exact-team benchmark supports it. Exact-set metrics cover the represented 15 categorical fields per Pokémon; IVs are not represented.

The preserved native observation uses base stats and HP fractions rather than the acting team's EVs, nature, IVs, or complete calculated stats. It can therefore omit strategically relevant information the player knows. This is an input limitation to diagnose, not evidence that the architecture is saturated. Fixed team-preview ordering is another limitation of these comparisons.

## Data and training

`pretrained_teams` validates submitted teams through the pinned Showdown simulator and groups splits by species composition: 500 training, 100 development, and 100 final teams. Final teams form two distinct confirmation cohorts. Candidate banks use training teams only. The additional-training split cannot guarantee absence from Kakuna's historical pretraining.

`pretrained_human` retains complete published match pairs, rating/date metadata, hashes, and conservative held-out roster exclusions. `pretrained_human_screen` checks reconstructed own sets against current rules, screens every native action, and tests a fixed observation-loading sample. Reconstructed human set labels remain unverified and receive no belief supervision. The available human archive ends in June 2025; its age is recorded rather than described as current data.

`pretrained_collect` generates exact submitted-team games against Foul Play, frozen Kakuna, and historical FOG learners in a seat-balanced 40/30/30 schedule. Player processes receive public protocol messages and their own requests. Complete teams enter offline label creation only. Failed attempts are retained, and malformed, truncated, or fallback games cannot enter training. Collection can run in private CPU notebooks without consuming the GPU budget.

Own roster reconciliation uses the player's private request to remove parser ghosts after Illusion. A native switch rejected because of hidden trapping is retried from its preceding temporal state using Showdown's updated request; the rejected action never becomes a training target. These information-revealing rejections are recorded separately. External teacher rejections and other native errors still fail the game. Foul Play uses a persistent spawn-based process pool to avoid unsafe Linux forks from its background decision thread.

For targeted collection recovery, `--slots` freezes selected positions in the original schedule and `--attempt-offset` preserves earlier failed-attempt identities. Recovery preparation requires a sibling `.provenance.json` identifying the parent outputs. Selected runs remain diagnostic inputs until an explicit merge validates all original slots, source provenance, opponent proportions, and both perspectives; they cannot independently satisfy the 1,000-game training gate.

Foul Play's public set-inference adapter replaces an unbounded move-proposal loop with the equivalent conditional first-success draw and terminates when remaining probabilities are zero. A source hash guards this targeted runtime adapter; upstream files and MCTS selection remain unchanged. Strong-opponent comparisons bind the source, cached priors, and search budget to their reports.

The ordinary and joint configurations share a 60/40 published/fresh trajectory mixture, native primary objective, sampling schedule, learning rate 2e-5, 500-update warmup, sequence length 128, effective batch 12, and clipping at 1.5. Joint normalized marginal and autoregressive losses each weigh 0.01, with outcome loss at 0.1. The accumulated auxiliary backbone gradient is capped at 25% of the accumulated primary norm, including cancellation between microbatches. Outcome and belief additions are tested together; a belief-specific claim requires an outcome-only ablation.

Observed outcomes supervise the value of the recorded behavior and team distribution; they are not proofs of optimal play. Gradient norm ratios and alignment are recorded to diagnose whether the auxiliary objectives influence or conflict with primary training.

## Frozen private workflow

Run each command from the repository root using the appropriate isolated Python environment:

```text
python -m fog.scripts.pretrained_experiment prepare --help
python -m fog.scripts.pretrained_experiment submit RUN_ID
python -m fog.scripts.pretrained_experiment status RUN_ID
python -m fog.scripts.pretrained_experiment collect RUN_ID
```

Preparation seals exact code, upstream sources/licenses, inputs, configuration, environment requirements, and checkpoint hashes in `runs/experiments/RUN_ID`. Published archives and weights are downloaded separately. Native Metamon uses its pinned poke-env fork; FOG collection uses modern poke-env; Foul Play has a separate GPL process and Rust engine build. Frozen native source paths are explicit because editable import hooks differ across notebook images.

Before GPU packaging, require at least 1,000 valid fresh games, the exact opponent mixture, screened human pairs, an original-roster isolation audit (`pretrained_data_audit`), and a sealed CPU probability reference. `pretrained_merge` checks returned shard hashes, source identity, seat balance, and chronological reconstruction of both perspectives. The notebook requires CPU/CUDA and dense/cached probability agreement within 1e-3 before training and on the returned final checkpoint. Both opposite-condition packages must be sealed with identical source/data inventories and optimization settings before either training job launches.

The persistent GPU ledger charges full requested time, including failed runs: 2 hours for paired pilots, 8 for main runs, and 8 for refinement. Main/refinement update counts use the slower measured pilot with export overhead reserved. A fresh quota check is required at submission. Save files include model/target buffers, optimizer moments, schedule counters, sampler state, and Python/NumPy/Torch/CUDA RNG state. Two recovery states, shared quarter-update milestones, and the latest native export bound output size. Each native milestone has a small hash-matched auxiliary-head export. Downloaded results are hash checked before local analysis. `pretrained_compare` verifies equal completed updates and identical data schedules before comparing development play or proposing the next update budget.

`pretrained_belief_benchmark` compares learned samples with the existing FOG donor/prior inference code using submitted training-pool sets as donors and exact-team development games as targets. It reports unordered joint-set accuracy, recall, and empirical marginal calibration at a fixed sample count. Unverified labels are excluded. These Monte Carlo metrics do not turn route probability traces into exact unordered-set likelihoods. A subsequent downstream comparison must keep search settings fixed before learned sampling is connected to play.

## Evaluation and deployment

Use `pretrained_evaluate` on development teams to select candidates. Final confirmation requires at least 1,024 crossed seat/team games against unchanged Kakuna and a matched strong-opponent regression comparison. `pretrained_gate` also requires the selected finalist's own hash-matched CPU/CUDA probability report. It combines the play reports using four-game block intervals, alpha 0.025 per planned confirmation attempt, and a predeclared five-percentage-point strong-opponent regression margin. The persistent registry prevents spending either final cohort on a different model or changing its protocol during resume. Integration success never promotes a model.

`pretrained_showdown` provides a challenge/ladder entrypoint through modern poke-env and the native decision process. Credentials use `FOG_SHOWDOWN_USERNAME` and `FOG_SHOWDOWN_PASSWORD`, outside Git. It enforces a five-second inference deadline and records failures and legal fallbacks. A rejected or lost temporal decision switches that game to fallbacks until reset. Public rated confirmation requires a passed, hash-matched strength gate and at least 200 games; the report retains recent Elo, service GXE/Glicko snapshots, win-rate uncertainty, and failures. Its profile API follows [Showdown's published web API](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md).

There is no demonstrated native fine-tuning gain yet. A failed 18-hour attempt means this recipe and budget did not establish improvement. The next experiment would initialize FOG's typed-token/action-query model independently, use mechanics-derived public action features without teacher-answer features, and train on strong public-information teachers with submitted-team supervision. Recovery or that first pilot uses the remaining weekly compute only after a fresh quota check.

See [upstream attribution and licenses](../integrations/THIRD_PARTY.md).
