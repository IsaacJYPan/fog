# FOG

*Playing the fog of war in both directions.*

A small (≤5M param) battle agent for Pokémon Showdown **Gen 9 OU singles** built around three
separate information sources: the acting player's private team, the current public board, and the
ordered public turn history. Policy, value, opponent belief, and self-belief all consume the three
sources from the appropriate legal perspective.

The current design and implementation boundary are tracked in [ARCHITECTURE.md](ARCHITECTURE.md).
The prioritized cleanup/removal plan is [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md). In particular,
two-sided public-belief search exists as an experimental engine-backed path; the live player still
wires only the older determinized search, so the equilibrium solver is not yet a production
inference claim.

Two research arms share one agent and one evaluation harness:

1. **Belief-space search + distillation (headline track).** The agent carries a learned
   **autoregressive belief model** over the opponent's hidden team and a **self-belief model**
   (the opponent's view of *us*). At decision time, **joint counterfactual search** optimizes the
   K private states the opponent thinks we could be in *together*, coupled through the belief
   update our action would induce — so bluffing and information cost are priced situationally.
   Search play is then **distilled** back into the fast policy (sampled/particle ReBeL for a
   simultaneous-move game whose private state space is too large for exact public belief states).
2. **Explicit information-leakage penalty (baseline arm, M3).** A train-time scalar penalty
   β·ΔI on measured information leakage. Nobody has the comparison "explicit information
   regularization vs implicit concealment from belief-consistent search distillation" — M3 exists
   to make that comparison.

**Design constraints:** no LLM inference during battles; search never runs inside the PPO gradient
loop (search enters training through logged data and distillation, never through the on-policy
rollout); every phase produces a usable artifact even if the project stops there.

## Simulator paths, one observation protocol

- **Direct pinned Pokémon Showdown** (`env.backend: showdown_direct`, the default) — the Node
  simulator runs in one persistent local Node worker per rollout process, with no server, accounts,
  HTTP, or WebSockets.
  Showdown's own player-censored messages and requests still pass through poke-env's Battle parser,
  so this changes transport rather than what the policy can observe. The worker also contains the
  exact controlled-replay chance primitive. The acceptance gates below block search
  data used for reported results whenever candidate-state or action-grid fidelity is unresolved.
- **poke-env + a Showdown server** (`env.backend: showdown`) — retained for external/live-play
  compatibility and differential validation.
- **poke-engine (Rust, in-process)** (`env.backend: engine`) — the historical volume backend:
  synchronous,
  so N battles step in lockstep and every decision wave is ONE batched forward. Measured
  ~335–350 decisions/sec vs ~25 on the Showdown path (13–14×). Training on it keeps
  `cross_sim.every_decisions > 0`: the same agent is periodically evaluated on BOTH backends and
  training HALTS if the win-rate gap exceeds 3% (an agent that farms engine-only mechanics has
  learned the engine, not Pokémon). Its known mechanics differences mean it is not the authority
  for new branch data used for reported results.

### Direct rollout parallelism

`showdown_direct` uses stable spawned worker processes when `env.rollout_workers > 1`. Each process
keeps its CPU actor, observation state, bounded frozen-opponent cache, and Node simulator alive
across collection chunks. The centralized learner owns the only CUDA model. A shared-memory CPU
actor is refreshed in place only after a PPO update, while all workers are idle, so every game in a
batch uses one coherent policy version without copying or rebuilding the model every 32 games.

Use one PyTorch inference thread per process first (`env.actor_threads: 1`) and benchmark
`rollout_workers: 4`, `6`, and `8`; more workers are accepted only when measured decisions/sec
improves and CPU temperature/RAM remain safe. `env.rollout_worker_mode: thread` retains the old
debug fallback, but it shares Python's GIL and rebuilds actors per chunk. `env.node_binary` can pin
an explicit Node executable for managed or portable environments. Worker identity, game tags, and
policy sampling seeds are deterministic; a worker exception terminates the pool and propagates its
remote traceback to the learner, and normal exit closes every Node child.

`league.latest_frac`, `league.past_frac`, and `league.scripted_frac` configure the opponent mixture
and must sum to one. A continuation that intends to train against frozen self-play must also seed
`league.past_ckpts`; without a past checkpoint, the safety fallback samples only scripted
opponents. Repeating a scripted name is an explicit weight within the scripted fraction.

## Layout

```
  server/            # pinned pokemon-showdown checkout — NOT vendored here
  data/              # team pools (pastes tracked: experiment definitions); replays/usage gitignored
  fog/
    constants/       # generated lookup tables pinned to the simulator commit + usage month
    env/             # direct Showdown worker/client, poke-env views, Rust backend, teams
    encoding/        # censored battle state -> tensors (reveal ledger, tracker, mirror)
    models/          # three encoders, belief + self-belief, policy/value, oracles
    training/        # BC, self-play PPO, leakage objective, value fine-tune, distill, belief refresh
    search/          # determinized baselines + experimental two-sided public-belief search
    eval/            # gauntlets (incl. search arms), Elo, oracle gaps, belief/leakage metrics
    scripts/         # pipeline one-liners, scrapers, parsers, pool builders
  tests/             # unit + integration (engine-marked tests skip without poke-engine)
  configs/           # yaml per experiment; every run reproducible from config + seed
```

`ARCHITECTURE.md` is the maintained architecture specification. Historical experiment notes may
remain local, but the portable design must not depend on them. A change to belief, search, value,
simulation, or distillation must update the architecture, the technical-debt register when relevant,
and its acceptance tests—not only the code.

## Theoretical design and invariants

### Information available to each component

At turn *t*, `h_t` is the public battle history: everything both players can legally observe.
`s_i` is player *i*'s private team state, including its unrevealed sets. The acting policy and its
opponent-belief model may use `h_t` and its own `s_i`; neither may ever receive the opponent's true
`s_-i`. The observer's own item, nature, and EVs are required to interpret damage, speed order, and
action constraints as evidence about the opponent. The self-belief model runs from the opponent's
mirrored legal view and uses that view's own private state—not ours—to estimate what the opponent
can infer about us. Oracle truth is confined to labeled evaluation arms and supervised targets; an
ordinary forward pass must remain unchanged when inaccessible opponent truth changes.

For candidate hidden state `s`, the public-belief update after actions `a_t` and public observation
`o_{t+1}` is

```
b_{t+1}(s) proportional to
    b_t(s) * policy_probability(a_t | h_t, s)
           * observation_probability(o_{t+1} | h_t, s, a_t).
```

The first factor is the prior belief. The second is present because an action is evidence about the
private state from which it was selected. The third is present because damage, activation messages,
switch results, and other public outcomes can be more likely in some hidden states than others.
The distribution is normalized only after multiplying all three factors. Impossible states receive
zero probability. Publicly known current facts override a candidate's original team-paste facts:
for example, search may not resurrect an item that was consumed, removed, or exchanged.

### Three-part observation backbone

The observation backbone has three explicit encoders rather than one mixed board/history trunk:

1. **Current-state encoder.** A fixed-width public snapshot of both teams and the field. Categorical
   species/item/ability/move identities use embeddings; per-Pokémon MLPs and pooling summarize the
   twelve records. There is no transformer here. Before this path runs, the acting player's private
   moves, item, ability, nature, EVs, PP, pre-Tera type, original item/ability, and own-knowledge
   flags are removed. Opponent records already contain only reveal-ledger facts.
2. **Turn-history encoder.** The completed public turn records, their categorical sidecars, and
   start-of-turn boost states receive learned position embeddings and a two-layer causal
   transformer. Its upper-triangular attention mask prevents every intermediate turn token from
   attending to a later turn. Front padding is repacked and masked before attention.
3. **Private-state encoder.** The acting player's six complete submitted sets, private PP, and Tera
   information use a separate per-Pokémon MLP and pooled/active summaries. It never reads opponent
   slots.

The current and history vectors fuse into one public context. Policy/value receive that public
context plus the separate private vector. Each belief consumer receives the public context plus a
learned projection of the private vector belonging to that same observer. Thus our opponent belief
may use our known set to interpret battle evidence, while self-belief uses the opponent-view private
vector and cannot see our private tensor. Each self-play side constructs these tensors independently
from its own perspective, preventing cross-player information leakage.

Species/item/ability/move/nature/EV embedding tables are shared for statistical efficiency, but the
three projections, sequence model, and output vectors are distinct. Their explicit output names are
`private_state`, `public_state`, and `history`; `public_context` refers only to the derived fusion of
`public_state` and `history`. Full training leaves the three encoders trainable;
`model.aux_into_trunk: true` lets
belief losses train the public encoders/fusion and the observer-private evidence path.

### Order-independent complete-team model

The belief model — the network that estimates a hidden team — is set-autoregressive. Every revealed
fact is available before the first hidden prediction, and every generated fact is added to the same
unordered context. Facts are bound within each Pokémon before the six Pokémon are mixed. Therefore
reordering reveal storage, team-sheet rows, or a Pokémon's move list cannot change the represented
distribution except for the matching output-row permutation.

A sequential generator still needs a generation route; “order independent” does not mean that a
sample appears without steps. The next-field scheduler is independent of field values and chooses
uniformly among mechanically eligible fields. Species precedes species-dependent legality, and a
nature and each EV field wait for species; no other global semantic order is imposed. Training
samples a route and a partial cut, then predicts the next field from exactly the kind of partially
completed set that sampling sees. This prevents the earlier mismatch between full teacher forcing
and progressive sampling.

For route `r` and completed team `x`, the model records

```text
P(x, r | public evidence)
  = product of scheduler probabilities
    × product of field-value probabilities.
```

The probability of unordered team `x` is the sum over all legal routes. Exact enumeration is
combinatorial, so production candidate scoring estimates that sum by evaluating several sampled
routes and taking their log-mean-exp. Hidden move sequences are sampled uniformly and receive the
factorial correction required to estimate the sum over their permutations. The route count and
residual order sensitivity are measured; one canonical route is retained only as a diagnostic.
Hidden move truths are randomly permuted during training and sampled as distinct set insertions.

### Native nature and EV representation

Complete candidate teams use a `[6, 15]` integer grid:

```text
species, item, ability, move0..move3,
nature, ev_hp, ev_atk, ev_def, ev_spa, ev_spd, ev_spe, tera
```

Nature is a 25-way categorical target. Each EV field is a shared 64-way categorical target in the
units the battle stat formula consumes: `floor(EV / 4)`, values 0 through 63. A decoded world
multiplies those units by four. The sampler dynamically enforces a maximum total of 127 units
(508 effective EVs) for each PokÃ©mon; Showdown's two remaining nominal EV points cannot change a
stat. This replaces the old autoregressive spread-cluster target. Usage-derived spread clusters
remain only as an offline prior for generating realistic team-pool pastes.

Public replay records do not reveal submitted nature or EVs, so those labels stay censored rather
than being guessed. Native heads are trained from complete team pastes/self-play truth. Legacy
9-column replay shards are migrated on load: public species/item/ability/moves/Tera are preserved,
the obsolete cluster id is discarded, and native fields remain unknown. Compatible checkpoint
loading reuses shape-matched policy, history, and backbone tensors. Nature plus six EV embeddings
are compressed to the old stat-block width. During a three-encoder warm start, the learned legacy
entity projection seeds both the public-current and private entity projections; categorical history
inputs, policy/value, belief heads, and shared embeddings are also retained when shapes match. The
new causal sequence layers, public fusion, and encoder output heads initialize fresh. A later
private-to-belief projection is zero-initialized so a compatible warm start exactly preserves the
old public-only belief at step zero, then learns how the observer's set contextualizes public
evidence. With `model.aux_into_trunk: true`, native belief loss trains the public
current/history encoders, their fusion, and this observer-private path; policy/value loss trains all
three encoders.

### Search target

The target expert is a two-sided public-belief search, not ordinary determinization. Ordinary
determinization samples an opponent team and optimizes as if that sample were true; this can choose
actions that are individually attractive but collectively inconsistent with what the opponent
believes. The target instead solves all supported private states together. An action is therefore
valued both for its battle effect and for how its publicly visible consequences change each
player's beliefs. This coupling is what permits situational concealment and bluffing without a
fixed hand-designed information penalty.

**Current implementation status (2026-08-06).** `fog.search.pb_search` implements the experimental
two-sided private-team grid and regret-matching solve on the engine-backed path. It expands supported
teams for both players and returns the mixed strategy row for the real private state. This is the
path intended to prevent strategy fusion. It is not wired into `FogPlayer`, whose optional live
search still uses one-ply determinization. A version used for reported results must use serverless
direct Showdown transitions, evaluate both players from their own censored views, and pass the stated
search-on/off and mechanics gates before its targets are distilled.

The small 2026-07-30 trace explains this binding decision. A one-ply search using a fixed
MaxDamage reply won both diagnostic games against MaxDamage but lost both against
SimpleHeuristics. Its mechanics were action-dependent and mostly reconstructed successfully, so
the result was not a no-op search; the reply-model mismatch was material. Search should not be
distilled or used to generate policy targets again until the symmetric direct-Showdown solver
passes a small held-out trace and paired search-on/search-off gate.

Pokémon turns contain simultaneous choices and random outcomes. The solver forms the complete
legal action grid for both players and enumerates the simulator's finite chance branches with their
native probabilities. "Exact" means following those returned probability branches; it does not
mean trying random seeds until a different result appears. Chance pruning is disabled by default.
If an experiment enables pruning, it must report discarded probability mass and show that the
decision is stable to removing the approximation.

The search expert is outside the policy-gradient loop. It generates decision distributions and
outcome targets, which are distilled into the deployment policy. After policy distillation, both
belief models are refreshed on new play because the policy is part of the evidence likelihood
above. The fast deployed policy remains a single network forward pass; search is an optional
inference-time arm and a training-data producer.

The value network caches public-leaf and candidate-team representations once. When solver
probabilities change, only the belief-dependent message passing and pair-value head rerun. This is
exact reuse for the declared architecture; it does not assume that one cached scalar can be
combined with policy probabilities to recover every nonlinear hidden-matchup value.

Selective depth two is available as a correctness scaffold. It solves depth one, ranks public
leaves by actual reach probability, expands a bounded number of likely child public-belief
subgames, returns their pair-specific continuation values, and solves the root again. The scalable
target is probability-guided outcome-sampling public-belief regret search: likely public histories
are revisited often, while sampling corrections preserve the underlying game probabilities.

### Simulator decision

Pinned Pokémon Showdown is authoritative for mechanics, censored observations, final evaluation,
and targeted exact replay. Direct Showdown needs no server or WebSocket. For the next search
implementation, direct in-memory Showdown is also the required transition backend for every
search configuration used for reported results. The corrected Gen 9 poke-engine build remains
useful for diagnostics, fuzzing, and explicitly labeled approximations, but it must not replace
direct Showdown in the symmetric search expert. Any future performance optimization must first
demonstrate action/legality, censored-observation, chance-mass, and leaf-value parity against the
direct path.
Seeds make repeated runs reproducible; they do not define branch probabilities. Headline strength
is always measured in Showdown.

### Binding rejection rules

An experienced implementation should preserve negative results instead of promoting a component
because its training loss decreased. The current records include two such findings:

- The learned public-belief value network fit its training target, but a matched public-only model
  had lower held-out error. It therefore failed its binding ablation and is not integrated.
- Two exact solver/distillation generations improved their within-generation imitation losses, but
  the second became worse on the fixed first-generation reference and substantially weaker in
  direct Showdown play. Generation 1 is rejected. Generation 0 remains a candidate, not a champion,
  until the interrupted matched evaluation is completed. The retained checkpoint is the input to
  generation 0.

These decisions may be reversed only by a predeclared evaluation on fresh held-out games. Required
checks include legal-information isolation, probability-mass conservation, deterministic replay,
belief responsiveness, permutation invariance, comparison with a public-only value model, matched
team/side battle schedules, failure and truncation counts, and at least 1,000 Showdown games for a
reported win-rate result.

## Stack

Python 3.11+ · PyTorch (CUDA for research runs) · [poke-env](https://github.com/hsahovic/poke-env) ·
poke-engine (Rust) · a pinned Pokémon Showdown checkout/build · Node.js · tensorboard.

## Getting started

```bash
# Reproduce the locked development environment (uv.lock is authoritative):
uv sync --extra dev

# Build the pinned local simulator once. Direct-backend runs need the build, not a running server:
cd server/pokemon-showdown && npm ci && npm run build

# Only external/live-play compatibility needs a server:
node pokemon-showdown start --no-security   # port 8000

# (only if re-pinning) regenerate constants + usage tables:
python -m fog.scripts.build_constants --showdown server/pokemon-showdown
python -m fog.scripts.download_usage  --month 2026-06 --rating 1695
python -m fog.scripts.build_team_pools --pool all
python -m fog.scripts.build_generative_pool --n 128 --seed 0
```

## Running things

Use the unified entry point for one PPO or behavior-cloning run:

```bash
python -m fog.train --config configs/set_ar_replay.yaml  # infers behavior cloning
python -m fog.train --config configs/set_ar_train.yaml   # infers PPO
python -m fog.train --config configs/set_ar_train.yaml --set ppo.lr=0.00003
```

`fog.train.TrainingSession` exposes the same resolved configuration as a mutable Python object for
custom scripts and notebooks. `fog.scripts.automated_training` is the separate long-run block,
evaluation, retention, and recovery orchestrator documented in `KAGGLE_TRAINING.md`.

Historical milestone recipes now live under `configs/archive/`. The former
`fog.scripts.pipeline` commands are not supported training entry points; use `fog.train` or the
automation driver above.

Current safety-gated training runs and evaluation:

```bash
# Mandatory mechanical gate before either full set-AR recipe. This runs five complete games on
# authoritative direct Showdown, writes per-decision JSONL under runs/preflight_audit/, and takes
# zero optimizer steps. set_ar_replay.yaml and set_ar_train.yaml refuse to start without a pass.
python -m fog.scripts.preflight_audit --games 5

python -m fog.train --config configs/set_ar_replay_smoke.yaml # optional censored-replay check
python -m fog.train --config configs/set_ar_replay.yaml       # optional belief-only replay arm
python -m fog.train --config configs/set_ar_smoke.yaml        # two-update compatibility check
python -m fog.train --config configs/set_ar_train.yaml        # set-AR warm start from retained G0
python -m fog.train --config configs/archive/m1_poc.yaml  # archived M1 reproduction only
python -m fog.train --config configs/archive/m2_oracle.yaml # archived oracle ceiling only
python -m fog.train --config configs/archive/m3_leakage_engine.yaml --set leakage.beta=0.03 ...
python -m fog.train --config configs/bc_ou.yaml            # M4 behavior cloning
python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt CKPT [--lvn N.pt|--readability|--pareto]
```

### Public full-team bank for open-OU self-play

Public battle replays are not complete team sheets: anything never revealed in battle remains
unknown. `collect_team_bank` therefore collects only exact imports that their authors explicitly
published through Pokepaste. It crawls a small allowlist of Smogon resource/tournament threads and
high-ladder RMT index pages, uses Pokepaste's public `/raw` representation, and then requires every
team to:

1. contain six explicit Pokemon blocks rather than a description page,
2. parse canonically through poke-env,
3. be semantically unique after packing, and
4. pass the pinned current Showdown `gen9ou` validator.

Every accepted paste retains its source, public URL, claimed ladder evidence, replay links, quality
score, and content hashes in `manifest.json`. Tournament/resource pastes and automatically screened
RMT candidates remain labeled separately so an evaluation pool can receive a final manual review.
HTTP responses are cached, requests are rate-limited, and collection is resumable from the cache.

```bash
# Build up to 150 current, legal, unique teams. Older illegal tournament pastes are rejected.
python -m fog.scripts.collect_team_bank --target 150 --min-quality 70

# Inspect data/teams/public_gen9ou/manifest.json before training. The packaged pool is already
# compatible with fog.env.teams.load_pool.
python -m fog.train --config configs/train_ou_public.yaml
```

Source discovery is configured in `configs/team_sources_gen9ou.yaml`. RMT titles must advertise at
least 1850 Elo, a top-200 rank, at least 80% GXE, or OLT qualification even to enter the queue.
Those claims are evidence signals, not independently verified facts; for final evaluation, prefer
the manifest's direct tournament/resource teams and keep the RMT-derived bank as training diversity
unless its screenshots and replay sample have been reviewed.
`train_ou_public.yaml` refuses to begin unless the packaged bank contains at least 100 teams, so a
small smoke collection cannot accidentally become a long self-play run.

For representative current-ladder diversity, `configs/team_sources_gen9ou_recent.yaml` separately
crawls only the first six recent forum pages with a 1600-Elo discovery floor. These remain labeled
recent-forum candidates even after the pinned current validator accepts them. Merge validated banks
with `fog.scripts.build_public_team_splits`; it deduplicates by exact packed-team hash and creates
source-stratified train/validation/test packages so no exact roster crosses evaluation boundaries.
The `gen9ou_public_v1` training package is sampled uniformly, placing a hard empirical ceiling on
any one roster's influence rather than heavily oversampling a small "current" subset.

The current preflight schema is v2. In addition to the checkpoint-compatible 108-float turn
record, it verifies a categorical history sidecar (active/target species, moves, switches,
item/ability/Tera identities, first actor, and pre/post-move switch phase), the individual boost
stages of both start-of-turn actives, and immutable own-sheet item/ability ids kept separately from
their mutable battle-state values. Changing this contract requires a fresh five-game certificate;
an older `latest.json` cannot authorize the set-AR smoke or full recipe. The certificate also
records a content fingerprint of the working Python/config/generated-table tree and the local
Showdown mechanics sources, so changing uncommitted code invalidates the gate just like changing a
commit.

Both `set_ar_smoke.yaml` and `set_ar_train.yaml` explicitly select `showdown_direct`. They must not
inherit the retained G0 run's `engine` override: poke-engine remains an optional, differentially
tested volume approximation, while in-process Showdown is the mechanics authority.

## Historical experiment gates

The M-series table records earlier experiment decisions; it is not a list of all currently supported
training entry points. Current architectural acceptance rules live in `ARCHITECTURE.md`.

| Milestone / phase | Gate |
|---|---|
| **M0** plumbing | ≥3 battles/sec (measured 13.3) |
| **M1/M1.5** PoC + belief conditioning | vs MaxDamage ≥75%, vs SimpleHeuristics ≥60%; pool-id ≥80% top-1 by turn 4; **b_vec ≥3% win-rate gain over M1 — the go/no-go for both research arms** |
| **M2** set-oracle ceiling | oracle ≥ N-agent; record the gap |
| **M3** leakage grid | β grid on POOL-A + pool_gen, ≥3 seeds/cell; readability + win-rate per cell |
| **SEARCH 4.2** posterior-aware value | aware < blind held-out MSE (**passed 2026-07-11**: 0.4730 vs 0.4901) |
| **SEARCH Phase 2** joint-search validation | joint > determinized AND joint > policy vs a reading opponent / head-to-head, Wilson CIs |
| **SEARCH Phase 3** sampled scalability | exact-vs-sampled decision agreement + gauntlet delta at K ∈ {8..64} |
| **SEARCH Phase 4** distillation | held-out KL(search‖π) shrinking across generations; readability vs the M3 β frontier at matched strength |
| **M4** open OU (BC → self-play) | BC top-1 ≥45%; post-self-play ≥80% vs SimpleHeuristics |
| **M5** final eval | full gauntlet/Elo/closure on Showdown |

## Reproducibility

`uv.lock` pins the Python dependency graph, including poke-engine 0.0.48 compiled with its Gen 9
Cargo feature and without older-generation defaults. CUDA wheels are
platform-specific; a research run that replaces the locked CPU PyTorch wheel must record the exact
PyTorch and CUDA build in its run manifest. Every run also gets a config hash + seed. The Showdown
commit and usage-month snapshot are pinned in config. Direct-backend probabilities use the exact
rational chance requested by Showdown's battle rules; seeded runs validate the distribution rather
than discovering branches. A golden-battle integration test must produce byte-identical tensors
across refactors. Engine-backend runs keep the cross-sim divergence control
on; headline numbers are Showdown-only.

Repository validation on this machine:

```bash
.venv/Scripts/python -m ruff check fog tests
.venv/Scripts/python -m pytest -q
```

The full first-party Ruff gate and the complete test suite are clean. Expected xfails document
deliberately unsupported mechanics or research boundaries and are not ordinary test failures.

Never run these checks, or any second PyTorch process, beside a Showdown training run on the
16 GB development machine.

## Continue on another machine

The portable checkpoints live under `artifacts/checkpoints/` and are Git LFS objects. Install Git
LFS before cloning or run `git lfs pull` after cloning. `manifest.json` records SHA-256 hashes,
provenance, and promotion status. Do not continue from `gen1_rejected.pt`; use
`g0_retained.pt` for a fresh larger generation, or finish the matched evaluation before deciding
whether `gen0_candidate.pt` deserves promotion. The active typed-token autonomous job uses
`typed_dagger_2048.pt` as its initial learner and the four `typed_dagger_*.pt` files as its bounded
historical league; these inputs are also tracked through LFS rather than depending on local `runs/`.

```bash
git submodule update --init --recursive
git lfs pull
uv sync --extra dev
cd server/pokemon-showdown && npm ci && npm run build && cd ../..

# Verify the transferred checkpoints before spending compute.
git lfs fsck

# Verify the redesigned belief path and inspect runs/set_ar_smoke/init_compatibility.json before
# spending a long run. Matching G0 tensors are reused; new set-encoder tensors initialize fresh.
uv run python -m fog.train --config configs/set_ar_smoke.yaml

# Step through both censored views for five complete authoritative games. The full replay/PPO
# recipes below are hard-gated on runs/preflight_audit/latest.json.
uv run python -m fog.scripts.preflight_audit --games 5

# Optional replay auxiliary. Replays label only facts eventually revealed, so this updates only the
# opponent belief model and does not replace complete-truth self-play. Run it as a separate arm;
# promote it only if complete-team likelihood and downstream search beat the no-replay arm.
uv run python -m fog.train --config configs/set_ar_replay.yaml

# Start the longer set-autoregressive run. It saves weight snapshots every 20k decisions.
uv run python -m fog.train --config configs/set_ar_train.yaml

# Complete the binding promotion test first. The report is saved every four games; rerun the
# identical command with --resume after an interruption.
uv run python -m fog.eval.showdown_direct_selection \
  --checkpoint g0=artifacts/checkpoints/g0_retained.pt \
  --checkpoint gen0=artifacts/checkpoints/gen0_candidate.pt \
  --pool data/teams/pool_a --head-to-head --games 1000 --seed 2107 \
  --device cuda --out runs/g0_vs_gen0_showdown_1000.json

# Experimental engine-backed two-sided generation from the retained checkpoint. Keep pruning
# disabled; this is not an authoritative direct-Showdown search claim.
uv run python -m fog.scripts.pb_loop \
  --ckpt artifacts/checkpoints/g0_retained.pt \
  --config configs/default.yaml --pool data/teams/pool_a \
  --out runs/pb_loop_large --generations 1 --games 400 --workers 8 \
  --iters 128 --min-prob 0 --top-j 8 --policy-init \
  --distill-epochs 20 --refresh-games 1024 --device cuda
```

The exact worker count should be set from available RAM, not just GPU size. Each worker holds its
own simulator/search state. Start with four workers, measure peak memory, then increase to eight
only if the machine retains safe headroom. Do not run tests or another PyTorch job concurrently.
An interrupted PPO run can warm-resume from its newest `runs/set_ar_train/checkpoints/ppo_*.pt` by
setting that file as `ppo.init_ckpt` and setting `ppo.total_decisions` to the remaining budget.
This restores weights, not optimizer moments or the exact random stream; preserve the previous
checkpoint as a league opponent and record the segment boundary.
