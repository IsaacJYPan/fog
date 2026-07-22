# FOG

*Playing the fog of war in both directions.*

A small (≤5M param) battle agent for Pokémon Showdown **Gen 9 OU singles** built around
hidden-information play: an edge comes from reading what the opponent conceals while controlling
what your own play reveals. Two research arms share one agent and one evaluation harness:

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
  simulator runs in one persistent local worker with no server, accounts, HTTP, or WebSockets.
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

## Layout

```
  server/            # pinned pokemon-showdown checkout — NOT vendored here
  data/              # team pools (pastes tracked: experiment definitions); replays/usage gitignored
  fog/
    constants/       # generated lookup tables pinned to the simulator commit + usage month
    env/             # direct Showdown worker/client, poke-env views, Rust backend, teams
    encoding/        # censored battle state -> tensors (reveal ledger, tracker, mirror)
    models/          # backbone, belief + self-belief, policy/value, oracles
    training/        # BC, self-play PPO, leakage objective, value fine-tune, distill, belief refresh
    search/          # determinized + joint counterfactual search, own-state sources, posterior log
    eval/            # gauntlets (incl. search arms), Elo, oracle gaps, belief/leakage metrics
    scripts/         # pipeline one-liners, scrapers, parsers, pool builders
  tests/             # unit + integration (engine-marked tests skip without poke-engine)
  configs/           # yaml per experiment; every run reproducible from config + seed
```

Architecture notes, experiment notes, issue notes, and implementation plans are maintained
locally. This README is the only tracked project document.

The full local decision record is `PROJECT_DESIGN_PROPOSAL.md`; the tracked invariants below are
the portable minimum needed to prevent design drift. A change to belief, search, value, simulation,
or distillation must update both the reasoning and its acceptance tests, not only the code.

## Theoretical design and invariants

### Information available to each component

At turn *t*, `h_t` is the public battle history: everything both players can legally observe.
`s_i` is player *i*'s private team state, including its unrevealed sets. The acting policy may use
`h_t` and its own `s_i`; it may never receive the opponent's true `s_-i`. The belief model estimates
the opponent's hidden state from the acting player's legal view. The self-belief model estimates
what the opponent can infer about the acting player's hidden state, using only information that
would be visible from the opponent's side. Oracle truth is confined to labeled evaluation arms and
supervised targets; an ordinary forward pass must remain unchanged when inaccessible truth changes.

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

### Search target

The target expert is a two-sided public-belief search, not ordinary determinization. Ordinary
determinization samples an opponent team and optimizes as if that sample were true; this can choose
actions that are individually attractive but collectively inconsistent with what the opponent
believes. The target instead solves all supported private states together. An action is therefore
valued both for its battle effect and for how its publicly visible consequences change each
player's beliefs. This coupling is what permits situational concealment and bluffing without a
fixed hand-designed information penalty.

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

### Simulator decision

Pinned Pokémon Showdown is authoritative for mechanics, censored observations, final evaluation,
and targeted exact replay. Direct Showdown needs no server or WebSocket. Full exact Showdown replay
is too slow for every search cell, so the corrected Gen 9 poke-engine build is retained for large
branch grids only after reconstruction, legality, chance-mass, and differential tests pass. Seeds
make repeated runs reproducible; they do not define branch probabilities. Headline strength is
always measured in Showdown.

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

`fog.scripts.pipeline` is the driver — one line per phase (add `--set key=value` overrides):

```bash
python -m fog.scripts.pipeline m15               # Phase 0: M1.5 belief-supervised PPO
python -m fog.scripts.pipeline gate-bvec         # THE go/no-go: M1.5-vs-M1 paired head-to-head
python -m fog.scripts.pipeline gate-belief       # M1.5 belief gate: pool-id top-1 by turn 4
python -m fog.scripts.pipeline m3-control        # M3 beta=0 control (engine backend)
python -m fog.scripts.pipeline m3-grid           # M3 beta grid x seeds (control must exist)
python -m fog.scripts.pipeline smoke-joint       # live joint-search smoke (no training)
python -m fog.scripts.pipeline log-posteriors    # search-play decision log (SEARCH_PLAN 4.2 / Phase-4 data)
python -m fog.scripts.pipeline finetune-value    # posterior-aware value fine-tune + gate
python -m fog.scripts.pipeline gauntlet-search   # Phase 2: policy/determinized/joint arm gauntlet
python -m fog.scripts.pipeline agreement         # Phase 3: exact-vs-sampled decision agreement
python -m fog.scripts.pipeline distill           # Phase 4: distill search play into the policy
python -m fog.scripts.pipeline refresh-beliefs   # Phase 4: belief consistency retrain
```

Milestone training runs (M-track) and evaluation:

```bash
python -m fog.training.ppo  --config configs/m1_poc.yaml          # M1 baseline (no belief vec)
python -m fog.training.ppo  --config configs/m2_oracle.yaml       # M2 set-oracle ceiling
python -m fog.training.ppo  --config configs/m3_leakage_engine.yaml --set leakage.beta=0.03 ...
python -m fog.training.bc   --config configs/bc_ou.yaml           # M4 behavior cloning
python -m fog.eval.run_eval --config configs/eval_ou.yaml --ckpt CKPT [--lvn N.pt|--readability|--pareto]
```

## Gates

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
.venv/Scripts/ruff check .
.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp runs/pytest-local
```

Never run these checks, or any second PyTorch process, beside a Showdown training run on the
16 GB development machine.

## Continue on another machine

The portable checkpoints live under `artifacts/checkpoints/` and are Git LFS objects. Install Git
LFS before cloning or run `git lfs pull` after cloning. `manifest.json` records SHA-256 hashes,
provenance, and promotion status. Do not continue from `gen1_rejected.pt`; use
`g0_retained.pt` for a fresh larger generation, or finish the matched evaluation before deciding
whether `gen0_candidate.pt` deserves promotion.

```bash
git submodule update --init --recursive
git lfs pull
uv sync --extra dev
cd server/pokemon-showdown && npm ci && npm run build && cd ../..

# Verify the transferred checkpoints before spending compute.
git lfs fsck

# Complete the binding promotion test first. The report is saved every four games; rerun the
# identical command with --resume after an interruption.
uv run python -m fog.eval.showdown_direct_selection \
  --checkpoint g0=artifacts/checkpoints/g0_retained.pt \
  --checkpoint gen0=artifacts/checkpoints/gen0_candidate.pt \
  --pool data/teams/pool_a --head-to-head --games 1000 --seed 2107 \
  --device cuda --out runs/g0_vs_gen0_showdown_1000.json

# Larger exact generation from the retained champion. Keep pruning disabled.
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
