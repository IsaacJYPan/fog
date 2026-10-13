# Independent teacher pilot

The next pilot starts FOG's typed-token/action-query agent from independent
neural weights. It retains the autoregressive belief decoder, trains policy,
belief and observed-outcome objectives through shared parameters, and uses
mechanics-derived action features with every teacher-answer column erased.
Native Kakuna and earlier FOG checkpoints remain available as baselines.

## Verified data

The native collection preserves submitted teams, simulator seeds, actions and
public logs. Replaying accepted actions through the authoritative Showdown
simulator recovers each private request and FOG's ordinary player-visible
encoding. A shard is accepted only when the public trace (excluding clock
timestamps), winner, turn count, decision counts and native legality masks
match. Opponent submitted sets are separate labels. Only the acting player's
own submitted set enters its private observation.

The first export has 999 training games, 64,653 decision rows and 56,986
strong-teacher policy labels. It includes 32 development games with 1,990
decision rows. Both perspectives stay in the same shard and split. Historical
learner decisions do not supervise the policy. One source game was excluded
because its original collection corrected a rejected request; that correction
requires its own verified reconstruction. Train/development team compositions
are disjoint, and final confirmation teams remain outside this pilot.

`fog.scripts.export_teacher_replays` records source and shard hashes. The
portable loader checks those hashes, feature sanitization, action legality,
provenance and row counts before training. The small local trial also checks
policy input isolation, finite shared gradients and exact model, optimizer,
sampler and RNG resume. Dataset integrity and a successful trial do not
establish playing strength.

## Fixed training and evaluation

The pilot uses the existing 5.87-million-parameter typed-token configuration.
Each effective batch draws 256 strong-teacher decisions from eight sampled
games, accumulated in microbatches of 32. AdamW starts at 3e-4 with 500 warmup
updates and a fixed cosine schedule. Policy loss has weight 1; marginal and
autoregressive belief losses have weight 0.1 each, and outcome loss has weight
0.25. Shared auxiliary gradients are capped at 25% of the summed policy gradient
norm per update; auxiliary head gradients retain their full supervision. Global
gradient clipping is 1.5. A norm cap does not bound Adam's parameter displacement
or guarantee useful hidden-state representations.

Training has a 20,000-update ceiling and a four-hour training-time ceiling,
inside a 4.5-hour GPU notebook cap. Draws repeat existing decisions and must not
be described as unique data. Complete training states save every 200 updates,
with weight anchors every 1,000 updates and fixed development scores every 500.
CPU/CUDA policy probabilities must agree within 1e-3 before and after training.

A separate CPU notebook verifies the exact GPU export before running 256
seat/team-balanced games against each of SimpleHeuristics and MaxDamage, plus
128 against unchanged Kakuna. Both neural policies use temperature zero and
the fixed original preview order in this screen. These settings differ from
the earlier native continuation benchmark, so its scores are not a paired
comparison with this pilot. Decisions have a five-second deadline; failures,
fallbacks and unfinished games are explicit. The screen is developmental,
without final-cohort confirmation or automatic model promotion.

The scheduler and artifact handoff are bound to the fixed training identity.
Each changed recipe receives new packages. Returned states, metrics, protocol
reports and logs are downloaded and reviewed before choosing more training.
Track direct marginals, correct-prefix predictions and ancestral sampling
separately, and diagnose state-distribution shift if teacher fitting does not
translate into play. Further training is justified by that evidence, rather
than assuming every flat win rate is a compute shortage.
