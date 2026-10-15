# Backward curriculum for Gen9OU

The candidate is demonstration-backed starting-state selection: learn to finish
real games from late decision points, then move the starting point earlier.
Transitions and rewards still run forward under authoritative Showdown rules.
The agent must earn a new outcome after the reset. A historical winner is not a
valid target for a newly played continuation.

[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html)
adapts starts outward from known goals in simulated control tasks.
[Backplay](https://arxiv.org/abs/1807.06919) instead uses recorded demonstration
states and includes experiments in the stochastic multiplayer game Pommerman.
That formulation fits irreversible battle histories better than taking random
steps from a terminal Pokémon victory. These results motivate an experiment;
they do not establish that the approach will improve this bot.

[Jump-Start Reinforcement Learning](https://arxiv.org/abs/2204.02372) supplies
another implementation option: a guide policy plays a prefix before the learner
takes over. Its formal guarantees assume state/feature coverage by the guide and
appropriate exploration and policy-learning procedures. We have not verified
these assumptions for partially observed, simultaneous-action Gen9OU with neural
PPO and changing opponents. There is no optimal-play or convergence guarantee
for the proposed experiment.

## Feasibility implementation

`backplay_bank` freezes hash-bound raw records and observation shards from the
already verified exact-team replay export. Whole-game composition splits remain
intact. `reset_from_record` replays the accepted prefix with the original seed,
validates its complete public trace and native action legality, and reconstructs
each role's reveal ledger and past executed actions. No prefix neural actions or
PPO rows are generated. Each encoder sees its own submitted private team and
ordinary player-visible protocol only. Complete opponent teams are labels.

At the selected boundary, the coordinator reseeds future chance without changing
requests or consuming messages. The continuation uses current controller actions
and its actual new terminal result. Historical suffix actions, future reveals,
future seeds and recorded winners are excluded from model inputs and PPO targets.
Rejected or unverifiable source games cannot become reset positions.

`backplay_probe` compares the independent initial FOG weights, its own 1,000-update
behavior-cloning checkpoint, and SimpleHeuristics from matched roots. It checks
root observation/history equality against the verified shards, fresh outcomes,
belief-label availability and runtime. Both seats include the historical winning
and losing perspectives. Source games are the sampling units; related roots and
arms are not independent trials. These are continuation diagnostics, not ordinary
full-game win rates. An optional train-only local smoke update checks finite PPO,
belief, chain and value losses and nonzero shared gradients.

## Review before spending GPU hours

Late starts shorten credit assignment and produce more terminal feedback per
learner decision. They can also be trivial wins, hopeless losses, or positions
with only one legal action. If initial and trained controllers score similarly,
high reset success alone does not demonstrate learning. Inspect branching,
outcome variability, game diversity and paired controller differences before
choosing a window. Preserve losing positions and draw outcomes in diagnostics.

Training only on endgames changes the objective and provides little supervision
for opening play or hidden-team inference. A training pilot should therefore mix
at least 25% ordinary complete games from fresh team pairings with the curriculum,
retain exact-team marginal/AR belief and fresh outcome supervision, and advance
the window using development competence rather than its historical distance
alone. The same teams and historical endings cannot be the whole training set.
Additional independently generated guide trajectories are needed for scale.

The matched online pilot starts both conditions from the same independently
trained FOG 1,000-update cloning checkpoint. One uses ordinary starts; the other
adds backward starts from 999 verified training games, mixed with at least one
ordinary episode in four. Reset pairs use both seats of the same source game.
The pilot advances through 4/8/16/32/64 historical decisions from the end, after
at least ten batches at a level and sixteen historically winning training roots.
Advancement requires 70% new wins on those winning roots with a Wilson lower
bound of at least 50%. This is a train-only pilot gate; held-out full-game play
must validate the progression before it becomes a longer training recipe.

Both arms face the same fixed SimpleHeuristics opponent to isolate the starting
distribution while measuring the heuristic warm-start objective. This is not a
strong-opponent league or a competitive-strength confirmation. Sixty batches
use 512 newly collected positions each, two epochs and minibatches of 128: 480
optimizer steps per arm. Completed episodes supply GAE and fresh outcome labels
before a uniform subset is selected for each fixed-size update. Extra collected
positions are reported separately; the optimized-position and update budgets
match exactly. A small KL penalty anchors the initial policy. Belief, AR and
outcome gradients into shared parameters are jointly capped at 25% of the policy
gradient norm; auxiliary heads retain their full gradients.

The pilot uses a measured-throughput learning-rate schedule and periodic full
state saves, including sampler, pending reset pair, curriculum stage, optimizer
and CPU/CUDA RNG. A local real-data trial reproduced model, optimizer, sampler,
Torch RNG and curriculum state exactly across interruption. The cloud worker
must check CPU/CUDA probabilities before and after GPU training. GPU and CPU
evaluation remain separate saved jobs.

Use equal new on-policy decision and optimizer-update budgets, optimizer settings,
opponent mixture, auxiliary objectives and evaluation schedules. Report extra
prefix replay work separately. The existing early BC checkpoint is a practical
warm start; a claim about direct random-initialization RL requires its own paired
comparison. Keep PPO behavior probabilities consistent with the executed policy,
save sampler/curriculum/optimizer/RNG state, and bound auxiliary shared gradients
before a longer GPU launch. Full-game development evaluation determines whether
the curriculum transfers. Final confirmation cohorts stay unused during pilots.

The old team pools contain published Metamon reconstructions. Once submitted
they give exact simulator labels, but some legal sets have poor strategic
coherence. Keep that distinction in interpreting results. The controlled pilot
does not solve this data-quality problem; a subsequent recipe should add
provenance-preserving sample/tournament teams, validate them in the pinned
simulator and keep their composition splits out of evaluation candidate banks.

This does not fix the separate native Kakuna search bridge. That path still needs
validated public-information world construction and a matched strength test.
The learned belief decoder has not yet justified replacing the stronger existing
set prior. A shared native backbone plus useful learned inference/search remains
a possible contribution; unchanged pretrained play is only its control.
