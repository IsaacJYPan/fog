# Brief 10: a wide team pool from replays, split once, and the anchor round-robin

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Gates G0 and G1 and every held-out evaluation need a wide team pool split into train,
validation and test (ROADMAP P0.7). The planned source, the public forum team bank, cannot be
collected from a cloud sandbox and has been waiting on the owner since 2026-11-12. The tracked
pools are 8 and 16 teams. Brief 04 showed that play on a narrow pool does not transfer.

The repository already holds 22,000 rated games whose team previews are public, and code that
completes a partly revealed team from usage statistics (`complete_private_team`) and writes
validator-clean teams (`fog/scripts/build_generative_pool.py`). This brief builds the pool from
those.

## 2. Rules

- Branch **`work/replay-team-pool`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first. Setup and the replay import are in
  `docs/briefs/02-p2-scale-prep.md` section 4. CPU only.
- No player names in any committed file. Do not fetch anything from the forum.
- Every number is labelled smoke unless it meets the roadmap's 1,000-game rule.

## 3. Stage 1: build the pool

Write `fog/scripts/build_replay_team_pool.py`:

1. Source: imported games rated 1700 or higher. Take each side's team: six species from the
   preview, every revealed item, ability, move and Tera type.
2. Keep a team only if all six species are known and at least 14 of its 24 moves were
   revealed. Complete the rest with `complete_private_team` (one draw, seed from the replay
   id). Record per team how many fields were revealed and how many were sampled.
3. Write each team in the format the existing pools use, and keep only teams that pinned
   Showdown's validator accepts for the current Gen 9 OU rules. Old games contain Pokémon that
   are banned now; dropping them is intended. Report counts at every filter.
4. Deduplicate by species multiset: keep the most-revealed team per composition.
5. Split **by composition and by time**: test = compositions first seen in the 2,000 newest
   games (the replay holdout ids in `data/replay_splits/p2_holdout_2000_ids.txt`) and never
   earlier; validation = a seeded 15% of the remaining compositions; train = the rest. No
   composition may appear in two splits. Cap each split's files at 150 test, 150 validation,
   600 train (seeded sample) so the repository stays small.
6. Output `data/teams/replay_v1/{train,validation,test}/*.txt` and a manifest JSON with, per
   team, the source replay id, rating, revealed and sampled field counts. **Commit the pool**:
   an exact tracked pool is the experiment definition.

Tests: the three splits share no composition; every committed team passes the validator;
the build is deterministic for a seed; no file contains a player name.

Say plainly in the report what this pool is not: items, spreads and unrevealed moves are
samples from usage statistics, so a team can be less coherent than the one a person built.

## 4. Stage 2: anchors on the validation split

With `fog.eval.showdown_direct_selection` on `data/teams/replay_v1/validation`, play the
round-robin of `A0` Random, `A1` MaxDamage and `A2` SimpleHeuristics, at least 1,000 matched
games per pair. Paste the printed ledger rows. `A3` (the retained imitation checkpoint) is not
in a fresh clone: if `runs/` on your machine has it, include it; otherwise write that it was
unavailable. Do not substitute another model for it.

Do not touch the test split.

## 5. What to hand back

1. The script, tests, the committed pool and manifest.
2. One ledger section (`## <date>: replay-derived team pool and anchor round-robin`) with the
   filter counts, split sizes, the SHA-256 of each split's manifest, and the round-robin rows.
3. A dated line under P0.7 in `ROADMAP.md`. Tick P0.7 only if stage 2 completed for `A0–A2`.
4. `docs/briefs/10-report.md`, at most one page: **Answer** (how many teams and compositions
   per split; how much of an average team is revealed rather than sampled; the round-robin
   table), then not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 6. How the reviewer will check this

- No composition is shared between splits; test compositions do not occur in older games.
- Every team validates on pinned Showdown; no player names; the build reproduces from the seed.
- Round-robin rows have at least 1,000 games per pair and come from the validation split.
