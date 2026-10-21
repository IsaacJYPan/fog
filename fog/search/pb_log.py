"""Decision logging and game execution for two-sided search.

`play_and_log_pb_games` maintains one JointBeliefTable per battle and logs root
observations, executed solver distributions, outcomes, and value-training inputs.
Own candidates must have stable identities and count throughout the battle.

After each modeled transition, the observed public transcript is scored by summing
over latent action choices and chance outcomes under the acting strategies.
Opponent likelihoods remain a model approximation outside shared-solve self-play.
Mechanical evidence removes candidates
on both sides that contradict public reveals, including before the first decision.
Move evidence uses subset matching; own-side evidence comes from the mirrored view.

Top-k/top-j truncation reduces the solve while retaining the table's full axes.
Persistent observation updates reject omitted positive-mass candidates, failed cells,
or pruned chance support; zero-mass axes may be omitted. Logs retain kept indices and
dropped mass. OpeningSolveCache reuses identical opening solves and samples a new
action from the stored distribution with the current game's RNG.

Our one-sided replacement decisions use a B=1 solve and are logged normally.
Opponent-only replacements have no solve or behavioral update in the single-side
runner; mechanical evidence still applies. In shared-solve self-play, one solve
supplies both simultaneous strategies, and both sides' rows are logged."""
from __future__ import annotations

import os
import random
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

import numpy as np

from fog.search.belief_table import JointBeliefTable
from fog.search.evidence import (
    keep_mask_from_initial_drive_activation,
    keep_mask_from_reveals,
)
from fog.search.pb_grid import public_label
from fog.search.pb_search import N_ACTIONS, pb_search_action
from fog.search.public_observation import public_observation_key


def _observe_public_outcome(table, result, battle, *, sigma1=None, sigma2=None):
    """Update once from the visible transcript, never from privately submitted action labels."""
    if battle.finished or result is None or result.sims_run == 0:
        return
    model = result.observation_model
    if model is None:
        raise RuntimeError("search result lacks the public observation likelihood model")
    kept = np.zeros(table.log_rho.shape, dtype=bool)
    kept[np.ix_(result.own_kept, result.opp_kept)] = True
    if (result.failed_cells or result.pruned_mass_max > 1e-6
            or np.any(np.isfinite(table.log_rho[~kept]))):
        raise RuntimeError("persistent observation updates require complete candidate/chance support")
    likelihood = model.likelihood(
        public_observation_key(battle),
        result.sigma1 if sigma1 is None else sigma1,
        result.sigma2 if sigma2 is None else sigma2,
    )
    full_likelihood = np.zeros(table.log_rho.shape, dtype=np.float64)
    full_likelihood[np.ix_(result.own_kept, result.opp_kept)] = likelihood
    if not table.observation_step(full_likelihood):
        raise RuntimeError("public observation exhausted candidate support; rejuvenation required")


class OpeningSolveCache:
    """Exact reuse of the opening solve per pool matchup.

    Valid because the solve is a deterministic function of (public state, candidate sets, rho,
    budget params) and none of those depend on the game seed at the factory-fresh opening state:
    grid content is fork-seed-independent (pinned) and the solver is deterministic. The key
    hashes the matchup truths (they determine the opening public state AND which sigma1 row is
    executable), both candidate token sets, the rounded rho, and the search kwargs. Exact-hit-
    only semantics: any difference is a miss, never an approximation. The runner re-samples the
    executed action from the stored distribution with the CURRENT game's rng, so exact sampling
    of the solved strategy is untouched."""

    def __init__(self) -> None:
        self._store: dict[tuple, Any] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(battle, role: str, own_tokens: np.ndarray, opp_pool_tokens: np.ndarray,
                 rho: np.ndarray, search_kwargs: dict | None) -> tuple:
        opp_role = "p2" if role == "p1" else "p1"
        return (
            role,
            battle.truth_tokens(role).tobytes(),
            battle.truth_tokens(opp_role).tobytes(),
            np.asarray(own_tokens, dtype=np.int64).tobytes(),
            np.asarray(opp_pool_tokens, dtype=np.int64).tobytes(),
            np.round(np.asarray(rho, dtype=np.float64), 12).tobytes(),
            repr(sorted((search_kwargs or {}).items())),
        )

    def get(self, key: tuple, rng: random.Random):
        """Stored result with a freshly sampled action, or None."""
        stored = self._store.get(key)
        if stored is None:
            self.misses += 1
            return None
        self.hits += 1
        action = int(rng.choices(range(N_ACTIONS), weights=stored.action_dist)[0])
        return _dc_replace(stored, action=action)

    def put(self, key: tuple, result: Any) -> None:
        self._store[key] = result


def _meta_kwargs(search_kwargs: dict | None) -> dict:
    """search_kwargs for the shard meta: primitives pass through, anything else (a value net
    module, a callable) becomes its repr — the meta must stay torch.load-able anywhere and must
    not embed model weights."""
    out = {}
    for k, v in (search_kwargs or {}).items():
        out[k] = v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)
    return out


def _save_rows(out_path: str, rows: list[dict], meta: dict) -> None:
    """Atomic shard write (tmp + os.replace): a reader or a crash mid-save never sees a torn
    file, and the .tmp name is outside the shard_*.pt glob the loaders scan."""
    import torch

    tmp = str(out_path) + ".tmp"
    torch.save({"rows": rows, "meta": meta}, tmp)
    os.replace(tmp, out_path)


def _heartbeat(out_path: str | None, done: int, total: int, n_rows: int, t0: float) -> None:
    tag = Path(out_path).stem if out_path else "pb_log"
    hours = (time.perf_counter() - t0) / 3600.0
    rate = done / hours if hours > 0 else 0.0
    print(f"[pb_log:{tag}] {done}/{total} games, {n_rows} rows, "
          f"{rate:.1f} games/h", flush=True)


def _action_dist_from_public_row(
    battle: Any, role: str, labels: list[str], row: np.ndarray
) -> np.ndarray:
    """Map one public-label strategy row back to the real 14-action interface for ``role``."""
    label_to_local = {
        public_label(battle, role, int(action)): int(action)
        for action in np.flatnonzero(battle.action_mask(role))
    }
    dist = np.zeros(N_ACTIONS, dtype=np.float64)
    for column, label in enumerate(labels):
        if label in label_to_local:
            dist[label_to_local[label]] += float(row[column])
    dist = np.where(np.asarray(battle.action_mask(role), dtype=bool), dist, 0.0)
    if dist.sum() <= 0.0:
        raise RuntimeError(f"shared solve produced no executable action mass for {role}")
    return dist / dist.sum()


def _mirror_search_result(
    result: Any,
    battle: Any,
    source_role: str,
    other_true_full_index: int,
    rng: random.Random,
) -> Any:
    """Turn one public two-sided solve into the other player's exact view of the same solution.

    ``result.sigma2`` is already the other player's equilibrium strategy. Swapping the two axes,
    transposing the public belief, and mapping its true row to local actions avoids a second solve
    that might predict a different opponent from the one that actually acts.
    """
    other_role = "p2" if source_role == "p1" else "p1"
    true_position = np.flatnonzero(result.opp_kept == int(other_true_full_index))
    if len(true_position) != 1:
        raise RuntimeError(
            "shared self-play solve omitted the acting opponent's true public type. "
            "Opponent-axis truncation is incompatible with shared-solve self-play."
        )
    true_index = int(true_position[0])
    action_dist = _action_dist_from_public_row(
        battle, other_role, result.labels2, result.sigma2[true_index]
    )
    action = int(rng.choices(range(N_ACTIONS), weights=action_dist)[0])

    def _swap_queries(entries):
        return [(label1, label2, np.asarray(belief).T.copy())
                for label1, label2, belief in entries]

    return _dc_replace(
        result,
        action=action,
        action_dist=action_dist,
        labels1=list(result.labels2),
        labels2=list(result.labels1),
        sigma1=np.asarray(result.sigma2).copy(),
        sigma2=np.asarray(result.sigma1).copy(),
        avail1=np.asarray(result.avail2).copy(),
        avail2=np.asarray(result.avail1).copy(),
        rho=np.asarray(result.rho).T.copy(),
        true_index=true_index,
        j_used=int(result.avail1.shape[0]),
        queried_beliefs=_swap_queries(result.queried_beliefs),
        sample_beliefs=_swap_queries(result.sample_beliefs),
        observation_model=(result.observation_model.transposed()
                           if result.observation_model is not None else None),
        own_kept=np.asarray(result.opp_kept).copy(),
        opp_kept=np.asarray(result.own_kept).copy(),
        dropped_own_mass=float(result.dropped_opp_mass),
        dropped_opp_mass=float(result.dropped_own_mass),
        own_tokens=np.asarray(result.opp_tokens).copy(),
        opp_tokens=np.asarray(result.own_tokens).copy(),
    )


def play_and_log_pb_games(
    agent: Any,
    battle_factory: Any,
    own_source: Any,
    opp_pool_tokens: np.ndarray,
    *,
    n_games: int = 1,
    role: str = "p1",
    opponent_policy: Any | None = None,
    out_path: str | None = None,
    seed: int = 0,
    max_decisions: int = 200,
    search_kwargs: dict | None = None,
    solve_cache: OpeningSolveCache | None = None,
    flush_every: int = 0,
) -> list[dict]:
    """Play `n_games` engine games; the learner (`role`) decides via pb_search_action in
    persistent mode over `opp_pool_tokens`, the opponent via `opponent_policy` (default: the
    scripted max-damage bot — data-plumbing default, NOT the self-play consistency target).
    Returns the decision rows (and torch.saves {rows, meta} to `out_path` if given).

    `flush_every` > 0 (with `out_path`): every N completed games, atomically rewrite the shard
    with the rows so far (meta carries partial=True + games_completed) and print a one-line
    heartbeat. Interrupted runs retain their completed games. Default 0 disables
    incremental saving.

    Row fields: game_id, turn, role, obs (deployment input), action, action_dist (the
    distillation target), labels1/labels2, sigma1, sigma2, rho, true_index (our k* on the row's
    KEPT own axis), opp_true_index (the opponent's j* on the row's KEPT opp axis, -1 if off-pool
    or truncated out this decision), own_tokens [K,6,15] / opp_tokens [J,6,15] (the KEPT candidate
    identities rho indexes — the value trainer embeds these and labels V[k*, j*] with
    the outcome), own_kept / opp_kept (full-pool indices behind those axes) with
    dropped_own_mass / dropped_opp_mass (truncation accounting), j_used, sims_run,
    iterations, converged, pruned_mass_max, queried_beliefs, sample_beliefs / sample_iteration
    (posteriors from one random solver iteration), outcome (+1/-1/0 for
    `role`, backfilled at game end), truncated (True when the game hit `max_decisions` — its
    outcome 0 is an artifact, not a draw; the value trainer must drop those rows). `meta` also
    carries `opp_pool_tokens` and `search_kwargs`. Trivial one-legal-action decisions are
    executed but not logged (no behavioral content)."""
    from fog.encoding.observation import Observation

    rng = random.Random(seed)
    opp_role = "p2" if role == "p1" else "p1"
    opp_pool_tokens = np.asarray(opp_pool_tokens, dtype=np.int64)

    def _meta(done: int) -> dict:
        return {"version": 2, "belief_update": "public_observation_likelihood_v1",
                "n_games": int(n_games), "role": role,
                "seed": seed, "j_pool": len(opp_pool_tokens),
                "opp_pool_tokens": opp_pool_tokens,
                "search_kwargs": _meta_kwargs(search_kwargs),
                "value_schema": "played_public_leaf_v1",
                "partial": done < int(n_games), "games_completed": done}

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    for game_id in range(int(n_games)):
        battle = battle_factory(game_id, rng.randrange(2**31))
        policy = opponent_policy
        if policy is None:
            from fog.env.engine_scripted import EngineMaxDamagePolicy

            policy = EngineMaxDamagePolicy(tables=battle.tables, seed=rng.randrange(2**31))
        own0 = own_source.states(battle, role)
        with np.errstate(divide="ignore"):
            table = JointBeliefTable.from_margins(
                np.log(np.asarray(own0.weights, dtype=np.float64)),
                np.zeros(len(opp_pool_tokens)) - np.log(len(opp_pool_tokens)))
        # j*: the opponent's TRUE team index in the pool the belief axis is built over — the
        # The value label is V[k*, j*] ~= outcome. -1 when the real opponent team is not in
        # `opp_pool_tokens` (an off-pool opponent); the value trainer drops those rows.
        opp_truth = np.asarray(battle.truth_tokens(opp_role), dtype=np.int64)
        opp_true_index = next(
            (j for j in range(len(opp_pool_tokens))
             if np.array_equal(opp_pool_tokens[j], opp_truth)), -1)
        if opp_true_index < 0:
            print(f"[pb_log] game {game_id}: opponent team not in opp_pool_tokens — "
                  "value rows for this game carry opp_true_index=-1 (droppable).", flush=True)

        from fog.constants.schema import NO_ITEM

        no_item = int(battle.tables.item_index(NO_ITEM))

        def _mechanical(
            msg: str,
            *,
            battle=battle,
            own0=own0,
            no_item=no_item,
            opp_true_index=opp_true_index,
            table=table,
        ) -> None:
            """Both-side reveal filtering; the own side uses the opponent's mirror view of us."""
            own_view = Observation.from_battle(battle.view(opp_role),
                                               tracker=battle.trackers[opp_role])
            opp_view = Observation.from_battle(battle.view(role),
                                               tracker=battle.trackers[role])
            keep_own = keep_mask_from_reveals(own_view.opp_revealed["tokens"], own0.tokens,
                                              no_item_token=no_item)
            keep_opp = keep_mask_from_reveals(opp_view.opp_revealed["tokens"], opp_pool_tokens,
                                              no_item_token=no_item)
            keep_own &= keep_mask_from_initial_drive_activation(battle, role, own0.tokens)
            keep_opp &= keep_mask_from_initial_drive_activation(
                battle, opp_role, opp_pool_tokens
            )
            if not keep_own[own0.true_index]:
                raise RuntimeError(f"{msg}: the learner's TRUE candidate was mechanically "
                                   "eliminated by its own reveals — a reveal-census bug")
            if opp_true_index >= 0 and not keep_opp[opp_true_index]:
                raise RuntimeError(f"{msg}: the opponent's TRUE pool team was mechanically "
                                   "eliminated — a reveal-census bug")
            table.mechanical_step(keep_own=keep_own, keep_opp=keep_opp)

        # Apply lead reveals before search: a contradicted candidate cannot be
        # matched coherently to the engine's active species when forking.
        _mechanical("initial mechanical step")

        game_rows: list[dict] = []
        outcome = 0.0
        stepped = False                     # False until this game's first battle.step — the
        #                                     factory-fresh state the opening cache keys on
        for decision_id in range(int(max_decisions)):
            if battle.finished:
                break
            pending = set(battle.pending_sides())
            moves: dict[str, str] = {}
            result = None
            if role in pending:
                cache_key = None
                if solve_cache is not None and not stepped:
                    cache_key = OpeningSolveCache.make_key(
                        battle, role, own0.tokens, opp_pool_tokens,
                        table.normalized(), search_kwargs)
                    result = solve_cache.get(cache_key, rng)
                if result is None:
                    result = pb_search_action(
                        agent, battle, role, own_source=own_source,
                        opp_tokens=opp_pool_tokens, table=table,
                        rng=rng, **(search_kwargs or {}))
                    if cache_key is not None and result.sims_run > 0:
                        solve_cache.put(cache_key, result)
                moves[role] = battle.engine_action(role, result.action)
            if opp_role in pending:
                opp_idx = policy.choose(battle, opp_role)
                moves[opp_role] = battle.engine_action(opp_role, int(opp_idx))

            solver_row = result is not None and result.sims_run > 0
            value_row = None
            if solver_row:
                my_label = public_label(battle, role, result.action)
                # j* on THIS row's kept axis (the axes rho/sigma2/opp_tokens live on)
                row_j = -1
                if opp_true_index >= 0:
                    pos = np.flatnonzero(result.opp_kept == opp_true_index)
                    row_j = int(pos[0]) if len(pos) else -1
                value_row = {
                    "game_id": game_id, "decision_id": decision_id, "turn": battle.turn, "role": role,
                    "obs": Observation.from_battle(battle.view(role),
                                                   tracker=battle.trackers[role]),
                    "action": result.action, "action_dist": result.action_dist,
                    # the executed action's column in labels1 — lets the distill-v2 loader
                    # compute the action-conditioned own posterior P(k | a) exactly (I.1)
                    "action_label_col": (result.labels1.index(my_label)
                                         if my_label in result.labels1 else -1),
                    "labels1": result.labels1, "labels2": result.labels2,
                    "sigma1": result.sigma1, "sigma2": result.sigma2, "rho": result.rho,
                    "true_index": result.true_index, "opp_true_index": row_j,
                    "own_tokens": result.own_tokens, "opp_tokens": result.opp_tokens,
                    "own_kept": result.own_kept, "opp_kept": result.opp_kept,
                    "dropped_own_mass": result.dropped_own_mass,
                    "dropped_opp_mass": result.dropped_opp_mass,
                    "j_used": result.j_used,
                    "sims_run": result.sims_run, "iterations": result.iterations,
                    "converged": result.converged,
                    "pruned_mass_max": result.pruned_mass_max,
                    "queried_beliefs": result.queried_beliefs,
                    "sample_beliefs": result.sample_beliefs,
                    "sample_iteration": result.sample_iteration,
                    "outcome": None, "truncated": None,
                }
                game_rows.append(value_row)

            battle.step(moves)
            stepped = True

            # Condition on the outcome, summing over private action choices producing it.
            if solver_row:
                _observe_public_outcome(table, result, battle)
            # ---- then mechanical evidence from the updated reveal state -------------------
            _mechanical(f"turn {battle.turn}")
            if value_row is not None and not battle.finished:
                # Outcome supervision belongs to the public leaf actually reached, paired with
                # the belief after the actions and new truthful evidence. Hypothetical queried
                # beliefs stay in the row for coverage evaluation, never attached to this board.
                value_row.update({
                    "value_obs": Observation.public_from_engine_battle(battle, role),
                    "value_rho": table.normalized().copy(),
                    "value_own_tokens": np.asarray(own0.tokens, dtype=np.int64).copy(),
                    "value_opp_tokens": opp_pool_tokens.copy(),
                    "value_true_index": int(own0.true_index),
                    "value_opp_true_index": int(opp_true_index),
                    "value_target_kind": "played_leaf_outcome",
                })

        if battle.finished:
            outcome = (1.0 if battle.winner == role
                       else -1.0 if battle.winner is not None else 0.0)
        for row in game_rows:
            row["outcome"] = outcome
            row["truncated"] = not battle.finished
        all_rows.extend(game_rows)
        if out_path is not None and flush_every > 0 and (game_id + 1) % int(flush_every) == 0:
            _save_rows(out_path, all_rows, _meta(game_id + 1))
            _heartbeat(out_path, game_id + 1, int(n_games), len(all_rows), t0)

    if out_path is not None:
        _save_rows(out_path, all_rows, _meta(int(n_games)))
    return all_rows


def play_and_log_pb_selfplay_games(
    agent: Any,
    battle_factory: Any,
    own_source: Any,
    pool_tokens: np.ndarray,
    *,
    n_games: int = 1,
    out_path: str | None = None,
    seed: int = 0,
    max_decisions: int = 200,
    search_kwargs: dict | None = None,
    solve_cache: OpeningSolveCache | None = None,
    flush_every: int = 0,
) -> list[dict]:
    """Shared-solve self-play; see the module docstring.

    At simultaneous decisions one call to ``pb_search_action`` produces both strategies; the
    second strategy is mirrored into the other player's result and is the one actually sampled.
    Each observed action therefore updates both tables with the distribution that really acted.

    `own_source` serves both roles and MUST index exactly `pool_tokens` (both teams come from
    the pool). Rows carry the same schema as `play_and_log_pb_games` with `role` distinguishing
    the deciding side (2x rows per game); outcome is per-row from that role's perspective.
    `flush_every` behaves as in `play_and_log_pb_games` (incremental atomic shard saves)."""
    from fog.constants.schema import NO_ITEM
    from fog.encoding.observation import Observation

    rng = random.Random(seed)
    pool_tokens = np.asarray(pool_tokens, dtype=np.int64)
    if not np.array_equal(np.asarray(own_source.tokens, dtype=np.int64), pool_tokens):
        raise ValueError("self-play needs ONE shared candidate space: own_source.tokens must "
                         "be exactly pool_tokens (both tables' axes index the same pool)")
    roles = ("p1", "p2")
    if (search_kwargs or {}).get("top_j") is not None:
        top_j = int((search_kwargs or {})["top_j"])
        if top_j < len(pool_tokens):
            raise ValueError(
                "shared-solve self-play requires the full opponent axis: top_j may not remove "
                "the private row that must execute. Use top_j=None for claim-bearing self-play."
            )

    def _meta(done: int) -> dict:
        return {"version": 2, "belief_update": "public_observation_likelihood_v1",
                "n_games": int(n_games), "mode": "selfplay",
                "seed": seed, "j_pool": len(pool_tokens),
                "opp_pool_tokens": pool_tokens,
                "search_kwargs": _meta_kwargs(search_kwargs),
                "value_schema": "played_public_leaf_v1",
                "partial": done < int(n_games), "games_completed": done}

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    for game_id in range(int(n_games)):
        battle = battle_factory(game_id, rng.randrange(2**31))
        no_item = int(battle.tables.item_index(NO_ITEM))
        own0 = {r: own_source.states(battle, r) for r in roles}
        tables = {}
        for r in roles:
            o = "p2" if r == "p1" else "p1"
            with np.errstate(divide="ignore"):
                tables[r] = JointBeliefTable.from_margins(
                    np.log(np.asarray(own0[r].weights, dtype=np.float64)),
                    np.log(np.asarray(own0[o].weights, dtype=np.float64)))

        def _mechanical(
            msg: str,
            *,
            battle=battle,
            no_item=no_item,
            own0=own0,
            tables=tables,
        ) -> None:
            """One reveal mask per side's public state; each table gets (own, opp) views."""
            rev = {}
            for r in roles:
                o = "p2" if r == "p1" else "p1"
                view = Observation.from_battle(battle.view(o), tracker=battle.trackers[o])
                rev[r] = keep_mask_from_reveals(view.opp_revealed["tokens"], pool_tokens,
                                                no_item_token=no_item)
                rev[r] &= keep_mask_from_initial_drive_activation(battle, r, pool_tokens)
                if not rev[r][own0[r].true_index]:
                    raise RuntimeError(f"{msg}: {r}'s TRUE candidate was mechanically "
                                       "eliminated by its own reveals — a reveal-census bug")
            for r in roles:
                o = "p2" if r == "p1" else "p1"
                tables[r].mechanical_step(keep_own=rev[r], keep_opp=rev[o])

        _mechanical("initial mechanical step")

        game_rows: list[dict] = []
        stepped = False
        for decision_id in range(int(max_decisions)):
            if battle.finished:
                break
            pending = set(battle.pending_sides())
            results: dict[str, Any] = {}
            labels: dict[str, str | None] = {}
            moves: dict[str, str] = {}
            legal = {r: np.flatnonzero(battle.action_mask(r)) for r in pending}
            acting = [r for r in roles if r in pending and len(legal[r]) >= 1]

            if acting:
                # Even a forced move has informative chance outcomes. One public solve supplies
                # both strategies, and the same observation updates both tables.
                source = "p1" if "p1" in acting else acting[0]
                other = "p2" if source == "p1" else "p1"
                if pending == {"p1", "p2"}:
                    np.testing.assert_allclose(
                        tables[source].normalized(), tables[other].normalized().T,
                        atol=1e-10,
                        err_msg="the two public belief tables stopped being transposes",
                    )
                cache_key = None
                result = None
                if solve_cache is not None and not stepped:
                    cache_key = OpeningSolveCache.make_key(
                        battle, source, pool_tokens, pool_tokens,
                        tables[source].normalized(), search_kwargs)
                    result = solve_cache.get(cache_key, rng)
                if result is None:
                    result = pb_search_action(
                        agent, battle, source, own_source=own_source,
                        opp_tokens=pool_tokens, table=tables[source],
                        rng=rng, **(search_kwargs or {}))
                    if cache_key is not None and result.sims_run > 0:
                        solve_cache.put(cache_key, result)
                results[source] = result
                labels[source] = public_label(battle, source, result.action)
                moves[source] = battle.engine_action(source, result.action)

                if other in acting and pending == {"p1", "p2"}:
                    mirrored = _mirror_search_result(
                        result, battle, source, own0[other].true_index, rng
                    )
                    results[other] = mirrored
                    labels[other] = public_label(battle, other, mirrored.action)
                    moves[other] = battle.engine_action(other, mirrored.action)

            step_value_rows: dict[str, dict] = {}
            for r, result in results.items():
                if result.sims_run == 0:
                    continue
                o = "p2" if r == "p1" else "p1"
                true_opp = own0[o].true_index          # shared axes: opp truth = its own index
                pos = np.flatnonzero(result.opp_kept == true_opp)
                row = {
                    "game_id": game_id, "decision_id": decision_id, "turn": battle.turn, "role": r,
                    "obs": Observation.from_battle(battle.view(r),
                                                   tracker=battle.trackers[r]),
                    "action": result.action, "action_dist": result.action_dist,
                    "action_label_col": (result.labels1.index(labels[r])
                                         if labels[r] in result.labels1 else -1),
                    "labels1": result.labels1, "labels2": result.labels2,
                    "sigma1": result.sigma1, "sigma2": result.sigma2, "rho": result.rho,
                    "true_index": result.true_index,
                    "opp_true_index": int(pos[0]) if len(pos) else -1,
                    "own_tokens": result.own_tokens, "opp_tokens": result.opp_tokens,
                    "own_kept": result.own_kept, "opp_kept": result.opp_kept,
                    "dropped_own_mass": result.dropped_own_mass,
                    "dropped_opp_mass": result.dropped_opp_mass,
                    "j_used": result.j_used,
                    "sims_run": result.sims_run, "iterations": result.iterations,
                    "converged": result.converged,
                    "pruned_mass_max": result.pruned_mass_max,
                    "queried_beliefs": result.queried_beliefs,
                    "sample_beliefs": result.sample_beliefs,
                    "sample_iteration": result.sample_iteration,
                    "outcome": None, "truncated": None,
                }
                game_rows.append(row)
                step_value_rows[r] = row

            battle.step(moves)
            stepped = True

            for r in roles:
                o = "p2" if r == "p1" else "p1"
                result, opposite = results.get(r), results.get(o)
                if result is None and opposite is not None and opposite.sims_run:
                    # This side only observed a replacement/pivot. Its posterior still changes.
                    result = _dc_replace(
                        opposite, sigma1=opposite.sigma2, sigma2=opposite.sigma1,
                        own_kept=opposite.opp_kept, opp_kept=opposite.own_kept,
                        observation_model=opposite.observation_model.transposed(),
                    )
                sigma2 = None
                if (r in results and result.sims_run and opposite is not None
                        and opposite.sims_run):
                    if (not np.array_equal(result.opp_kept, opposite.own_kept)
                            or set(result.labels2) != set(opposite.labels1)):
                        raise RuntimeError("acting strategy axes do not match the observation model")
                    sigma2 = opposite.sigma1[:, [opposite.labels1.index(label)
                                                for label in result.labels2]]
                _observe_public_outcome(tables[r], result, battle, sigma2=sigma2)
            _mechanical(f"turn {battle.turn}")
            if not battle.finished:
                for r, row in step_value_rows.items():
                    o = "p2" if r == "p1" else "p1"
                    row.update({
                        "value_obs": Observation.public_from_engine_battle(battle, r),
                        "value_rho": tables[r].normalized().copy(),
                        "value_own_tokens": pool_tokens.copy(),
                        "value_opp_tokens": pool_tokens.copy(),
                        "value_true_index": int(own0[r].true_index),
                        "value_opp_true_index": int(own0[o].true_index),
                        "value_target_kind": "played_leaf_outcome",
                    })

        winner = battle.winner if battle.finished else None
        for row in game_rows:
            row["outcome"] = (0.0 if winner is None
                              else 1.0 if winner == row["role"] else -1.0)
            row["truncated"] = not battle.finished
        all_rows.extend(game_rows)
        if out_path is not None and flush_every > 0 and (game_id + 1) % int(flush_every) == 0:
            _save_rows(out_path, all_rows, _meta(game_id + 1))
            _heartbeat(out_path, game_id + 1, int(n_games), len(all_rows), t0)

    if out_path is not None:
        _save_rows(out_path, all_rows, _meta(int(n_games)))
    return all_rows
