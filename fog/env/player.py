"""poke-env Player wrapper.

The module stays import-light: it does not subclass poke-env's Player until `FogPlayer.build()` is
called at runtime.

Training integration: `attach_buffer(RolloutBuffer)` turns every `act` into a recorded decision
(unbatched C3 fragment + action + log-prob + value + the mirror observation for self-belief training)
and every battle end into `end_game(±1, final_leak_ce)`. All per-battle state (RevealLedger,
BattleTracker, fed-turn bookkeeping) is keyed by battle_tag so `max_concurrent_battles > 1` is safe.
History/ledger reset between battles is handled in the battle-finished callback — the M1 silent-bug
guard.

Leakage (§7): `phi_hat` is a `FrozenProbe` — a frozen (backbone + self-belief) pair, so the ΔI
measurement never depends on live weights. Each decision records the probe's readability CE; the
buffer differences consecutive CEs into per-turn ΔI, and the battle-finished hook contributes the
final post-game measurement so end-of-game reveals are penalized too.

Team preview: `teampreview` runs the agent's TeamPreviewHead over the preview entities and returns
the sampled order; the preview pick is recorded into the buffer as a bandit decision (game outcome
as reward — see ppo_update's preview term).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from fog.env.actions import active_entity_indices, map_action_to_order
from fog.training.progress_reward import battle_progress_score as _progress_score

if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle

    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer


class FogPlayer:
    """Thin adapter from `FogAgent` decisions to poke-env battle orders."""

    def __init__(
        self,
        agent: FogAgent | None = None,
        *,
        temperature: float = 1.0,
        tera_logit_bonus: float = 0.0,
    ) -> None:
        self.agent = agent
        self.temperature = temperature
        self.tera_logit_bonus = float(tera_logit_bonus)
        if not np.isfinite(self.tera_logit_bonus) or self.tera_logit_bonus < 0:
            raise ValueError("tera_logit_bonus must be finite and non-negative")
        self._player_ref: Any | None = None
        self._team_builder: Any | None = None
        self._buffer: RolloutBuffer | None = None
        self._game_ids: dict[str, int] = {}
        # per-battle state, keyed by battle_tag (safe under max_concurrent_battles > 1)
        self._ledgers: dict[str, Any] = {}
        self._trackers: dict[str, Any] = {}
        self._fed_turns: dict[str, set[int]] = {}
        self.on_battle_end: Callable[[Any], None] | None = None
        self.on_battle_start: Callable[[Any], None] | None = None
        self.decision_delay_seconds = 0.0
        self._announced_battles: set[str] = set()
        # Optional policy-distillation tap: called with the exact unbatched fragment and acting
        # output before the decision is added to the rollout buffer. It is deliberately inert in
        # ordinary training and evaluation.
        self.decision_observer: Callable[[str, dict, Any], None] | None = None
        # Optional exact action-label provider used by supervised policy collection. It receives
        # the live battle, exact observation, and agent output, and must return a legal project
        # action index. Ordinary PPO/inference leaves this unset.
        self.action_override: Callable[[Any, Any, Any], int] | None = None
        # Optional DAgger labeler. Unlike ``action_override`` this never changes the live action:
        # it only stores a legal target alongside the exact accepted rollout fragment, so rejected
        # Showdown choices are rolled back atomically with their supervision.
        self.supervised_label_provider: Callable[[Any, Any, Any], int] | None = None
        # Opt-in supervised-data metadata. When true, the finalized rollout fragment retains the
        # donor's full masked policy distribution. It remains inside RolloutBuffer so a rejected
        # Showdown choice is rolled back together with the action row.
        self.record_policy_reference = False
        # optional: finished battle -> [6,15] opponent truth tokens (self-play belief supervision)
        self.truth_provider: Callable[[Any], Any] | None = None
        # Optional immutable submitted private-state truth for the acting player's own team.
        # Direct-Showdown training has no poke-env team builder, so reading the live battle objects
        # here would incorrectly turn temporary in-battle changes (Skill Swap, Transform, etc.) into
        # self-belief labels.
        self.private_state_truth_provider: Callable[[Any], Any] | None = None
        # optional leakage measurement (§7): a FrozenProbe (frozen backbone + self-belief). When set,
        # each decision records the probe's self-readability CE so the rollout buffer can derive the
        # per-turn ΔI penalty. `leak_family_weights` rescales per-family CE (weighted-ΔI ablation).
        self.phi_hat = None
        self.leak_family_weights: dict[str, float] | None = None
        # mirror recording for self-belief TRAINING (§3.3): tri-state — None = auto (try once,
        # disable permanently on first failure), True/False = forced. Only active with a buffer.
        self.record_mirror: bool | None = None
        self.mirror_last_error: str | None = None
        # optional set-oracle truth injection (§3.4 eval): battle -> [6,15] opponent truth tokens fed
        # to the belief net as opp_truth_tokens (perfect-information ceiling).
        self.oracle_truth_provider: Callable[[Any], Any] | None = None
        # optional determinized search at action time (§8): when set, act() selects via
        # search.determinized instead of sampling the policy. Reconstructs an engine position from
        # the live battle + belief-sampled worlds and rolls one ply each (search/engine_step). Dict:
        # {"k": int, "prior_weight": float, "seed": int}. Inference-only; never during training.
        self.search: dict | None = None
        # Search failures used to be swallowed by the policy fallback, making a broken search row
        # look indistinguishable from policy-only play. Evaluations read these counters through the
        # built player's `_fog_adapter` reference.
        self.search_stats: dict[str, Any] = {
            "decisions": 0,
            "simulation_attempts": 0,
            "simulation_failures": 0,
            "fallbacks": 0,
            "overrides": 0,
            "last_error": None,
            "failure_reasons": {},
        }
        # Optional subsystems may degrade without aborting a battle, but never silently.  Each
        # boundary records a stable name, count, and last concrete error for run reports/tests.
        self.diagnostic_errors: dict[str, dict[str, Any]] = {}

    def _record_diagnostic_error(self, name: str, exc: BaseException) -> None:
        entry = self.diagnostic_errors.setdefault(name, {"count": 0, "last_error": None})
        entry["count"] += 1
        entry["last_error"] = f"{type(exc).__name__}: {exc}"

    def attach_buffer(self, buffer: RolloutBuffer) -> None:
        """Record every decision + game outcome into `buffer` (training mode)."""
        self._buffer = buffer

    def build(self, **poke_env_kwargs: Any):
        """Construct and return the actual poke-env Player subclass bound to this adapter."""
        try:
            from poke_env.player import Player
        except ImportError as exc:  # pragma: no cover - depends on optional poke-env install
            raise RuntimeError(
                "poke-env is required to build a live FogPlayer. Install project dependencies first."
            ) from exc

        adapter = self

        # NOTE: the class name feeds poke-env's default username. It must contain no characters
        # Showdown strips (a leading underscore once made the client log in as "_Player 1" while the
        # server registered "Player 1" — the challenge handshake then targeted a ghost name and every
        # battle hung forever). Prefer passing an explicit unique account via `unique_account()`.
        class FogBot(Player):  # type: ignore[misc]
            def choose_move(self, battle):
                return adapter.act(battle)

            def teampreview(self, battle):
                return adapter.choose_teampreview(battle)

            def _battle_finished_callback(self, battle):
                try:
                    callback = getattr(super(), "_battle_finished_callback", None)
                    return callback(battle) if callback is not None else None
                finally:
                    adapter._battle_finished(battle)

        self._team_builder = poke_env_kwargs.get("team")
        self._player_ref = FogBot(**poke_env_kwargs)
        self._player_ref._fog_adapter = self
        return self._player_ref

    # ---------------------------------------------------------------------------- per-battle state
    def _battle_state(self, battle: Any) -> tuple[str, Any, Any, set[int]]:
        """(tag, ledger, tracker, fed_turns) for this battle, creating them on first sight."""
        tag = str(getattr(battle, "battle_tag", "battle"))
        if tag not in self._trackers:
            from fog.encoding.tracker import BattleTracker

            role = getattr(battle, "player_role", None) or "p1"
            tracker = BattleTracker(my_role=role)
            private_truth = None
            if self.private_state_truth_provider is not None:
                private_truth = self.private_state_truth_provider(battle)
            if private_truth is None:
                paste = getattr(self._team_builder, "last_paste", None)
                if paste:
                    try:
                        from fog.constants.tables import TABLES
                        from fog.env.team_truth import truth_tokens_from_paste

                        private_truth = truth_tokens_from_paste(paste, TABLES)
                    except Exception as exc:  # noqa: BLE001 - optional builder metadata boundary
                        self._record_diagnostic_error("team_builder_private_truth", exc)
                        # Non-pool teambuilders and malformed optional metadata keep the
                        # request-stat fallback; ordinary policy inference remains available.
                        private_truth = None
            if private_truth is not None:
                tracker.record_exact_private_truth(private_truth)
            self._trackers[tag] = tracker
            self._fed_turns[tag] = set()
            from fog.encoding.reveals import RevealLedger

            self._ledgers[tag] = RevealLedger()
        return tag, self._ledgers[tag], self._trackers[tag], self._fed_turns[tag]

    def _game_id(self, tag: str) -> int:
        gid = self._game_ids.get(tag)
        if gid is None:
            # NOT setdefault(tag, start_game()): that evaluates start_game() eagerly on every
            # decision and litters the buffer with phantom unfinished games.
            gid = self._buffer.start_game()
            self._game_ids[tag] = gid
        return gid

    # ------------------------------------------------------------------------------- decision path
    def act(self, battle: AbstractBattle):
        """Build an observation, sample an agent action, and return the corresponding BattleOrder."""
        player = self._player_ref
        if player is None:
            raise RuntimeError("FogPlayer.act requires a built poke-env Player instance.")
        if self.agent is None:
            return player.choose_default_move()

        import torch

        from fog.encoding.batch import observation_to_batch
        from fog.encoding.observation import Observation

        tag, ledger, tracker, fed = self._battle_state(battle)
        if tag not in self._announced_battles:
            self._announced_battles.add(tag)
            if self.on_battle_start is not None:
                self.on_battle_start(battle)
        tracker.record_private_team(battle)
        self._feed_events(battle, ledger, tracker, fed)
        obs = Observation.from_battle(battle, tracker=tracker)
        obs.validate()

        device = _agent_device(self.agent)
        active_idx = (
            obs.active_idx
            if obs.active_idx is not None
            else np.asarray(active_entity_indices(battle), dtype=np.int64)
        )
        batch = observation_to_batch(obs, device)
        oracle_truth = None
        if self.oracle_truth_provider is not None:
            oracle_truth = self.oracle_truth_provider(battle)
            if oracle_truth is not None:
                oracle_truth = np.asarray(oracle_truth, dtype=np.int64)
                batch["opp_truth_tokens"] = torch.as_tensor(
                    oracle_truth, dtype=torch.long, device=device
                ).unsqueeze(0)

        with torch.no_grad():
            out = self.agent(batch)
        recorded_log_probs = out.log_probs[0]
        if self.action_override is not None:
            action_idx = int(self.action_override(battle, obs, out))
            if not bool(obs.legality_mask[action_idx]):
                raise ValueError(f"action_override returned illegal action {action_idx}")
        elif self.search is not None:
            action_idx = self._search_action(battle, obs, tracker, batch, device)
        else:
            action_idx = _sample_action(
                out.log_probs[0],
                temperature=self.temperature,
                tera_logit_bonus=self.tera_logit_bonus,
            )
            if self.temperature > 0:
                # PPO must store the probability of the distribution that actually sampled the
                # action.  Previously temperature changed the sample but the untempered policy
                # probability was recorded, silently corrupting every importance ratio at T != 1.
                recorded_log_probs = _behavior_log_probs(
                    out.log_probs[0],
                    temperature=self.temperature,
                    tera_logit_bonus=self.tera_logit_bonus,
                )

        if self._buffer is not None or self.decision_observer is not None:
            frag = {
                "entities": obs.entities,
                "entity_ids": obs.entity_ids,
                "field": obs.field,
                "history": obs.history,
                "history_ids": obs.history_ids,
                "history_state": obs.history_state,
                "history_len": np.int64(obs.history_len),
                "active_idx": active_idx,
                "legality_mask": obs.legality_mask,
                "action_features": obs.action_features,
                "opp_revealed": obs.opp_revealed,
                "turn": np.int64(getattr(battle, "turn", 0) or 0),
            }
            mirror = None
            if self._buffer is not None:
                gid = self._game_id(tag)
                if self.record_policy_reference:
                    frag["reference_log_probs"] = out.log_probs[0].detach().cpu().numpy()
                if self.supervised_label_provider is not None:
                    supervised_label = int(self.supervised_label_provider(battle, obs, out))
                    if not 0 <= supervised_label < len(obs.legality_mask):
                        raise ValueError(
                            f"supervised_label_provider returned out-of-range action "
                            f"{supervised_label}"
                        )
                    if not bool(obs.legality_mask[supervised_label]):
                        raise ValueError(
                            f"supervised_label_provider returned illegal action {supervised_label}"
                        )
                    frag["supervised_label"] = np.int64(supervised_label)
                if self.oracle_truth_provider is not None:
                    # The update-time forward must see what the acting forward saw (acting with
                    # truth but updating without silently breaks the PPO ratio). Key always
                    # present so fragment structure is batch-constant.
                    frag["opp_truth_tokens"] = (
                        oracle_truth if oracle_truth is not None
                        else np.full((6, 15), -1, dtype=np.int64)
                    )
                mirror = self._mirror(battle, obs, ledger)
                if mirror is not None:
                    mirror_frag, my_view = mirror
                    frag["opp_view"] = {**mirror_frag, "my_view": my_view}
            if self.decision_observer is not None:
                self.decision_observer(tag, frag, out)
            if self._buffer is not None:
                self._buffer.add_decision(
                    gid,
                    frag,
                    action=action_idx,
                    log_prob=float(recorded_log_probs[action_idx]),
                    value=float(out.value[0]),
                    leak_ce=self._leakage_ce(mirror, obs, device),
                    progress_score=_progress_score(battle),
                )

        # history retro-fidelity: snapshot + my chosen action for this turn (§2.3)
        from fog.constants.tables import TABLES
        from fog.encoding.events import record_history_decision

        record_history_decision(
            tracker, int(getattr(battle, "turn", 0) or 0), battle, TABLES, action_idx
        )

        if self.decision_delay_seconds > 0:
            import time

            time.sleep(self.decision_delay_seconds)
        return map_action_to_order(battle, action_idx, player)

    def discard_last_uncommitted_decision(self, battle: Any) -> None:
        """Remove the PPO row for a Showdown-rejected choice before retrying the request."""
        if self._buffer is None:
            return
        tag = str(getattr(battle, "battle_tag", ""))
        gid = self._game_ids.get(tag)
        if gid is None:
            raise RuntimeError(f"no rollout game exists for rejected choice in {tag!r}")
        self._buffer.discard_last_decision(gid)

    # ------------------------------------------------------------------------------- team preview
    def choose_teampreview(self, battle: Any) -> str:
        """Order my team with the agent's TeamPreviewHead (§3.1). Falls back to default order."""
        if self.agent is None or getattr(self.agent, "preview", None) is None:
            return "/team 123456"
        try:
            import torch

            from fog.encoding.observation import Observation

            tag, _, tracker, _ = self._battle_state(battle)
            tracker.record_private_team(battle)
            obs = Observation.from_battle(battle, tracker=tracker)
            device = _agent_device(self.agent)
            pre = {
                "entities": torch.as_tensor(obs.entities, dtype=torch.float32, device=device).unsqueeze(0),
                "entity_ids": torch.as_tensor(obs.entity_ids, dtype=torch.long, device=device).unsqueeze(0),
                "field": torch.as_tensor(obs.field, dtype=torch.float32, device=device).unsqueeze(0),
            }
            with torch.no_grad():
                logits = self.agent.preview_logits(pre)[0]                      # [6]
            n_mons = len(getattr(battle, "team", None) or {}) or 6
            logits = logits[:n_mons]
            raw_log_probs = torch.log_softmax(logits, dim=-1)
            if self.temperature <= 0:
                lead = int(logits.argmax())
                behavior_log_probs = raw_log_probs
            else:
                behavior_log_probs = _temperature_log_probs(
                    raw_log_probs, temperature=self.temperature
                )
                probs = behavior_log_probs.exp()
                lead = int(torch.multinomial(probs, 1))
            log_prob = float(behavior_log_probs[lead])
            order = [lead] + [i for i in range(n_mons) if i != lead]

            if self._buffer is not None:
                gid = self._game_id(tag)
                self._buffer.add_preview(
                    gid,
                    {"entities": obs.entities, "entity_ids": obs.entity_ids, "field": obs.field},
                    slot=lead,
                    log_prob=log_prob,
                )
            return "/team " + "".join(str(i + 1) for i in order)
        except Exception as exc:  # noqa: BLE001 - team-preview fallback is explicit and recorded
            self._record_diagnostic_error("team_preview", exc)
            return "/team 123456"

    # -------------------------------------------------------------------------------------- search
    def _search_action(self, battle: Any, obs: Any, tracker: Any, batch: dict, device) -> int:
        """Determinized action via engine rollouts reconstructed from the live battle (§8). Any
        failure (engine missing, reconstruction/step error) falls back to policy sampling so a
        search run never crashes a game."""
        import torch

        self.search_stats["decisions"] += 1
        stepper = None
        stepper_failures_accounted = False
        try:
            cfg = self.search or {}
            if cfg.get("mode") == "direct_two_sided":
                from fog.search.direct_two_sided import direct_two_sided_action

                role = getattr(battle, "player_role", None) or "p1"
                result = direct_two_sided_action(
                    self.agent,
                    battle,
                    role=role,
                    device=device,
                    iters=int(cfg.get("iters", 256)),
                    floor=float(cfg.get("floor", 0.02)),
                    min_probability=float(cfg.get("min_probability", 0.0)),
                    max_branches=cfg.get("max_branches"),
                )
                self.search_stats["simulation_attempts"] += result.simulations
                self.search_stats["chance_branches"] = (
                    int(self.search_stats.get("chance_branches", 0)) + result.chance_branches
                )
                self.search_stats["pruned_mass_max"] = max(
                    float(self.search_stats.get("pruned_mass_max", 0.0)),
                    result.pruned_mass_max,
                )
                with torch.no_grad():
                    policy_action = int(self.agent(batch).log_probs[0].argmax())
                self.search_stats["overrides"] += int(result.action != policy_action)
                return result.action

            self.search_stats["simulation_attempts"] += (
                int(batch["legality_mask"][0].sum().item()) * int(cfg.get("k", 16))
            )
            from fog.search.determinized import determinized_action
            from fog.search.engine_step import EngineAgentPolicy, ShowdownWorldStepper

            role = getattr(battle, "player_role", None) or "p1"
            opponent_policy = None
            if cfg.get("opponent_model") == "policy":
                opponent_policy = EngineAgentPolicy(self.agent, temperature=0.0)
            stepper = ShowdownWorldStepper(
                battle, tracker, role=role, opponent_policy=opponent_policy,
                seed=int(cfg.get("seed", 0)),
            )
            action = determinized_action(
                self.agent, batch, step_fn=stepper,
                k=int(cfg.get("k", 16)), prior_weight=float(cfg.get("prior_weight", 0.5)),
                min_override_margin=float(cfg.get("min_override_margin", 0.0)),
                uncertainty_z=float(cfg.get("uncertainty_z", 0.0)),
            )
            with torch.no_grad():
                policy_action = int(self.agent(batch).log_probs[0].argmax())
            self.search_stats["overrides"] += int(action != policy_action)
            self.search_stats["simulation_failures"] += int(stepper.failures)
            self.search_stats["failure_reasons"] = dict(
                getattr(stepper, "failure_reasons", {})
            )
            stepper_failures_accounted = True
            # if every world's reconstruction failed, search saw nothing — fall back to the policy
            if stepper.failures and not stepper._bases:
                raise RuntimeError("all search reconstructions failed")
            return action
        except Exception as exc:  # noqa: BLE001 - inference fallback is counted in search_stats
            if stepper is not None and not stepper_failures_accounted:
                self.search_stats["simulation_failures"] += int(stepper.failures)
                self.search_stats["failure_reasons"] = dict(
                    getattr(stepper, "failure_reasons", {})
                )
            self.search_stats["fallbacks"] += 1
            self.search_stats["last_error"] = f"{type(exc).__name__}: {exc}"
            with torch.no_grad():
                out = self.agent(batch)
            return _sample_action(out.log_probs[0], temperature=self.temperature)

    # ------------------------------------------------------------------------------ mirror/leakage
    def _mirror(self, battle: Any, obs: Any, ledger: Any) -> tuple[dict, dict] | None:
        """Unbatched (mirror_frag, my_view) numpy pair, or None when mirror recording is off/broken.

        Auto-disables permanently on the first failure (before anything was recorded) so every frag
        in a batch has identical structure; after the first success, failures degrade to a zero-fill
        mirror with -1 labels (no loss contribution) instead of breaking frag structure.
        """
        if self.record_mirror is False or (self._buffer is None and self.phi_hat is None):
            return None
        try:
            from fog.constants.tables import TABLES
            from fog.encoding.mirror import build_opponent_view_np
            from fog.encoding.reveals import tokens_from_views

            team = getattr(battle, "team", None) or {}
            my_keys = [str(k) for k in team]
            my_species = [TABLES.species_index(_species_id(m)) for m in team.values()]
            my_view_tokens = tokens_from_views(
                ledger.views(my_keys, my_species) if ledger else []
            )
            my_truth_tokens = None
            if self.private_state_truth_provider is not None:
                try:
                    my_truth_tokens = self.private_state_truth_provider(battle)
                except Exception as exc:  # noqa: BLE001 - provider boundary; mirror records failure
                    self._record_diagnostic_error("mirror_private_truth_provider", exc)
                    my_truth_tokens = None
            if my_truth_tokens is None:
                paste = getattr(self._team_builder, "last_paste", None)
            else:
                paste = None
            if paste:
                from fog.env.team_truth import truth_tokens_from_paste

                my_truth_tokens = truth_tokens_from_paste(paste, TABLES)
            elif my_truth_tokens is None:
                my_truth_tokens = _own_truth_tokens(battle, TABLES)
            out = build_opponent_view_np(obs, my_view_tokens, my_truth_tokens, TABLES)
            if self.record_mirror is None:
                self.record_mirror = True
            return out
        except Exception as exc:  # noqa: BLE001 - optional mirror boundary; error retained below
            self.mirror_last_error = f"{type(exc).__name__}: {exc}"
            if self.record_mirror is None:
                self.record_mirror = False   # never worked: drop the opp_view key entirely
                return None
            return _zero_mirror(obs)         # worked before: keep frag structure, no labels

    def _leakage_ce(self, mirror: tuple[dict, dict] | None, obs: Any, device) -> float | None:
        """Frozen-probe self-readability CE for this decision (§7); None when leakage is off."""
        if self.phi_hat is None or mirror is None:
            return None
        try:
            import torch

            mirror_frag, my_view = mirror
            # Structure-preserving zero mirror (mid-game mirror failure): all labels are -1, its CE
            # is 0 by construction, and differencing a real CE against it fabricates a large ΔI
            # spike. No labels -> no measurement; the buffer forward-fills the gap.
            sp_labels = np.asarray((my_view.get("labels") or {}).get("species", -1))
            if not (sp_labels >= 0).any():
                return None
            mb = {
                "entities": _t1(mirror_frag["entities"], torch.float32, device),
                "entity_ids": _t1(mirror_frag["entity_ids"], torch.long, device),
                "field": _t1(mirror_frag["field"], torch.float32, device),
                "history": _t1(mirror_frag["history"], torch.float32, device),
                "history_ids": _t1(mirror_frag["history_ids"], torch.long, device),
                "history_state": _t1(mirror_frag["history_state"], torch.float32, device),
                "history_len": torch.as_tensor(
                    [mirror_frag["history_len"]], dtype=torch.long, device=device
                ),
                "active_idx": _t1(mirror_frag["active_idx"], torch.long, device),
            }
            mv = {
                "tokens": _t1(my_view["tokens"], torch.long, device),
                "masks": {k: _t1(v, torch.bool, device) for k, v in my_view["masks"].items()},
                "labels": {k: _t1(v, torch.long, device) for k, v in my_view["labels"].items()},
            }
            ce = self.phi_hat.leakage_ce(mb, mv, family_weights=self.leak_family_weights)
            return float(ce[0])
        except Exception as exc:  # noqa: BLE001 - optional metric boundary; error is recorded
            self._record_diagnostic_error("leakage_measurement", exc)
            return None  # leakage measurement is never battle-critical

    def _final_leak_ce(self, battle: Any, ledger: Any, tracker: Any) -> float | None:
        """Post-game readability CE — the `after` term for the LAST decision's ΔI (§7)."""
        if self.phi_hat is None:
            return None
        try:
            from fog.encoding.observation import Observation

            obs = Observation.from_battle(battle, tracker=tracker)
            mirror = self._mirror(battle, obs, ledger)
            return self._leakage_ce(mirror, obs, _agent_device(self.agent))
        except Exception as exc:  # noqa: BLE001 - optional final metric boundary
            self._record_diagnostic_error("final_leakage_measurement", exc)
            return None

    # -------------------------------------------------------------------------------- battle hooks
    def _feed_events(self, battle: Any, ledger: Any, tracker: Any, fed: set[int]) -> None:
        """Feed newly completed turns' events into the ledger + tracker (idempotent per turn)."""
        from fog.constants.tables import TABLES
        from fog.encoding.events import feed_ledger_from_events
        from fog.encoding.history import observations_by_turn

        role = getattr(battle, "player_role", None) or "p1"
        obs_map = observations_by_turn(battle)
        current = int(getattr(battle, "turn", 0) or 0)
        for t in sorted(obs_map):
            if t < current and t not in fed:
                events = getattr(obs_map[t], "events", None) or []
                if ledger is not None:
                    feed_ledger_from_events(events, role, ledger, TABLES)
                tracker.feed_turn(t, events)
                fed.add(t)
        # Already-observed pre-turn/mid-turn events belong in the current state and reveal
        # ledger immediately. The tracker consumes only the newly appended suffix. This does
        # not make the turn part of encode_history, which still requires t < current.
        current_obs = obs_map.get(current)
        if current_obs is not None:
            events = getattr(current_obs, "events", None) or []
            if ledger is not None:
                feed_ledger_from_events(events, role, ledger, TABLES)
            tracker.feed_turn(current, events)

    def _battle_finished(self, battle: Any) -> None:
        tag = str(getattr(battle, "battle_tag", "battle"))
        ledger = self._ledgers.get(tag)
        tracker = self._trackers.get(tag)
        if self._buffer is not None:
            gid = self._game_ids.pop(tag, None)
            if gid is not None:
                # drain any final-turn events so the last measurement sees the full game
                fed = self._fed_turns.get(tag, set())
                if tracker is not None:
                    from fog.encoding.history import observations_by_turn

                    obs_map = observations_by_turn(battle)
                    for t in sorted(obs_map):
                        if t not in fed:
                            tracker.feed_turn(t, getattr(obs_map[t], "events", None) or [])
                            fed.add(t)
                won = getattr(battle, "won", None)
                outcome = 0.0 if won is None else (1.0 if won else -1.0)
                truth = None
                if self.truth_provider is not None:
                    truth = self.truth_provider(battle)
                self._buffer.end_game(
                    gid, outcome=outcome, belief_targets=truth,
                    final_leak_ce=self._final_leak_ce(battle, ledger, tracker),
                    final_progress_score=_progress_score(battle),
                )
        if self.on_battle_end is not None:
            self.on_battle_end(battle)
        # drop this battle's state (other concurrent battles keep theirs)
        self._ledgers.pop(tag, None)
        self._trackers.pop(tag, None)
        self._fed_turns.pop(tag, None)
        self._announced_battles.discard(tag)


def unique_account(prefix: str = "fog"):
    """A collision-proof poke-env AccountConfiguration (<=18 chars, Showdown's limit).

    poke-env auto-numbers usernames only within one process; concurrent/successive processes (or a
    zombie holding its socket) collide on the server and battles hang on |nametaken|. Always pass
    one of these per player in scripts and training."""
    import os
    import random

    from poke_env import AccountConfiguration

    name = f"{prefix}{os.getpid() % 10000}x{random.randrange(10000)}"
    return AccountConfiguration(name[:18], None)


def _species_id(mon) -> str:
    from fog.encoding.events import submitted_species_id

    return submitted_species_id(getattr(mon, "species", ""))


def _own_truth_tokens(battle, tables):
    """My own full team as [6,15] truth tokens for self-belief labels (§3.3)."""

    from fog.constants.schema import TYPE_INDEX
    from fog.encoding.events import submitted_species_id, to_id
    from fog.encoding.reveals import SlotView, tokens_from_views

    views = []
    for mon in list((getattr(battle, "team", None) or {}).values())[:6]:
        v = SlotView()
        sp = getattr(mon, "species", None)
        if sp:
            v.species = tables.species_index(submitted_species_id(sp))
        item = getattr(mon, "item", None)
        if item and to_id(item) != "unknownitem":
            v.item = tables.item_index(to_id(item))
        ability = getattr(mon, "ability", None)
        if ability:
            v.ability = tables.ability_index(to_id(ability), default=None)
        for mid in list(getattr(mon, "moves", None) or {})[:4]:
            idx = tables.move_index(to_id(mid), default=None)
            if idx is not None and idx not in v.moves:
                v.moves.append(idx)
        if sp:
            from fog.encoding.entities import _own_nature_evs

            v.nature, v.evs = _own_nature_evs(mon, to_id(sp), tables)
        tt = getattr(mon, "tera_type", None)
        if tt is not None:
            v.tera = TYPE_INDEX.get(to_id(getattr(tt, "name", tt)), 18)
        views.append(v)
    while len(views) < 6:
        views.append(SlotView())
    return tokens_from_views(views)


def _zero_mirror(obs) -> tuple[dict, dict]:
    """Structure-preserving mirror fallback: zero entities, -1 tokens/labels (no loss contribution).

    Masks are omitted entirely — my_view["masks"] = {} stacks consistently only if EVERY frag uses
    {}; to avoid a mixed batch the fallback keeps the same mask keys as the real path but all-True.
    """
    import numpy as np

    from fog.constants.schema import SET_N_ATTRS
    from fog.constants.tables import TABLES

    frag = {
        "entities": np.zeros_like(obs.entities),
        "entity_ids": np.zeros_like(obs.entity_ids),
        "field": np.zeros_like(obs.field),
        "active_idx": np.array([0, 6], dtype=np.int64),
        # Opponent request legality is private and unavailable in a mirror fallback.  Keep the
        # structure identical to real mirrors with a constant, non-informative public placeholder.
        "legality_mask": np.ones(14, dtype=bool),
    }
    tokens = np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
    masks = {
        "species": np.ones((6, TABLES.n_species), dtype=bool),
        "ability": np.ones((6, TABLES.n_abilities), dtype=bool),
        "move": np.ones((6, TABLES.n_moves), dtype=bool),
    }
    labels = {
        "species": np.full(6, -1, dtype=np.int64), "item": np.full(6, -1, dtype=np.int64),
        "ability": np.full(6, -1, dtype=np.int64), "move": np.full((6, 4), -1, dtype=np.int64),
        "nature": np.full(6, -1, dtype=np.int64),
        "ev": np.full((6, 6), -1, dtype=np.int64),
        "tera": np.full(6, -1, dtype=np.int64),
    }
    return frag, {"tokens": tokens, "masks": masks, "labels": labels}


def _agent_device(agent: FogAgent):
    import torch

    try:
        return next(agent.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _t1(a, dtype, device):
    import torch

    t = torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)
    return (t.to(device) if device is not None else t).unsqueeze(0)


def _sample_action(
    log_probs: Any, *, temperature: float, tera_logit_bonus: float = 0.0
) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(log_probs).item())

    finite = torch.isfinite(log_probs)
    if not bool(finite.any()):
        return 13
    probs = _behavior_log_probs(
        log_probs,
        temperature=temperature,
        tera_logit_bonus=tera_logit_bonus,
    ).exp()
    if not bool(torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0:
        legal = torch.nonzero(finite, as_tuple=True)[0]
        return int(legal[0].item())
    return int(torch.multinomial(probs, 1).item())


def _temperature_log_probs(log_probs: Any, *, temperature: float):
    """Return the exact normalized distribution sampled at a positive temperature.

    ``log_probs`` may contain ``-inf`` for illegal actions.  Keeping the transformation in one
    helper lets rollout collection and PPO recompute byte-for-byte equivalent likelihoods.
    Temperature zero is deterministic evaluation and has no differentiable sampling
    distribution, so callers must handle it separately.
    """
    return _behavior_log_probs(log_probs, temperature=temperature)


def _behavior_log_probs(
    log_probs: Any, *, temperature: float = 1.0, tera_logit_bonus: float = 0.0
):
    """Exact normalized training policy after optional temperature and Tera-only exploration.

    A fixed Tera bonus is deliberately narrower than global temperature: it changes only the
    relative mass of legal slots 4..7 versus all other legal actions, while preserving the policy's
    learned preferences within each group.  The same transformation is used for acting and PPO's
    update-time likelihood, so the importance ratio starts at one.
    """
    import torch

    value = float(temperature)
    bonus = float(tera_logit_bonus)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(bonus) or bonus < 0:
        raise ValueError("tera_logit_bonus must be finite and non-negative")
    if value == 1.0 and bonus == 0.0:
        return log_probs
    scaled = log_probs / value
    if bonus:
        scaled = scaled.clone()
        scaled[..., 4:8] = scaled[..., 4:8] + bonus
    return torch.log_softmax(scaled, dim=-1)
