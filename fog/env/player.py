"""Live battle decisions, rollout collection, and poke-env lifecycle hooks.

The poke-env subclass is created by ``FogPlayer.build()``. Battle trackers and rollout game IDs
are kept per battle tag so one player can handle concurrent games. Opponent-view recording and
frozen-probe measurements live in ``MirrorRecorder``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from fog.env.actions import active_entity_indices, map_action_to_order
from fog.env.mirror_recording import MirrorRecorder
from fog.env.mirror_recording import _own_truth_tokens as _own_truth_tokens
from fog.env.mirror_recording import _species_id as _species_id
from fog.env.mirror_recording import _t1 as _t1
from fog.env.mirror_recording import _zero_mirror as _zero_mirror
from fog.training.progress_reward import battle_progress_score as _progress_score

if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle

    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer


class FogPlayer:
    """Coordinate agent decisions, battle state, and training hooks for poke-env."""

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
        # Distillation hooks receive the observation and policy output used for this action.
        self.decision_observer: Callable[[str, dict, Any], None] | None = None
        # Overrides must return a legal action index.
        self.action_override: Callable[[Any, Any, Any], int] | None = None
        # DAgger labels accompany the rollout row and are discarded with rejected choices.
        self.supervised_label_provider: Callable[[Any, Any, Any], int] | None = None
        # Retain the donor's masked policy distribution for supervised training.
        self.record_policy_reference = False
        # optional: finished battle -> [6,15] opponent truth tokens (self-play belief supervision)
        self.truth_provider: Callable[[Any], Any] | None = None
        # Optional immutable submitted private-state truth for the acting player's own team.
        # Direct-Showdown training has no poke-env team builder, so reading the live battle objects
        # here would incorrectly turn temporary in-battle changes (Skill Swap, Transform, etc.) into
        # self-belief labels.
        self.private_state_truth_provider: Callable[[Any], Any] | None = None
        self._mirror_recorder = MirrorRecorder(self._record_diagnostic_error)
        # Set-oracle truth injection: battle -> [6,15] opponent truth tokens fed
        # to the belief net as opp_truth_tokens (perfect-information ceiling).
        self.oracle_truth_provider: Callable[[Any], Any] | None = None
        # Optional determinized search at action time: when set, act() selects via
        # search.determinized instead of sampling the policy. Reconstructs an engine position from
        # the live battle + belief-sampled worlds and rolls one ply each (search/engine_step). Dict:
        # {"k": int, "prior_weight": float, "seed": int}. Inference-only; never during training.
        self.search: dict | None = None
        # Evaluations read search accounting through the built player's `_fog_adapter` reference.
        self.search_stats: dict[str, Any] = {
            "decisions": 0,
            "simulation_attempts": 0,
            "simulation_failures": 0,
            "fallbacks": 0,
            "overrides": 0,
            "last_error": None,
            "failure_reasons": {},
        }
        # Retain errors from optional subsystems for run reports.
        self.diagnostic_errors: dict[str, dict[str, Any]] = {}

    @property
    def phi_hat(self):
        return self._mirror_recorder.probe

    @phi_hat.setter
    def phi_hat(self, probe) -> None:
        self._mirror_recorder.probe = probe

    @property
    def leak_family_weights(self) -> dict[str, float] | None:
        return self._mirror_recorder.family_weights

    @leak_family_weights.setter
    def leak_family_weights(self, weights: dict[str, float] | None) -> None:
        self._mirror_recorder.family_weights = weights

    @property
    def record_mirror(self) -> bool | None:
        """None tries once; True keeps batch structure after failures; False disables recording."""
        return self._mirror_recorder.enabled

    @record_mirror.setter
    def record_mirror(self, enabled: bool | None) -> None:
        self._mirror_recorder.enabled = enabled

    @property
    def mirror_last_error(self) -> str | None:
        return self._mirror_recorder.last_error

    @mirror_last_error.setter
    def mirror_last_error(self, error: str | None) -> None:
        self._mirror_recorder.last_error = error

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

        # poke-env uses this class name in default usernames. Showdown strips punctuation,
        # which can break the challenge handshake; use unique_account() for explicit accounts.
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
            # setdefault would call start_game() even when the battle already has a game ID.
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
                frag["behavior_log_probs"] = recorded_log_probs.detach().cpu().numpy()
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
                    # PPO recomputes the acting policy with the same oracle input.
                    # Fill missing truth so fragments retain the same keys.
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

        # Save the public state and chosen action for history encoding.
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
        """Order my team with the agent's preview head, falling back to the default order."""
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
        """Select by search, counting failures before falling back to policy sampling."""
        import torch

        self.search_stats["decisions"] += 1
        stepper = None
        stepper_failures_accounted = False
        try:
            cfg = self.search or {}
            if cfg.get("mode") == "direct_two_sided":
                import random

                from fog.search.direct_two_sided import direct_two_sided_action

                if not hasattr(self, "_direct_search_rng"):
                    self._direct_search_rng = random.Random(int(cfg.get("seed", 0)))
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
                    rng=self._direct_search_rng,
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
        return self._mirror_recorder.build(
            battle,
            obs,
            ledger,
            recording=self._buffer is not None,
            private_truth_provider=self.private_state_truth_provider,
            team_builder=self._team_builder,
        )

    def _leakage_ce(self, mirror: tuple[dict, dict] | None, obs: Any, device) -> float | None:
        return self._mirror_recorder.leakage_ce(mirror, device)

    def _final_leak_ce(self, battle: Any, ledger: Any, tracker: Any) -> float | None:
        """Post-game readability CE for the last decision's leakage penalty."""
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
    """A poke-env account with a process/random suffix, within Showdown's 18-character limit.

    poke-env's automatic numbering is local to each process; reused names can hang challenges.
    """
    import os
    import random

    from poke_env import AccountConfiguration

    name = f"{prefix}{os.getpid() % 10000}x{random.randrange(10000)}"
    return AccountConfiguration(name[:18], None)


def _agent_device(agent: FogAgent):
    import torch

    try:
        return next(agent.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sample_action(
    log_probs: Any, *, temperature: float, tera_logit_bonus: float = 0.0
) -> int:
    import torch

    if temperature <= 0:
        bonus = float(tera_logit_bonus)
        if not np.isfinite(bonus) or bonus < 0:
            raise ValueError("tera_logit_bonus must be finite and non-negative")
        scores = log_probs
        if bonus:
            scores = log_probs.clone()
            scores[..., 4:8] += bonus  # -inf legal masks remain impossible.
        return int(torch.argmax(scores).item())

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
    """Normalized positive-temperature policy used by acting and PPO.

    Illegal actions keep -inf. Callers handle deterministic temperature-zero selection separately.
    """
    return _behavior_log_probs(log_probs, temperature=temperature)


def _behavior_log_probs(
    log_probs: Any, *, temperature: float = 1.0, tera_logit_bonus: float = 0.0
):
    """Normalize temperature-adjusted logits with an optional bonus for legal Tera actions.

    Acting and PPO use the same transformation so the initial importance ratio is one.
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
