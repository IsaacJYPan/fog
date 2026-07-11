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

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from fog.env.actions import active_entity_indices, map_action_to_order

if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle

    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer


class FogPlayer:
    """Thin adapter from `FogAgent` decisions to poke-env battle orders."""

    def __init__(self, agent: "FogAgent | None" = None, *, temperature: float = 1.0) -> None:
        self.agent = agent
        self.temperature = temperature
        self._player_ref: Any | None = None
        self._buffer: "RolloutBuffer | None" = None
        self._game_ids: dict[str, int] = {}
        # per-battle state, keyed by battle_tag (safe under max_concurrent_battles > 1)
        self._ledgers: dict[str, Any] = {}
        self._trackers: dict[str, Any] = {}
        self._fed_turns: dict[str, set[int]] = {}
        self.on_battle_end: Callable[[Any], None] | None = None
        # optional: finished battle -> [6,9] opponent truth tokens (self-play belief supervision)
        self.truth_provider: Callable[[Any], Any] | None = None
        # optional leakage measurement (§7): a FrozenProbe (frozen backbone + self-belief). When set,
        # each decision records the probe's self-readability CE so the rollout buffer can derive the
        # per-turn ΔI penalty. `leak_family_weights` rescales per-family CE (weighted-ΔI ablation).
        self.phi_hat = None
        self.leak_family_weights: dict[str, float] | None = None
        # mirror recording for self-belief TRAINING (§3.3): tri-state — None = auto (try once,
        # disable permanently on first failure), True/False = forced. Only active with a buffer.
        self.record_mirror: bool | None = None
        # optional set-oracle truth injection (§3.4 eval): battle -> [6,9] opponent truth tokens fed
        # to the belief net as opp_truth_tokens (perfect-information ceiling).
        self.oracle_truth_provider: Callable[[Any], Any] | None = None
        # optional determinized search at action time (§8): when set, act() selects via
        # search.determinized instead of sampling the policy. Reconstructs an engine position from
        # the live battle + belief-sampled worlds and rolls one ply each (search/engine_step). Dict:
        # {"k": int, "prior_weight": float, "seed": int}. Inference-only; never during training.
        self.search: dict | None = None

    def attach_buffer(self, buffer: "RolloutBuffer") -> None:
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

        self._player_ref = FogBot(**poke_env_kwargs)
        return self._player_ref

    # ---------------------------------------------------------------------------- per-battle state
    def _battle_state(self, battle: Any) -> tuple[str, Any, Any, set[int]]:
        """(tag, ledger, tracker, fed_turns) for this battle, creating them on first sight."""
        tag = str(getattr(battle, "battle_tag", "battle"))
        if tag not in self._trackers:
            from fog.encoding.tracker import BattleTracker

            role = getattr(battle, "player_role", None) or "p1"
            self._trackers[tag] = BattleTracker(my_role=role)
            self._fed_turns[tag] = set()
            try:
                from fog.encoding.reveals import RevealLedger

                self._ledgers[tag] = RevealLedger()
            except Exception:  # pragma: no cover
                self._ledgers[tag] = None
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
    def act(self, battle: "AbstractBattle"):
        """Build an observation, sample an agent action, and return the corresponding BattleOrder."""
        player = self._player_ref
        if player is None:
            raise RuntimeError("FogPlayer.act requires a built poke-env Player instance.")
        if self.agent is None:
            return player.choose_default_move()

        import torch

        from fog.encoding.observation import Observation

        tag, ledger, tracker, fed = self._battle_state(battle)
        self._feed_events(battle, ledger, tracker, fed)
        obs = Observation.from_battle(battle, tracker=tracker)
        obs.validate()

        device = _agent_device(self.agent)
        active_idx = (
            obs.active_idx
            if obs.active_idx is not None
            else np.asarray(active_entity_indices(battle), dtype=np.int64)
        )
        batch = {
            "entities": torch.as_tensor(obs.entities, dtype=torch.float32, device=device).unsqueeze(0),
            "entity_ids": torch.as_tensor(obs.entity_ids, dtype=torch.long, device=device).unsqueeze(0),
            "field": torch.as_tensor(obs.field, dtype=torch.float32, device=device).unsqueeze(0),
            "history": torch.as_tensor(obs.history, dtype=torch.float32, device=device).unsqueeze(0),
            "active_idx": torch.as_tensor(active_idx, dtype=torch.long, device=device).unsqueeze(0),
            "legality_mask": torch.as_tensor(
                obs.legality_mask, dtype=torch.bool, device=device
            ).unsqueeze(0),
            "opp_revealed": _opp_revealed_to_torch(obs.opp_revealed, device=device),
        }
        oracle_truth = None
        if self.oracle_truth_provider is not None:
            try:
                oracle_truth = self.oracle_truth_provider(battle)
            except Exception:
                oracle_truth = None
            if oracle_truth is not None:
                oracle_truth = np.asarray(oracle_truth, dtype=np.int64)
                batch["opp_truth_tokens"] = torch.as_tensor(
                    oracle_truth, dtype=torch.long, device=device
                ).unsqueeze(0)

        with torch.no_grad():
            out = self.agent(batch)
        if self.search is not None:
            action_idx = self._search_action(battle, obs, tracker, batch, device)
        else:
            action_idx = _sample_action(out.log_probs[0], temperature=self.temperature)

        if self._buffer is not None:
            gid = self._game_id(tag)
            frag = {
                "entities": obs.entities,
                "entity_ids": obs.entity_ids,
                "field": obs.field,
                "history": obs.history,
                "active_idx": active_idx,
                "legality_mask": obs.legality_mask,
                "opp_revealed": obs.opp_revealed,
                "turn": np.int64(getattr(battle, "turn", 0) or 0),
            }
            if self.oracle_truth_provider is not None:
                # The update-time forward must see what the acting forward saw (acting with truth
                # but updating without silently breaks the PPO ratio). Key always present so frag
                # structure is batch-constant; -1 rows merge to "no truth" in apply_set_oracle.
                frag["opp_truth_tokens"] = (
                    oracle_truth if oracle_truth is not None
                    else np.full((6, 9), -1, dtype=np.int64)
                )
            mirror = self._mirror(battle, obs, ledger)
            if mirror is not None:
                mirror_frag, my_view = mirror
                frag["opp_view"] = {**mirror_frag, "my_view": my_view}
            self._buffer.add_decision(
                gid,
                frag,
                action=action_idx,
                log_prob=float(out.log_probs[0, action_idx]),
                value=float(out.value[0]),
                leak_ce=self._leakage_ce(mirror, obs, device),
            )

        # history retro-fidelity: snapshot + my chosen action for this turn (§2.3)
        try:
            from fog.constants.tables import TABLES
            from fog.encoding.events import build_snapshot

            tracker.record_decision(
                int(getattr(battle, "turn", 0) or 0), build_snapshot(battle, TABLES), action_idx
            )
        except Exception:
            pass

        return map_action_to_order(battle, action_idx, player)

    # ------------------------------------------------------------------------------- team preview
    def choose_teampreview(self, battle: Any) -> str:
        """Order my team with the agent's TeamPreviewHead (§3.1). Falls back to default order."""
        if self.agent is None or getattr(self.agent, "preview", None) is None:
            return "/team 123456"
        try:
            import torch

            from fog.encoding.observation import Observation

            tag, _, tracker, _ = self._battle_state(battle)
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
            if self.temperature <= 0:
                lead = int(logits.argmax())
            else:
                probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
                lead = int(torch.multinomial(probs, 1))
            log_prob = float(torch.log_softmax(logits, dim=-1)[lead])
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
        except Exception:
            return "/team 123456"

    # -------------------------------------------------------------------------------------- search
    def _search_action(self, battle: Any, obs: Any, tracker: Any, batch: dict, device) -> int:
        """Determinized action via engine rollouts reconstructed from the live battle (§8). Any
        failure (engine missing, reconstruction/step error) falls back to policy sampling so a
        search run never crashes a game."""
        import torch

        try:
            from fog.search.determinized import determinized_action
            from fog.search.engine_step import ShowdownWorldStepper

            cfg = self.search or {}
            role = getattr(battle, "player_role", None) or "p1"
            stepper = ShowdownWorldStepper(
                battle, tracker, role=role, seed=int(cfg.get("seed", 0)),
            )
            action = determinized_action(
                self.agent, batch, step_fn=stepper,
                k=int(cfg.get("k", 16)), prior_weight=float(cfg.get("prior_weight", 0.5)),
            )
            # if every world's reconstruction failed, search saw nothing — fall back to the policy
            if stepper.failures and not stepper._bases:
                raise RuntimeError("all search reconstructions failed")
            return action
        except Exception:
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
            my_truth_tokens = _own_truth_tokens(battle, TABLES)
            out = build_opponent_view_np(obs, my_view_tokens, my_truth_tokens, TABLES)
            if self.record_mirror is None:
                self.record_mirror = True
            return out
        except Exception:
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
                "history": _t1(obs.history, torch.float32, device),
                "active_idx": _t1(mirror_frag["active_idx"], torch.long, device),
            }
            mv = {
                "tokens": _t1(my_view["tokens"], torch.long, device),
                "masks": {k: _t1(v, torch.bool, device) for k, v in my_view["masks"].items()},
                "labels": {k: _t1(v, torch.long, device) for k, v in my_view["labels"].items()},
            }
            ce = self.phi_hat.leakage_ce(mb, mv, family_weights=self.leak_family_weights)
            return float(ce[0])
        except Exception:
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
        except Exception:
            return None

    # -------------------------------------------------------------------------------- battle hooks
    def _feed_events(self, battle: Any, ledger: Any, tracker: Any, fed: set[int]) -> None:
        """Feed newly completed turns' events into the ledger + tracker (idempotent per turn)."""
        try:
            from fog.constants.tables import TABLES
            from fog.encoding.events import feed_ledger_from_events

            role = getattr(battle, "player_role", None) or "p1"
            obs_map = getattr(battle, "observations", None) or {}
            current = int(getattr(battle, "turn", 0) or 0)
            for t in sorted(obs_map):
                if t < current and t not in fed:
                    events = getattr(obs_map[t], "events", None) or []
                    if ledger is not None:
                        feed_ledger_from_events(events, role, ledger, TABLES)
                    tracker.feed_turn(t, events)
                    fed.add(t)
        except Exception:
            pass  # event tracking is data collection, never battle-critical

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
                    try:
                        obs_map = getattr(battle, "observations", None) or {}
                        for t in sorted(obs_map):
                            if t not in fed:
                                tracker.feed_turn(t, getattr(obs_map[t], "events", None) or [])
                                fed.add(t)
                    except Exception:
                        pass
                won = getattr(battle, "won", None)
                outcome = 0.0 if won is None else (1.0 if won else -1.0)
                truth = None
                if self.truth_provider is not None:
                    try:
                        truth = self.truth_provider(battle)
                    except Exception:
                        truth = None
                self._buffer.end_game(
                    gid, outcome=outcome, belief_targets=truth,
                    final_leak_ce=self._final_leak_ce(battle, ledger, tracker),
                )
        if self.on_battle_end is not None:
            self.on_battle_end(battle)
        # drop this battle's state (other concurrent battles keep theirs)
        self._ledgers.pop(tag, None)
        self._trackers.pop(tag, None)
        self._fed_turns.pop(tag, None)


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
    import re

    return re.sub(r"[^a-z0-9]", "", str(getattr(mon, "species", "")).lower())


def _own_truth_tokens(battle, tables):
    """My own full team as [6,9] truth tokens (I always know my team) for self-belief labels (§3.3)."""

    from fog.constants.schema import TYPE_INDEX
    from fog.encoding.reveals import SlotView, tokens_from_views

    def _id(x):
        import re

        return re.sub(r"[^a-z0-9]", "", str(x).lower())

    views = []
    for mon in list((getattr(battle, "team", None) or {}).values())[:6]:
        v = SlotView()
        sp = getattr(mon, "species", None)
        if sp:
            v.species = tables.species_index(_id(sp))
        item = getattr(mon, "item", None)
        if item and _id(item) != "unknownitem":
            v.item = tables.item_index(_id(item))
        ability = getattr(mon, "ability", None)
        if ability:
            v.ability = tables.ability_index(_id(ability), default=None)
        for mid in list((getattr(mon, "moves", None) or {}))[:4]:
            idx = tables.move_index(_id(mid), default=None)
            if idx is not None and idx not in v.moves:
                v.moves.append(idx)
        tt = getattr(mon, "tera_type", None)
        if tt is not None:
            v.tera = TYPE_INDEX.get(_id(getattr(tt, "name", tt)), 18)
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

    from fog.constants.schema import MAX_SPREADS_PER_SPECIES
    from fog.constants.tables import TABLES

    frag = {
        "entities": np.zeros_like(obs.entities),
        "entity_ids": np.zeros_like(obs.entity_ids),
        "field": np.zeros_like(obs.field),
        "active_idx": np.array([0, 6], dtype=np.int64),
    }
    tokens = np.full((6, 9), -1, dtype=np.int64)
    masks = {
        "species": np.ones((6, TABLES.n_species), dtype=bool),
        "ability": np.ones((6, TABLES.n_abilities), dtype=bool),
        "move": np.ones((6, TABLES.n_moves), dtype=bool),
        "spread": np.ones((6, MAX_SPREADS_PER_SPECIES + 1), dtype=bool),
    }
    labels = {
        "species": np.full(6, -1, dtype=np.int64), "item": np.full(6, -1, dtype=np.int64),
        "ability": np.full(6, -1, dtype=np.int64), "move": np.full((6, 4), -1, dtype=np.int64),
        "spread": np.full(6, -1, dtype=np.int64), "tera": np.full(6, -1, dtype=np.int64),
    }
    return frag, {"tokens": tokens, "masks": masks, "labels": labels}


def _agent_device(agent: "FogAgent"):
    import torch

    try:
        return next(agent.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _t1(a, dtype, device):
    import torch

    t = torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)
    return (t.to(device) if device is not None else t).unsqueeze(0)


def _opp_revealed_to_torch(opp_revealed: dict[str, Any] | None, *, device: Any) -> dict[str, Any]:
    import torch

    if opp_revealed is None:
        return {
            "tokens": torch.full((1, 6, 9), -1, dtype=torch.long, device=device),
            "masks": {},
        }

    tokens = torch.as_tensor(np.asarray(opp_revealed["tokens"]), dtype=torch.long, device=device)
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)

    masks: dict[str, Any] = {}
    for name, mask in (opp_revealed.get("masks") or {}).items():
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=device)
        masks[name] = m.unsqueeze(0) if m.ndim == 2 else m
    return {"tokens": tokens, "masks": masks}


def _sample_action(log_probs: Any, *, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(log_probs).item())

    finite = torch.isfinite(log_probs)
    if not bool(finite.any()):
        return 13
    scaled = log_probs if temperature == 1.0 else log_probs / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    if not bool(torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0:
        legal = torch.nonzero(finite, as_tuple=True)[0]
        return int(legal[0].item())
    return int(torch.multinomial(probs, 1).item())
