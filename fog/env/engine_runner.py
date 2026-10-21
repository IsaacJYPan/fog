"""Batched lockstep self-play on poke-engine.

Agent decisions across active EngineBattles share a batched forward pass. Recording
uses the same rollout format and opponent-view helpers as FogPlayer; submitted
pastes provide exact belief labels. Scripted opponents act on engine state, while
frozen-checkpoint opponents use a separate batched forward pass.

Probe throughput with:

    python -m fog.env.engine_runner --probe --pool data/teams/pool_a --config configs/default.yaml
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fog.encoding.batch import collate_observations
from fog.training.progress_reward import battle_progress_score as _progress_score

if TYPE_CHECKING:
    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

LEARNER = "p1"
OPPONENT = "p2"


class FrozenAgentPolicy:
    """A frozen FogAgent as the engine opponent (league checkpoint). Batched, never recorded."""

    def __init__(self, agent: FogAgent, name: str = "frozen", temperature: float = 1.0):
        agent.eval()
        self.agent = agent
        self.name = name
        self.temperature = temperature


@dataclass
class _Slot:
    battle: EngineBattle
    gid: int | None = None
    decisions: int = 0
    stalled_turns: int = field(default=0)


class EngineRunner:
    """Lockstep batch of engine battles: learner is always side p1, opponent side p2."""

    def __init__(
        self,
        agent: FogAgent,
        buffer: RolloutBuffer | None = None,
        *,
        pool: list[str],
        batch_size: int = 64,
        seed: int = 0,
        temperature: float = 1.0,
        tera_logit_bonus: float = 0.0,
        phi_hat=None,
        leak_family_weights: dict | None = None,
        oracle_truth: bool = False,
        tables=None,
        max_batch_forward: int = 256,
    ) -> None:
        import random

        from fog.env.player import FogPlayer

        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        self.tables = tables
        self.agent = agent
        self.buffer = buffer
        self.pool = list(pool)
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.tera_logit_bonus = float(tera_logit_bonus)
        self.oracle_truth = bool(oracle_truth)
        self.max_batch_forward = int(max_batch_forward)
        self.rng = random.Random(seed)
        self._battle_count = 0
        self._truth_cache: dict[str, np.ndarray] = {}

        # FogPlayer as a stateless helper: one shared implementation of the mirror/leakage
        # measurement for both backends (it never touches its poke-env player here).
        self._helper = FogPlayer(agent, temperature=temperature)
        if buffer is not None:
            self._helper.attach_buffer(buffer)
        self._helper.phi_hat = phi_hat
        self._helper.leak_family_weights = leak_family_weights

    # ------------------------------------------------------------------------------ battle lifecycle
    def _new_slot(self, opponent) -> _Slot:
        from fog.env.engine_battle import EngineBattle

        self._battle_count += 1
        my_paste = self.rng.choice(self.pool)
        opp_paste = self.rng.choice(self.pool)
        battle = EngineBattle(
            my_paste, opp_paste,
            seed=self.rng.randrange(2**31),
            tables=self.tables,
            battle_tag=f"engine-{self._battle_count}",
        )
        slot = _Slot(battle=battle)
        my_lead = self._preview(slot)
        opp_lead = self._opponent_lead(battle, opponent)
        battle.set_leads(my_lead, opp_lead)
        return slot

    def _preview(self, slot: _Slot) -> int:
        """Learner team-preview lead pick — mirrors FogPlayer.choose_teampreview.
        Unbatched on purpose: one preview per ~30 decisions is off the critical path."""
        import torch

        from fog.encoding.observation import Observation

        if self.agent is None or getattr(self.agent, "preview", None) is None:
            return 0
        battle = slot.battle
        obs = Observation.from_battle(battle.view(LEARNER), tracker=battle.trackers[LEARNER])
        device = _device(self.agent)
        pre = {
            "entities": _t(obs.entities, torch.float32, device).unsqueeze(0),
            "entity_ids": _t(obs.entity_ids, torch.long, device).unsqueeze(0),
            "field": _t(obs.field, torch.float32, device).unsqueeze(0),
        }
        with torch.no_grad():
            logits = self.agent.preview_logits(pre)[0]
        n_mons = len(battle.paste_species[LEARNER]) or 6
        logits = logits[:n_mons]
        if self.temperature <= 0:
            lead = int(logits.argmax())
        else:
            lead = int(torch.multinomial(
                torch.softmax(logits / max(self.temperature, 1e-6), dim=-1), 1
            ))
        if self.buffer is not None:
            slot.gid = self.buffer.start_game()
            self.buffer.add_preview(
                slot.gid,
                {"entities": obs.entities, "entity_ids": obs.entity_ids, "field": obs.field},
                slot=lead,
                log_prob=float(torch.log_softmax(logits, dim=-1)[lead]),
            )
        return lead

    def _opponent_lead(self, battle, opponent) -> int:
        if isinstance(opponent, FrozenAgentPolicy):
            import torch

            from fog.encoding.observation import Observation

            obs = Observation.from_battle(battle.view(OPPONENT), tracker=battle.trackers[OPPONENT])
            device = _device(opponent.agent)
            pre = {
                "entities": _t(obs.entities, torch.float32, device).unsqueeze(0),
                "entity_ids": _t(obs.entity_ids, torch.long, device).unsqueeze(0),
                "field": _t(obs.field, torch.float32, device).unsqueeze(0),
            }
            with torch.no_grad():
                logits = opponent.agent.preview_logits(pre)[0][: len(battle.paste_species[OPPONENT]) or 6]
            return int(logits.argmax())
        return 0  # scripted poke-env baselines lead slot 0 ("/team 123456")

    def _finish(self, slot: _Slot, stats: dict) -> None:
        battle = slot.battle
        stats["games"] += 1
        won = battle.winner
        if won == LEARNER:
            stats["wins"] += 1
        if self.buffer is not None and slot.gid is not None:
            outcome = 0.0 if won is None else (1.0 if won == LEARNER else -1.0)
            truth = self._truth(battle)
            self.buffer.end_game(
                slot.gid,
                outcome=outcome,
                belief_targets=truth,
                final_leak_ce=self._final_leak_ce(battle),
                final_progress_score=_progress_score(battle, LEARNER),
            )

    def _truth(self, battle) -> np.ndarray | None:
        paste = battle._pastes[OPPONENT]
        tok = self._truth_cache.get(paste)
        if tok is None:
            tok = self._truth_cache[paste] = battle.truth_tokens(OPPONENT)
        return tok.copy()

    def _final_leak_ce(self, battle) -> float | None:
        if self._helper.phi_hat is None:
            return None
        try:
            from fog.encoding.observation import Observation

            view = battle.view(LEARNER)
            obs = Observation.from_battle(view, tracker=battle.trackers[LEARNER])
            mirror = self._helper._mirror(view, obs, battle.ledgers[LEARNER])
            return self._helper._leakage_ce(mirror, obs, _device(self.agent))
        except Exception as exc:  # noqa: BLE001 - optional metric boundary; recorded by helper
            self._helper._record_diagnostic_error("engine_final_leakage", exc)
            return None

    # -------------------------------------------------------------------------------- decision waves
    def _agent_wave(self, agent, items: list[tuple[_Slot, str]], *, record: bool,
                    temperature: float) -> list[int]:
        """One batched forward for [(slot, role)] decisions; returns per-item action indices."""
        import torch

        from fog.constants.tables import TABLES
        from fog.encoding.events import record_history_decision
        from fog.encoding.observation import Observation
        from fog.env.player import _behavior_log_probs, _sample_action

        if not items:
            return []
        if len(items) > self.max_batch_forward:  # bound peak memory on huge batches
            head = self._agent_wave(agent, items[: self.max_batch_forward],
                                    record=record, temperature=temperature)
            tail = self._agent_wave(agent, items[self.max_batch_forward:],
                                    record=record, temperature=temperature)
            return head + tail

        device = _device(agent)
        views, obs_list = [], []
        for slot, role in items:
            view = slot.battle.view(role)
            obs = Observation.from_battle(view, tracker=slot.battle.trackers[role])
            views.append(view)
            obs_list.append(obs)

        batch = _stack_obs(obs_list, device)
        oracle_rows: list[np.ndarray | None] = [None] * len(items)
        if record and self.oracle_truth:
            rows = []
            for i, (slot, _) in enumerate(items):
                truth = self._truth(slot.battle)
                oracle_rows[i] = truth
                from fog.constants.schema import SET_N_ATTRS

                rows.append(
                    truth
                    if truth is not None
                    else np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
                )
            batch["opp_truth_tokens"] = torch.as_tensor(
                np.stack(rows), dtype=torch.long, device=device
            )

        with torch.no_grad():
            out = agent(batch)

        actions: list[int] = []
        for i, (slot, role) in enumerate(items):
            action_idx = _sample_action(
                out.log_probs[i],
                temperature=temperature,
                tera_logit_bonus=self.tera_logit_bonus if record else 0.0,
            )
            recorded_log_probs = (
                _behavior_log_probs(
                    out.log_probs[i],
                    temperature=temperature,
                    tera_logit_bonus=self.tera_logit_bonus,
                )
                if temperature > 0
                else out.log_probs[i]
            )
            actions.append(action_idx)
            battle = slot.battle
            obs, view = obs_list[i], views[i]

            if record and self.buffer is not None:
                if slot.gid is None:
                    slot.gid = self.buffer.start_game()
                frag = {
                    "entities": obs.entities,
                    "entity_ids": obs.entity_ids,
                    "field": obs.field,
                    "history": obs.history,
                    "history_ids": obs.history_ids,
                    "history_state": obs.history_state,
                    "history_len": np.int64(obs.history_len),
                    "active_idx": obs.active_idx if obs.active_idx is not None
                    else np.array([0, 6], dtype=np.int64),
                    "legality_mask": obs.legality_mask,
                    "action_features": obs.action_features,
                    "opp_revealed": obs.opp_revealed,
                    "turn": np.int64(battle.turn),
                }
                if self.oracle_truth:
                    frag["opp_truth_tokens"] = (
                        oracle_rows[i] if oracle_rows[i] is not None
                        else np.full((6, SET_N_ATTRS), -1, dtype=np.int64)
                    )
                mirror = self._helper._mirror(view, obs, battle.ledgers[role])
                if mirror is not None:
                    mirror_frag, my_view = mirror
                    frag["opp_view"] = {**mirror_frag, "my_view": my_view}
                self.buffer.add_decision(
                    slot.gid,
                    frag,
                    action=action_idx,
                    log_prob=float(recorded_log_probs[action_idx]),
                    value=float(out.value[i]),
                    leak_ce=self._helper._leakage_ce(mirror, obs, device),
                    progress_score=_progress_score(battle, role),
                )
                slot.decisions += 1

            # History is part of the next observation, so a mapping failure is training-critical.
            record_history_decision(
                battle.trackers[role], battle.turn, view, TABLES, action_idx
            )
        return actions

    # --------------------------------------------------------------------------------- main loop
    def play_games(self, opponent, n_games: int, *, max_decisions: int | None = None) -> dict:
        """Play `n_games` vs `opponent` (engine_scripted policy or FrozenAgentPolicy), keeping up
        to `batch_size` battles in flight. Returns {"games", "wins", "decisions", "turns"}."""
        stats = {"games": 0, "wins": 0, "decisions": 0, "turns": 0}
        to_start = int(n_games)
        slots: list[_Slot] = []
        while to_start > 0 and len(slots) < self.batch_size:
            slots.append(self._new_slot(opponent))
            to_start -= 1

        while slots:
            learner_items: list[tuple[_Slot, str]] = []
            opp_items: list[tuple[_Slot, str]] = []
            for slot in slots:
                for role in slot.battle.pending_sides():
                    (learner_items if role == LEARNER else opp_items).append((slot, role))

            moves: dict[int, dict[str, str]] = {id(s): {} for s in slots}
            for (slot, role), action in zip(
                learner_items,
                self._agent_wave(self.agent, learner_items, record=True,
                                 temperature=self.temperature),
            ):
                moves[id(slot)][role] = slot.battle.engine_action(role, action)
                stats["decisions"] += 1
            if isinstance(opponent, FrozenAgentPolicy):
                for (slot, role), action in zip(
                    opp_items,
                    self._agent_wave(opponent.agent, opp_items, record=False,
                                     temperature=opponent.temperature),
                ):
                    moves[id(slot)][role] = slot.battle.engine_action(role, action)
            else:
                for slot, role in opp_items:
                    moves[id(slot)][role] = slot.battle.engine_action(
                        role, opponent.choose(slot.battle, role)
                    )

            still: list[_Slot] = []
            for slot in slots:
                slot.battle.step(moves[id(slot)])
                if slot.battle.finished:
                    stats["turns"] += slot.battle.turn
                    self._finish(slot, stats)
                    if to_start > 0:
                        still.append(self._new_slot(opponent))
                        to_start -= 1
                else:
                    still.append(slot)
            slots = still
            if max_decisions is not None and stats["decisions"] >= max_decisions:
                # stop STARTING games; ending in-flight games early as draws would corrupt PPO
                # returns, so the ones already running always finish out
                to_start = 0
        return stats


# --------------------------------------------------------------------------------------- helpers
def _device(agent):
    import torch

    try:
        return next(agent.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _t(a, dtype, device):
    import torch

    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def _stack_obs(obs_list, device) -> dict:
    """Compatibility alias; new callers should import ``collate_observations`` directly."""
    return collate_observations(obs_list, device)


# ---------------------------------------------------------------------------- throughput probe
def _probe(args) -> None:
    import time

    import torch

    from fog.config import load_config_with_args
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

    cfg = load_config_with_args(args.config) if args.config else {}
    pool = load_pool(args.pool)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.eval()

    print(f"probe: pool={args.pool} device={device} games/batch-size={args.games}")
    for bs in [int(b) for b in args.batches.split(",")]:
        buf = RolloutBuffer()
        runner = EngineRunner(agent, buf, pool=pool, batch_size=bs, seed=0)
        opponent = EngineMaxDamagePolicy(seed=1)
        t0 = time.monotonic()
        stats = runner.play_games(opponent, args.games)
        dt = time.monotonic() - t0
        print(
            f"  batch={bs:4d}: {stats['decisions'] / dt:8.1f} decisions/s "
            f"| {stats['games'] / dt:6.2f} games/s | {stats['games']} games "
            f"| win-rate {stats['wins'] / max(1, stats['games']):.2f} "
            f"| avg turns {stats['turns'] / max(1, stats['games']):.1f}"
        )


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Engine self-play throughput probe.")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--batches", type=str, default="1,8,32,64,128")
    args = ap.parse_args()
    _probe(args)


if __name__ == "__main__":
    main()
