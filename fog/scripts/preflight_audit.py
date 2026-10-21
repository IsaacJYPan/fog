"""Inspect observations, legality, and belief queries during local Showdown games.

Runs complete games on the pinned in-memory Pokemon Showdown simulator, through the same censored
poke-env views and Observation encoder used by training. Every decision records enough state to
inspect history construction, current board/private-information censorship, legality, and belief
queries. It never constructs an optimizer and verifies that model parameters did not change.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fog.constants.schema import (
    N_EV_UNITS,
    N_NATURES,
    SET_EV_COLS,
    SET_N_ATTRS,
    SET_NATURE_COL,
)
from fog.constants.tables import TABLES
from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    EV_ID_FIELDS,
    NATURE_ID_FIELD,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
    L,
)
from fog.encoding.history import observations_by_turn
from fog.encoding.observation import Observation
from fog.encoding.tracker import BattleTracker
from fog.env.actions import map_action_to_order
from fog.env.showdown_direct import ShowdownDirectClient
from fog.env.showdown_local_runner import LocalOrderFactory, play_local_battle
from fog.env.team_truth import truth_tokens_from_paste
from fog.env.teams import load_pool, to_packed
from fog.models.agent import FogAgent
from fog.scripts.audit_support import AuditRecorder as Recorder
from fog.scripts.audit_support import git_revision as _git_revision
from fog.scripts.audit_support import parameter_digest as _parameter_digest
from fog.training.preflight import source_fingerprint

SCHEMA_VERSION = 5
FAMILIES = ("item", "ability", "move", "nature", "ev", "tera", "species")


class AuditController:
    """Deterministic max-damage-ish controller with an observation audit at every request."""

    def __init__(
        self,
        *,
        game: int,
        side: str,
        own_truth: np.ndarray,
        opponent_truth: np.ndarray,
        agent: FogAgent,
        recorder: Recorder,
        rng: random.Random,
        run_belief_sampling: bool = False,
    ) -> None:
        self.game = game
        self.side = side
        self.own_truth = own_truth
        self.opponent_truth = opponent_truth
        self.agent = agent
        self.recorder = recorder
        self.rng = rng
        self.tracker = BattleTracker(side)
        self.tracker.record_exact_private_truth(own_truth)
        self.fed: set[int] = set()
        self.previous_history: np.ndarray | None = None
        self.previous_history_ids: np.ndarray | None = None
        self.previous_history_state: np.ndarray | None = None
        self.previous_tokens: np.ndarray | None = None
        self.decision = 0
        self.max_history_len = 0
        self.tera_used = False
        self.run_belief_sampling = run_belief_sampling
        self.training_rows: list[tuple[dict[str, Any], int]] = []

    def _feed_completed(self, battle: Any, *, include_current: bool = False) -> None:
        obs_map = observations_by_turn(battle)
        current = int(getattr(battle, "turn", 0) or 0)
        for turn in sorted(obs_map):
            if turn < current and turn not in self.fed:
                self.tracker.feed_turn(turn, getattr(obs_map[turn], "events", None) or [])
                self.fed.add(turn)
            elif turn == current:
                self.tracker.feed_turn(turn, getattr(obs_map[turn], "events", None) or [])
                if include_current:
                    self.fed.add(turn)

    def choose_teampreview(self, battle: Any) -> str:
        self.tracker.record_private_team(battle)
        obs = Observation.from_battle(battle, tracker=self.tracker)
        self._inspect(battle, obs, phase="preview", action=None)
        members = list(range(1, len(getattr(battle, "team", None) or {}) + 1))
        self.rng.shuffle(members)
        return "/team " + "".join(str(member) for member in members)

    def _choose_action(self, battle: Any, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask).tolist()
        move_slots = [a for a in legal if 0 <= a < 4]
        if move_slots:
            by_slot = list((getattr(getattr(battle, "active_pokemon", None), "moves", None) or {}).values())
            action = max(
                move_slots,
                key=lambda a: float(getattr(by_slot[a], "base_power", 0) or 0)
                if a < len(by_slot) else 0.0,
            )
            # Exercise tera encoding/censorship once per side without turning every move into tera.
            if not self.tera_used and int(getattr(battle, "turn", 0) or 0) >= 3 and mask[4 + action]:
                self.tera_used = True
                return 4 + action
            return action
        non_default = [a for a in legal if a != 13]
        return non_default[self.rng.randrange(len(non_default))] if non_default else 13

    def act(self, battle: Any):
        self.tracker.record_private_team(battle)
        self._feed_completed(battle)
        obs = Observation.from_battle(battle, tracker=self.tracker)
        action = self._choose_action(battle, obs.legality_mask)
        self._inspect(battle, obs, phase="decision", action=action)
        turn = int(getattr(battle, "turn", 0) or 0)
        from fog.encoding.events import record_history_decision

        record_history_decision(self.tracker, turn, battle, TABLES, action)
        return map_action_to_order(battle, action, LocalOrderFactory())

    def _battle_finished(self, battle: Any) -> None:
        self._feed_completed(battle, include_current=True)
        record = {
            "game": self.game, "side": self.side, "phase": "terminal",
            "turn": int(getattr(battle, "turn", 0) or 0),
            "won": getattr(battle, "won", None),
            "fed_turns": sorted(self.fed),
            "tracker_faints": dict(self.tracker.faints),
            "issues": [],
        }
        self._check_replay_training(record, battle)
        self.recorder.check(
            record, bool(getattr(battle, "finished", False)),
            "terminal_not_finished", "poke-env view was not marked finished",
        )
        self.recorder.check(
            record, int(getattr(battle, "turn", 0) or 0) <= 1 or self.max_history_len > 0,
            "history_never_advanced", "a multi-turn game produced no completed-turn history",
        )
        self.recorder.append(record)

    def _check_replay_training(self, record: dict, battle: Any) -> None:
        """Execute the exact replay-censored forward/loss/backward path, without an optimizer."""
        if not self.training_rows:
            self.recorder.issue(record, "replay_training_empty", "no decision rows were captured")
            return
        from fog.constants.schema import NO_ITEM
        from fog.encoding.reveals import FAMILY_COLS
        from fog.training.losses import belief_loss, belief_output_loss
        from fog.training.replay_supervision import oracle_censored_truth
        from fog.training.rollout import RolloutBuffer

        labels = oracle_censored_truth(battle, self.opponent_truth, TABLES)
        self.recorder.check(
            record, not bool((labels[:, SET_NATURE_COL:list(SET_EV_COLS)[-1] + 1] >= 0).any()),
            "replay_native_spread_leak",
            "a public replay fabricated an exact nature or EV label",
        )
        self.recorder.check(
            record, not bool((labels[:, 1] == TABLES.item_index(NO_ITEM)).any()),
            "replay_no_item_target", "<NO_ITEM> entered submitted-item supervision",
        )

        buffer = RolloutBuffer(gamma=1.0, lam=1.0)
        game_id = buffer.start_game()
        for frag, action in self.training_rows:
            buffer.add_decision(
                game_id, frag, action=action, log_prob=0.0, value=0.0
            )
        buffer.end_game(game_id, outcome=1.0, belief_targets=labels)
        device = next(self.agent.parameters()).device
        batch = buffer.to_batch(device)
        valid_counts = {
            family: int(mask.sum().item())
            for family, mask in batch["belief_valid"].items()
        }
        final_label_counts = {
            family: int(
                (labels[..., list(columns)] >= 0).sum()
            )
            for family, columns in FAMILY_COLS.items()
        }
        record["replay_supervision"] = {
            "rows": len(self.training_rows),
            "final_labels": int((labels >= 0).sum()),
            "final_labels_by_family": final_label_counts,
            "oracle_fields_withheld": int(
                ((self.opponent_truth >= 0) & (labels < 0)).sum()
            ),
            "hidden_targets_by_family": valid_counts,
            "no_item_current_rows": int(
                (
                    batch["opp_revealed"]["tokens"][:, :, 1]
                    == TABLES.item_index(NO_ITEM)
                ).sum().item()
            ),
        }
        self.recorder.check(
            record, sum(valid_counts.values()) > 0,
            "replay_training_no_targets",
            "the completed game produced no still-hidden, eventually-revealed targets",
            severity="warning",
        )

        was_training = self.agent.training
        self.agent.train()
        self.agent.zero_grad(set_to_none=True)
        output = self.agent(batch)
        assert output.belief is not None
        marginal = belief_loss(
            output.belief.logits, batch["belief_targets"], batch["belief_valid"]
        )
        chain = (
            belief_output_loss(
                output.belief_chain, batch["belief_targets"], batch["belief_valid"]
            )
            if output.belief_chain is not None
            else marginal.new_zeros(())
        )
        loss = marginal + chain
        self.recorder.check(
            record, torch.isfinite(loss).item(),
            "replay_training_loss_nonfinite", f"dry-run loss was {float(loss.detach())}",
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.agent.parameters()
            if parameter.grad is not None
        ]
        finite_gradients = all(torch.isfinite(gradient).all().item() for gradient in gradients)
        grad_norm = float(
            torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in gradients))
        ) if gradients else 0.0
        self.recorder.check(
            record, bool(gradients) and grad_norm > 0,
            "replay_training_no_gradient", "dry-run loss produced no nonzero gradient",
        )
        self.recorder.check(
            record, finite_gradients,
            "replay_training_gradient_nonfinite", "dry-run backward produced NaN/Inf gradients",
        )
        selected = 0
        last = 0
        selected_by_family: dict[str, int] = {}
        selected_slots: set[int] = set()
        selected_probabilities: list[torch.Tensor] = []
        if output.belief_chain is not None and output.belief_chain.loss_valid is not None:
            for family, mask in output.belief_chain.loss_valid.items():
                selected_by_family[family] = int(mask.sum().item())
                selected += selected_by_family[family]
                target = output.belief_chain.loss_targets[family]
                logits = output.belief_chain.logits[family]
                use = mask & batch["belief_valid"][family]
                if bool(use.any()):
                    probability = torch.softmax(logits, dim=-1).gather(
                        -1,
                        target.clamp(min=0, max=logits.shape[-1] - 1).unsqueeze(-1),
                    ).squeeze(-1)
                    selected_probabilities.append(probability[use])
                    for index in torch.nonzero(use, as_tuple=False):
                        selected_slots.add(int(index[1]))
            if (
                output.belief_chain.context_count is not None
                and output.belief_chain.route_length is not None
            ):
                last = int(
                    (
                        output.belief_chain.context_count
                        == output.belief_chain.route_length - 1
                    ).sum().item()
                )
                counts = output.belief_chain.context_count[
                    output.belief_chain.route_length > 0
                ].to(torch.float32)
                context_summary = {
                    "min": int(counts.min().item()) if len(counts) else 0,
                    "max": int(counts.max().item()) if len(counts) else 0,
                    "mean": float(counts.mean().item()) if len(counts) else 0.0,
                }
            else:
                context_summary = {}
        expected_selected = sum(
            any(bool(mask[row].any()) for mask in batch["belief_valid"].values())
            for row in range(len(self.training_rows))
        )
        self.recorder.check(
            record, selected == expected_selected,
            "replay_query_selection",
            f"selected {selected} arbitrary queries for {expected_selected} trainable rows",
        )
        if selected_probabilities:
            probabilities = torch.cat(selected_probabilities)
            self.recorder.check(
                record, torch.isfinite(probabilities).all().item(),
                "replay_query_probability_nonfinite",
                "a selected true-target probability was NaN/Inf",
            )
            probability_summary = {
                "min": float(probabilities.min().detach()),
                "max": float(probabilities.max().detach()),
                "mean": float(probabilities.mean().detach()),
            }
        else:
            probability_summary = {}
        record["replay_supervision"].update({
            "marginal_loss": float(marginal.detach()),
            "chain_loss": float(chain.detach()),
            "selected_queries": selected,
            "selected_queries_by_family": selected_by_family,
            "selected_slots": sorted(selected_slots),
            "selected_true_probability": probability_summary,
            "assumption_prefix": context_summary,
            "last_field_queries": last,
            "gradient_tensors": len(gradients),
            "gradient_norm": grad_norm,
            "gradients_finite": finite_gradients,
        })
        self.agent.zero_grad(set_to_none=True)
        self.agent.train(was_training)

    def _inspect(
        self, battle: Any, obs: Observation, *, phase: str, action: int | None
    ) -> None:
        turn = int(getattr(battle, "turn", 0) or 0)
        tokens = np.asarray((obs.opp_revealed or {})["tokens"], dtype=np.int64)
        history = np.asarray(obs.history, dtype=np.float32)
        history_ids = np.asarray(obs.history_ids, dtype=np.int64)
        history_state = np.asarray(obs.history_state, dtype=np.float32)
        record: dict[str, Any] = {
            "game": self.game, "side": self.side, "phase": phase, "decision": self.decision,
            "turn": turn, "history_len": int(obs.history_len), "history_rows": len(history),
            "active_idx": np.asarray(obs.active_idx).tolist(),
            "legal_actions": np.flatnonzero(obs.legality_mask).tolist(),
            "chosen_action": action,
            "field": np.asarray(obs.field).round(6).tolist(),
            "entity_ids": np.asarray(obs.entity_ids).tolist(),
            "entity_state": [
                {
                    "hp": float(row[L.hp]), "fainted": bool(row[L.fainted]),
                    "status": int(np.argmax(row[L.status:L.status + DIMS.status])),
                    "boosts": row[L.boosts:L.boosts + DIMS.boosts].round(6).tolist(),
                    "booster_consumed": bool(row[L.booster]),
                    "fallen": float(row[L.fallen]), "tera_used": bool(row[L.tera_used]),
                    "epistemic": row[
                        L.epistemic:L.epistemic + DIMS.epistemic_flags
                    ].astype(int).tolist(),
                }
                for row in np.asarray(obs.entities)
            ],
            "opp_tokens": tokens.tolist(),
            "history_last": history[-1].round(6).tolist() if obs.history_len else None,
            "history_ids_last": history_ids[-1].tolist() if obs.history_len else None,
            "history_state_last": (
                history_state[-1].round(6).tolist() if obs.history_len else None
            ),
            "newly_fed_turns": sorted(self.fed),
            "issues": [],
        }
        self.decision += int(phase == "decision")
        self.max_history_len = max(self.max_history_len, int(obs.history_len))

        try:
            obs.validate()
        except AssertionError as exc:
            self.recorder.issue(record, "observation_validate", repr(exc))
        for name, value in (
            ("entities", obs.entities), ("field", obs.field), ("history", obs.history),
            ("history_state", history_state),
        ):
            self.recorder.check(
                record, np.isfinite(value).all(), f"{name}_nonfinite",
                f"{name} contained NaN or infinity",
            )
        self.recorder.check(
            record, 0 <= obs.history_len <= min(DIMS.history_cap, len(history)),
            "history_length", "history_len was outside the stored window",
        )
        if phase == "decision":
            expected_history = min(max(0, turn - 1), DIMS.history_cap)
            self.recorder.check(
                record, int(obs.history_len) == expected_history,
                "history_turn_alignment",
                f"turn {turn} should expose {expected_history} completed rows, "
                f"not {obs.history_len}",
            )
        if phase == "decision":
            self.recorder.check(
                record, action is not None and bool(obs.legality_mask[action]),
                "illegal_choice", f"chosen action {action} was not legal",
            )

        self._check_history(record, history, history_ids, history_state, int(obs.history_len))
        self._check_entities(
            record, obs, tokens, source_sheet_phase=(phase == "preview"),
            require_active=(phase == "decision"),
        )
        self._check_reveals(record, tokens)
        self._check_model(record, obs, tokens)
        self.previous_history = history[-obs.history_len:].copy() if obs.history_len else np.empty(
            (0, DIMS.turn_dim), dtype=np.float32
        )
        self.previous_history_ids = (
            history_ids[-obs.history_len:].copy() if obs.history_len
            else np.empty((0, DIMS.hist_id_fields), dtype=np.int64)
        )
        self.previous_history_state = (
            history_state[-obs.history_len:].copy() if obs.history_len
            else np.empty((0, DIMS.hist_state), dtype=np.float32)
        )
        self.previous_tokens = tokens.copy()
        if phase == "decision" and action is not None:
            self.training_rows.append((self._training_fragment(obs, turn), int(action)))
        self.recorder.append(record)

    @staticmethod
    def _training_fragment(obs: Observation, turn: int) -> dict[str, Any]:
        revealed = obs.opp_revealed or {
            "tokens": np.full((6, SET_N_ATTRS), -1), "masks": {}
        }
        return {
            "entities": np.asarray(obs.entities).copy(),
            "entity_ids": np.asarray(obs.entity_ids).copy(),
            "field": np.asarray(obs.field).copy(),
            "history": np.asarray(obs.history).copy(),
            "history_ids": np.asarray(obs.history_ids).copy(),
            "history_state": np.asarray(obs.history_state).copy(),
            "history_len": np.int64(obs.history_len),
            "active_idx": np.asarray(obs.active_idx).copy(),
            "legality_mask": np.asarray(obs.legality_mask).copy(),
            "opp_revealed": {
                "tokens": np.asarray(revealed["tokens"]).copy(),
                "masks": {
                    family: np.asarray(mask).copy()
                    for family, mask in (revealed.get("masks") or {}).items()
                },
            },
            "turn": np.int64(turn),
        }

    def _check_history(
        self,
        record: dict,
        history: np.ndarray,
        history_ids: np.ndarray,
        history_state: np.ndarray,
        length: int,
    ) -> None:
        real = history[-length:] if length else history[:0]
        real_ids = history_ids[-length:] if length else history_ids[:0]
        real_state = history_state[-length:] if length else history_state[:0]
        if self.previous_history is not None:
            old = self.previous_history
            self.recorder.check(
                record, length >= len(old) or length == DIMS.history_cap,
                "history_shrank", f"history changed from {len(old)} to {length} rows",
            )
            overlap = min(len(old), len(real))
            if overlap:
                expected = old[-overlap:]
                actual = real[:overlap] if len(real) > len(old) else real[-overlap:]
                self.recorder.check(
                    record, np.array_equal(expected, actual),
                    "history_prefix_changed", "previous completed-turn records changed retroactively",
                )
                for name, old_sidecar, new_sidecar in (
                    ("ids", self.previous_history_ids, real_ids),
                    ("state", self.previous_history_state, real_state),
                ):
                    expected_sidecar = old_sidecar[-overlap:]
                    actual_sidecar = (
                        new_sidecar[:overlap]
                        if len(new_sidecar) > len(old_sidecar)
                        else new_sidecar[-overlap:]
                    )
                    self.recorder.check(
                        record, np.array_equal(expected_sidecar, actual_sidecar),
                        f"history_{name}_prefix_changed",
                        f"previous completed-turn {name} changed retroactively",
                    )
        if not length:
            return
        my_action = real[:, 30:44].sum(axis=1)
        self.recorder.check(
            record, np.allclose(my_action, 1.0),
            "history_my_action", f"completed turns had action sums {my_action.tolist()}",
        )
        opp = real[:, 44:84]
        opp_categories = opp[:, 0] + opp[:, 2] + opp[:, 5]
        self.recorder.check(
            record, np.all(opp_categories >= 1),
            "history_opponent_action", "a completed turn lacked move/switch/unseen opponent evidence",
        )
        resolution = real[:, 84:]
        self.recorder.check(
            record,
            np.allclose(resolution[:, 4:9].sum(axis=1), 1.0)
            and np.allclose(resolution[:, 9:14].sum(axis=1), 1.0),
            "history_effectiveness", "effectiveness buckets were not one-hot",
        )
        self.recorder.check(
            record, real_ids.shape == (length, DIMS.hist_id_fields),
            "history_ids_shape", f"categorical sidecar shape was {real_ids.shape}",
        )
        self.recorder.check(
            record, real_state.shape == (length, DIMS.hist_state)
            and bool((np.abs(real_state) <= 1).all()),
            "history_state_shape", "per-stat history state was mis-shaped or outside [-1,1]",
        )
        self.recorder.check(
            record, bool(((real_ids[:, 14:17] >= 0) & (real_ids[:, 14:17] <= 2)).all()),
            "history_order_range", "actor/switch-phase categorical ids left [0,2]",
        )
        self.recorder.check(
            record,
            bool((real_ids[:, :2] < TABLES.n_species).all()),
            "history_start_species_missing",
            "a completed live turn lacked a recorded start-of-turn active species",
        )

    def _truth_row(self, species: int) -> np.ndarray | None:
        matches = self.own_truth[self.own_truth[:, 0] == species]
        return matches[0] if len(matches) == 1 else None

    def _check_entities(
        self,
        record: dict,
        obs: Observation,
        tokens: np.ndarray,
        *,
        source_sheet_phase: bool,
        require_active: bool,
    ) -> None:
        ids = np.asarray(obs.entity_ids)
        entities = np.asarray(obs.entities)
        hp = entities[:, L.hp]
        self.recorder.check(
            record, bool(((hp >= 0) & (hp <= 1)).all()),
            "hp_bounds", f"HP fractions outside [0,1]: {hp.tolist()}",
        )
        self.recorder.check(
            record, bool((np.abs(entities[:, L.boosts:L.boosts + DIMS.boosts]) <= 1).all()),
            "boost_bounds", "normalized boosts left [-1,1]",
        )
        mine, opponent = map(int, np.asarray(obs.active_idx))
        self.recorder.check(record, 0 <= mine < 6 and 6 <= opponent < 12,
                            "active_index_range", f"active_idx={obs.active_idx}")
        if require_active:
            self.recorder.check(
                record, entities[mine, L.is_active] == 1 and entities[opponent, L.is_active] == 1,
                "active_flag_mismatch", "active indices did not point to active entity flags",
            )
        own_epistemic = entities[:6, L.epistemic:L.epistemic + DIMS.epistemic_flags]
        self.recorder.check(
            record, np.array_equal(own_epistemic, np.ones_like(own_epistemic)),
            "own_private_flags", "the acting player's private fields were not all marked known",
        )
        self.recorder.check(
            record,
            bool((ids[6:, ORIGINAL_ITEM_ID_FIELD] == TABLES.n_items).all())
            and bool((ids[6:, ORIGINAL_ABILITY_ID_FIELD] == TABLES.n_abilities).all()),
            "opponent_original_sheet_leak",
            "immutable private item/ability ids were populated for the opponent",
        )

        unknown = {
            1: TABLES.n_items, 2: TABLES.n_abilities,
            3: TABLES.n_moves, 4: TABLES.n_moves, 5: TABLES.n_moves, 6: TABLES.n_moves,
            SET_NATURE_COL: N_NATURES,
            **{col: N_EV_UNITS for col in SET_EV_COLS},
        }
        id_column = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
            SET_NATURE_COL: NATURE_ID_FIELD,
            **{col: target for col, target in zip(SET_EV_COLS, EV_ID_FIELDS)},
        }
        for slot in range(6):
            for token_col, unk in unknown.items():
                actual = int(ids[6 + slot, id_column[token_col]])
                if tokens[slot, token_col] < 0:
                    self.recorder.check(
                        record, actual == unk, "opponent_private_leak",
                        f"slot {slot} token {token_col} hidden but entity id was {actual}, not UNK {unk}",
                    )
            expected_bits = (tokens[slot, 1:SET_N_ATTRS] >= 0).astype(np.float32)
            actual_bits = entities[
                6 + slot, L.epistemic:L.epistemic + DIMS.epistemic_flags
            ]
            self.recorder.check(
                record, np.array_equal(expected_bits, actual_bits),
                "opponent_epistemic_mismatch",
                f"slot {slot} entity flags disagreed with belief reveal tokens",
            )

        # Initial requests must carry the complete own sheet. Later item/ability state may legally
        # mutate, so stable species/moves/nature/EVs remain strict.
        for slot in range(6):
            truth = self._truth_row(int(ids[slot, 0]))
            if truth is None:
                self.recorder.issue(
                    record, "own_truth_alignment",
                    f"encoded own slot {slot} species was absent/duplicated in source paste",
                )
                continue
            encoded_moves = {int(x) for x in ids[slot, 3:7] if int(x) < TABLES.n_moves}
            truth_moves = {int(x) for x in truth[3:7] if int(x) >= 0}
            self.recorder.check(
                record, encoded_moves == truth_moves, "own_move_sheet",
                f"slot {slot} moves {sorted(encoded_moves)} != truth {sorted(truth_moves)}",
            )
            self.recorder.check(
                record,
                int(ids[slot, NATURE_ID_FIELD]) == int(truth[SET_NATURE_COL]),
                "own_nature_sheet",
                f"slot {slot} nature mismatch",
            )
            self.recorder.check(
                record,
                np.array_equal(
                    ids[slot, list(EV_ID_FIELDS)], truth[list(SET_EV_COLS)]
                ),
                "own_ev_sheet",
                f"slot {slot} EV units did not match truth",
            )
            if source_sheet_phase:
                self.recorder.check(
                    record, int(ids[slot, 1]) == int(truth[1]), "own_item_sheet",
                    f"slot {slot} item {ids[slot, 1]} != truth {truth[1]}",
                )
                self.recorder.check(
                    record, int(ids[slot, 2]) == int(truth[2]), "own_ability_sheet",
                    f"slot {slot} ability {ids[slot, 2]} != truth {truth[2]}",
                )
            self.recorder.check(
                record,
                int(ids[slot, ORIGINAL_ITEM_ID_FIELD]) == int(truth[1]),
                "own_original_item_sheet",
                f"slot {slot} original item {ids[slot, ORIGINAL_ITEM_ID_FIELD]} "
                f"!= truth {truth[1]}",
            )
            self.recorder.check(
                record,
                int(ids[slot, ORIGINAL_ABILITY_ID_FIELD]) == int(truth[2]),
                "own_original_ability_sheet",
                f"slot {slot} original ability {ids[slot, ORIGINAL_ABILITY_ID_FIELD]} "
                f"!= truth {truth[2]}",
            )
            # Booster Energy disappears on entry before the turn-1 request. Preserve that known
            # private fact through the dedicated consumed bit even though current held item is none.
            started_with_booster = int(truth[1]) == TABLES.item_index("boosterenergy")
            currently_empty = int(ids[slot, 1]) == TABLES.item_index("<NO_ITEM>")
            if started_with_booster and currently_empty:
                self.recorder.check(
                    record, entities[slot, L.booster] == 1.0, "booster_history_lost",
                    f"slot {slot} consumed Booster Energy but its persistent bit was unset",
                )

    def _check_reveals(self, record: dict, tokens: np.ndarray) -> None:
        if self.previous_tokens is None:
            return
        previous = self.previous_tokens
        for slot in range(6):
            old_moves = {int(x) for x in previous[slot, 3:7] if x >= 0}
            new_moves = {int(x) for x in tokens[slot, 3:7] if x >= 0}
            self.recorder.check(
                record, old_moves.issubset(new_moves), "move_reveal_regressed",
                f"slot {slot} moves changed from {sorted(old_moves)} to {sorted(new_moves)}",
            )
            for column in (0, 8):
                old, new = int(previous[slot, column]), int(tokens[slot, column])
                self.recorder.check(
                    record, old < 0 or old == new, "stable_reveal_changed",
                    f"slot {slot} column {column} changed {old}->{new}",
                )

    def _check_model(self, record: dict, obs: Observation, tokens: np.ndarray) -> None:
        device = next(self.agent.parameters()).device

        def tensor(value, dtype):
            return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device).unsqueeze(0)

        masks = {
            name: tensor(mask, torch.bool)
            for name, mask in ((obs.opp_revealed or {}).get("masks") or {}).items()
        }
        batch = {
            "entities": tensor(obs.entities, torch.float32),
            "entity_ids": tensor(obs.entity_ids, torch.long),
            "field": tensor(obs.field, torch.float32),
            "history": tensor(obs.history, torch.float32),
            "history_ids": tensor(obs.history_ids, torch.long),
            "history_state": tensor(obs.history_state, torch.float32),
            "history_len": torch.as_tensor([obs.history_len], dtype=torch.long, device=device),
            "active_idx": tensor(obs.active_idx, torch.long),
            "legality_mask": tensor(obs.legality_mask, torch.bool),
            "opp_revealed": {"tokens": tensor(tokens, torch.long), "masks": masks},
        }
        with torch.no_grad():
            output = self.agent(batch)
        legal = batch["legality_mask"][0]
        self.recorder.check(
            record, torch.isfinite(output.log_probs[0, legal]).all().item(),
            "policy_nonfinite", "legal policy log-probabilities were non-finite",
        )
        self.recorder.check(
            record, torch.isneginf(output.log_probs[0, ~legal]).all().item(),
            "policy_illegal_mass", "an illegal action did not have -inf log-probability",
        )
        self.recorder.check(
            record, torch.isfinite(output.value).all().item(),
            "value_nonfinite", "value output was non-finite",
        )
        assert output.belief is not None
        for family, logits in output.belief.logits.items():
            flat = logits.reshape(-1, logits.shape[-1])
            self.recorder.check(
                record, not torch.isnan(flat).any().item() and torch.isfinite(flat).any(dim=1).all().item(),
                "belief_nonfinite", f"{family} had NaN or an all-masked row",
            )

        family = FAMILIES[self.decision % len(FAMILIES)]
        slot = (self.game + self.decision) % 6
        contexts = self.agent.encode_contexts(batch)
        belief_context = self.agent.belief_context(contexts)
        with torch.no_grad():
            query = self.agent.belief.query_field(
                belief_context, batch["opp_revealed"], slot=slot, field=family
            )
        finite = torch.isfinite(query[0])
        self.recorder.check(
            record, finite.any().item() and not torch.isnan(query).any().item(),
            "belief_query_nonfinite", f"query {family}[{slot}] was invalid",
        )
        probs = torch.softmax(query[0], dim=-1)
        values, indices = torch.topk(probs, min(5, probs.numel()))
        record["belief_query"] = {
            "slot": slot, "field": family,
            "top": [[int(i), float(v)] for i, v in zip(indices, values, strict=True)],
        }

        if self.run_belief_sampling:
            with torch.no_grad():
                sample = self.agent.belief.sample_team(
                    belief_context, batch["opp_revealed"], k=1,
                    generator=torch.Generator(device=device).manual_seed(777),
                    return_trace=True,
                )
            generated = sample.teams[0, 0]
            revealed = batch["opp_revealed"]["tokens"][0] >= 0
            self.recorder.check(
                record, torch.equal(generated[revealed], batch["opp_revealed"]["tokens"][0][revealed]),
                "belief_sample_unclamped", "ancestral sample changed a public reveal",
            )
            self.recorder.check(
                record, torch.isfinite(sample.log_joint_path_prob).all().item(),
                "belief_sample_nonfinite", "ancestral sample probability was non-finite",
            )
            self.run_belief_sampling = False


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(UTC)
    out_dir = Path(args.out)
    recorder = Recorder(out_dir)
    pool = load_pool(args.pool)
    if len(pool) < 2:
        raise ValueError("preflight audit needs at least two source teams")
    rng = random.Random(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    agent = FogAgent.from_tables({}).to(device).eval()
    before = _parameter_digest(agent)
    results = []

    with ShowdownDirectClient() as client:
        for game in range(args.games):
            p1_paste = pool[(2 * game) % len(pool)]
            p2_paste = pool[(2 * game + 1) % len(pool)]
            p1_truth = truth_tokens_from_paste(p1_paste, TABLES)
            p2_truth = truth_tokens_from_paste(p2_paste, TABLES)
            p1 = AuditController(
                game=game, side="p1", own_truth=p1_truth, opponent_truth=p2_truth,
                agent=agent, recorder=recorder, rng=random.Random(rng.randrange(2**63)),
                run_belief_sampling=(game == 0),
            )
            p2 = AuditController(
                game=game, side="p2", own_truth=p2_truth, opponent_truth=p1_truth,
                agent=agent, recorder=recorder, rng=random.Random(rng.randrange(2**63)),
            )
            result = play_local_battle(
                client,
                {
                    "formatid": args.format,
                    "seed": f"sodium,{rng.getrandbits(256):064x}",
                    "strictChoices": True,
                    "p1": {"name": f"AuditA{game}", "team": to_packed(p1_paste)},
                    "p2": {"name": f"AuditB{game}", "team": to_packed(p2_paste)},
                },
                p1, p2, seed=rng.randrange(2**31),
                battle_tag=f"battle-preflight-{game}", max_turns=args.max_turns,
            )
            results.append({
                "game": game, "winner": result.winner, "turns": result.turns,
                "truncated": result.truncated,
            })
            if result.truncated:
                recorder.issues.append({
                    "game": game, "side": "both", "turn": result.turns, "phase": "terminal",
                    "code": "game_truncated", "severity": "error",
                    "detail": f"game reached max_turns={args.max_turns}",
                })

    after = _parameter_digest(agent)
    if before != after:
        recorder.issues.append({
            "game": None, "side": None, "turn": None, "phase": "run",
            "code": "parameters_changed", "severity": "error",
            "detail": "model parameter digest changed during a no-optimizer audit",
        })
    if any(parameter.grad is not None for parameter in agent.parameters()):
        recorder.issues.append({
            "game": None, "side": None, "turn": None, "phase": "run",
            "code": "gradients_created", "severity": "error",
            "detail": "a parameter gradient was populated during the audit",
        })

    recorder.write_trace()
    errors = [issue for issue in recorder.issues if issue["severity"] == "error"]
    warnings = [issue for issue in recorder.issues if issue["severity"] == "warning"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": (datetime.now(UTC) - started).total_seconds(),
        "git_revision": _git_revision(),
        "source_fingerprint": source_fingerprint(),
        "backend": "showdown_direct",
        "format": args.format,
        "device": str(device),
        "games_requested": args.games,
        "games_completed": sum(not result["truncated"] for result in results),
        "optimizer_steps": 0,
        "decisions_recorded": sum(record["phase"] == "decision" for record in recorder.records),
        "replay_dryruns": sum("replay_supervision" in record for record in recorder.records),
        "replay_training_rows": sum(
            int(record.get("replay_supervision", {}).get("rows", 0))
            for record in recorder.records
        ),
        "replay_hidden_targets": sum(
            sum(record.get("replay_supervision", {}).get("hidden_targets_by_family", {}).values())
            for record in recorder.records
        ),
        "parameter_digest_before": before,
        "parameter_digest_after": after,
        "passed": not errors and sum(not result["truncated"] for result in results) == args.games,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issue_counts": dict(Counter(issue["code"] for issue in recorder.issues)),
        "games": results,
        "trace": str((out_dir / "decisions.jsonl").resolve()),
        "issues": recorder.issues,
    }
    text = json.dumps(summary, indent=2)
    (out_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
    latest = out_dir.parent / "latest.json"
    latest.write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument(
        "--out", type=Path,
        default=Path("runs/preflight_audit") / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    summary = run(args)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
