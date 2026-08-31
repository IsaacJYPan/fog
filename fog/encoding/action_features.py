"""Per-choice facts for rule-guided policy learning.

The main observation encodes a battle as team and field summaries.  That is a good general state
representation, but it makes a deterministic rule such as ``SimpleHeuristicsPlayer`` needlessly
hard to copy: the policy has to rediscover move power and then associate a pooled party member with
the switch button whose position changes whenever the active Pokemon changes.

This module keeps the existing observation intact and adds a small [14, F] table.  Each row describes
the choice at the same row of the fixed action space.  The final feature is the exact choice made by
the rule player.  It is deliberately explicit: a new run can start with that rule as a strong prior
and then learn a residual, while old configurations and checkpoints remain byte-compatible.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fog.constants.schema import N_ACTIONS
from fog.env.actions import (
    action_legality_mask,
    available_move_by_slot,
    available_switch_by_slot,
    map_order_to_action,
)

ACTION_FEATURE_NAMES = (
    "legal",
    "ordinary_move",
    "tera_move",
    "switch",
    "forced",
    "slot_fraction",
    "move_power",
    "move_accuracy",
    "expected_hits",
    "physical",
    "special",
    "status_move",
    "same_type_bonus",
    "type_effectiveness",
    "physical_stat_ratio",
    "special_stat_ratio",
    "estimated_damage",
    "hazard_setup_ready",
    "hazard_removal_ready",
    "setup_move_ready",
    "tera_improves_matchup",
    "switch_matchup",
    "positive_switch_matchup",
    "active_matchup",
    "should_switch_out",
    "own_team_remaining",
    "opponent_team_remaining",
    "active_full_health",
    "opponent_full_health",
    "own_side_has_hazard",
    "can_tera",
    "move_branch_active",
    "switch_branch_active",
    "simple_rule_pick",
)
ACTION_FEATURE_DIM = len(ACTION_FEATURE_NAMES)
_I = {name: index for index, name in enumerate(ACTION_FEATURE_NAMES)}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _remaining(team: Any, *, opponent: bool = False) -> int:
    mons = list((team or {}).values()) if isinstance(team, dict) else list(team or [])
    fainted = sum(bool(getattr(mon, "fainted", False)) for mon in mons)
    # The rule player assumes six opposing Pokemon even before all have been revealed.
    return 6 - fainted if opponent else sum(not bool(getattr(mon, "fainted", False)) for mon in mons)


def encode_action_features(battle: Any) -> np.ndarray:
    """Return finite per-action features aligned with the project's fourteen action slots."""
    features = np.zeros((N_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32)
    if battle is None:
        features[13, _I["legal"]] = 1.0
        features[13, _I["forced"]] = 1.0
        features[13, _I["simple_rule_pick"]] = 1.0
        return features

    legal = np.asarray(action_legality_mask(battle), dtype=bool)
    features[:, _I["legal"]] = legal.astype(np.float32)
    features[:4, _I["ordinary_move"]] = 1.0
    features[4:8, _I["tera_move"]] = 1.0
    features[8:13, _I["switch"]] = 1.0
    features[13, _I["forced"]] = 1.0
    features[:4, _I["slot_fraction"]] = np.arange(4, dtype=np.float32) / 3.0
    features[4:8, _I["slot_fraction"]] = np.arange(4, dtype=np.float32) / 3.0
    features[8:13, _I["slot_fraction"]] = np.arange(5, dtype=np.float32) / 4.0

    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    moves = available_move_by_slot(battle)
    switches = available_switch_by_slot(battle)
    can_tera = bool(getattr(battle, "can_tera", False))
    own_remaining = _remaining(getattr(battle, "team", None))
    opp_remaining = _remaining(getattr(battle, "opponent_team", None), opponent=True)
    own_hazards = bool(getattr(battle, "side_conditions", None))

    matchup = should_switch = 0.0
    physical_ratio = special_ratio = 1.0
    move_branch = bool(moves)
    if active is not None and opponent is not None:
        from poke_env.player import SimpleHeuristicsPlayer

        try:
            matchup = _safe_float(SimpleHeuristicsPlayer._estimate_matchup(active, opponent))
            should_switch = float(bool(SimpleHeuristicsPlayer._should_switch_out(battle)))
            physical_ratio = _safe_float(
                SimpleHeuristicsPlayer._stat_estimation(active, "atk")
                / max(SimpleHeuristicsPlayer._stat_estimation(opponent, "def"), 1e-6),
                1.0,
            )
            special_ratio = _safe_float(
                SimpleHeuristicsPlayer._stat_estimation(active, "spa")
                / max(SimpleHeuristicsPlayer._stat_estimation(opponent, "spd"), 1e-6),
                1.0,
            )
        except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
            # Lightweight replay/test doubles need only shape-safe features.  Real authoritative
            # battles expose the complete poke-env mechanics queried above.
            pass
        move_branch = bool(moves) and (not bool(should_switch) or not bool(switches))
    switch_branch = bool(switches) and not move_branch

    common = {
        "active_matchup": np.clip(matchup / 4.0, -1.0, 1.0),
        "should_switch_out": should_switch,
        "own_team_remaining": own_remaining / 6.0,
        "opponent_team_remaining": opp_remaining / 6.0,
        "active_full_health": float(
            active is not None and _safe_float(getattr(active, "current_hp_fraction", 0.0)) == 1.0
        ),
        "opponent_full_health": float(
            opponent is not None
            and _safe_float(getattr(opponent, "current_hp_fraction", 0.0)) == 1.0
        ),
        "own_side_has_hazard": float(own_hazards),
        "can_tera": float(can_tera),
        "move_branch_active": float(move_branch),
        "switch_branch_active": float(switch_branch),
    }
    for name, value in common.items():
        features[:, _I[name]] = value

    if active is not None and opponent is not None:
        from poke_env.player import SimpleHeuristicsPlayer
        from poke_env.player.baselines import MoveCategory

        active_types = tuple(getattr(active, "types", None) or ())
        for slot, move in moves.items():
            base_power = _safe_float(getattr(move, "base_power", 0.0))
            accuracy = getattr(move, "accuracy", 1.0)
            accuracy = 1.0 if accuracy is True else _safe_float(accuracy, 1.0)
            expected_hits = _safe_float(getattr(move, "expected_hits", 1.0), 1.0)
            move_type = getattr(move, "type", None)
            category = getattr(move, "category", None)
            physical = category == MoveCategory.PHYSICAL
            special = category == MoveCategory.SPECIAL
            ratio = physical_ratio if physical else special_ratio
            try:
                effectiveness = _safe_float(opponent.damage_multiplier(move), 1.0)
            except (AttributeError, TypeError, ValueError):
                effectiveness = 1.0
            same_type = move_type in active_types
            damage = base_power * (1.5 if same_type else 1.0) * ratio * accuracy * expected_hits * effectiveness
            move_id = str(getattr(move, "id", "")).lower()
            target = str(getattr(move, "target", "")).lower()
            boosts = getattr(move, "boosts", None) or {}
            setup_ready = (
                common["active_full_health"] > 0
                and matchup > 0
                and bool(boosts)
                and sum(_safe_float(value) for value in boosts.values()) >= 2
                and "self" in target
                and min(
                    (_safe_float((getattr(active, "boosts", None) or {}).get(stat, 0))
                     for stat, value in boosts.items() if _safe_float(value) > 0),
                    default=6,
                ) < 6
            )
            hazard = (
                opp_remaining >= 3
                and move_id in SimpleHeuristicsPlayer.ENTRY_HAZARDS
                and SimpleHeuristicsPlayer.ENTRY_HAZARDS[move_id]
                not in (getattr(battle, "opponent_side_conditions", None) or {})
            )
            removal = (
                own_hazards
                and move_id in SimpleHeuristicsPlayer.ANTI_HAZARDS_MOVES
                and own_remaining >= 2
            )
            try:
                tera_help = bool(SimpleHeuristicsPlayer._should_terastallize(battle, move))
            except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
                tera_help = False
            for action in (slot, 4 + slot):
                features[action, _I["move_power"]] = np.clip(base_power / 250.0, 0.0, 2.0)
                features[action, _I["move_accuracy"]] = np.clip(accuracy, 0.0, 1.0)
                features[action, _I["expected_hits"]] = np.clip(expected_hits / 5.0, 0.0, 2.0)
                features[action, _I["physical"]] = float(physical)
                features[action, _I["special"]] = float(special)
                features[action, _I["status_move"]] = float(not physical and not special)
                features[action, _I["same_type_bonus"]] = float(same_type)
                features[action, _I["type_effectiveness"]] = np.clip(effectiveness / 4.0, 0.0, 2.0)
                features[action, _I["physical_stat_ratio"]] = np.clip(physical_ratio / 4.0, 0.0, 2.0)
                features[action, _I["special_stat_ratio"]] = np.clip(special_ratio / 4.0, 0.0, 2.0)
                features[action, _I["estimated_damage"]] = np.clip(np.log1p(max(0.0, damage)) / 8.0, 0.0, 2.0)
                features[action, _I["hazard_setup_ready"]] = float(hazard)
                features[action, _I["hazard_removal_ready"]] = float(removal)
                features[action, _I["setup_move_ready"]] = float(setup_ready)
                features[action, _I["tera_improves_matchup"]] = float(tera_help)

        for slot, candidate in switches.items():
            try:
                score = _safe_float(SimpleHeuristicsPlayer._estimate_matchup(candidate, opponent))
            except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
                score = 0.0
            action = 8 + slot
            features[action, _I["switch_matchup"]] = np.clip(score / 4.0, -1.0, 1.0)
            features[action, _I["positive_switch_matchup"]] = float(score > 0)

    # Exact, transparent starting rule.  This is not a training label smuggled in from future data:
    # it is recomputed from the current legal battle state at inference time.  A rule-guided model
    # may use it as a prior and learn only deviations that have positive gameplay evidence.
    try:
        from poke_env.player import SimpleHeuristicsPlayer

        order = SimpleHeuristicsPlayer.choose_singles_move(battle)[0]
        preferred = map_order_to_action(battle, order)
    except (AssertionError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        legal_indices = np.flatnonzero(legal)
        preferred = int(legal_indices[0]) if len(legal_indices) else 13
    features[preferred, _I["simple_rule_pick"]] = 1.0
    return features


__all__ = ["ACTION_FEATURE_DIM", "ACTION_FEATURE_NAMES", "encode_action_features"]
