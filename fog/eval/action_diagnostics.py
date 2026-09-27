"""Action-family and raw Tera-margin diagnostics for the actual acting policy."""
from __future__ import annotations

from typing import Any

FAMILIES = {"ordinary_move": range(4), "tera_move": range(4, 8),
            "switch": range(8, 13), "forced": range(13, 14)}


def new_action_stats() -> dict:
    return {"decisions": 0, "selected_by_slot": [0] * 14,
            "legal_opportunities_by_slot": [0] * 14,
            "selected_by_family": dict.fromkeys(FAMILIES, 0),
            "legal_decisions_by_family": dict.fromkeys(FAMILIES, 0),
            "games": {}, "games_with_tera": 0, "first_tera_turns": []}


def record_action(stats: dict, battle: Any, legal: Any, log_probs: Any, action: int,
                  *, policy_diagnostics: bool = False) -> None:
    import torch

    stats["decisions"] += 1
    stats["selected_by_slot"][action] += 1
    for index, value in enumerate(legal):
        stats["legal_opportunities_by_slot"][index] += int(bool(value))
    for family, indices in FAMILIES.items():
        stats["legal_decisions_by_family"][family] += int(any(legal[i] for i in indices))
        stats["selected_by_family"][family] += int(action in indices)
    game = stats["games"].setdefault(str(battle.battle_tag),
        {"decisions": 0, "tera_legal_decisions": 0, "first_tera_turn": None})
    game["decisions"] += 1
    game["tera_legal_decisions"] += int(any(legal[4:8]))
    if action in FAMILIES["tera_move"] and game["first_tera_turn"] is None:
        game["first_tera_turn"] = int(battle.turn)
        stats["first_tera_turns"].append(int(battle.turn))
        stats["games_with_tera"] += 1
    if not policy_diagnostics or not any(legal[4:8]):
        return
    raw = log_probs.detach().cpu()
    diag = stats.setdefault("raw_tera_diagnostics", {
        "paired_margins": [[] for _ in range(4)], "max_tera_minus_max_nontera": [],
        "tera_probability_t1": []})
    # Log-probability differences equal raw logit differences (normalizer cancels).
    for i in range(4):
        if legal[i] and legal[i + 4]:
            diag["paired_margins"][i].append(float(raw[i + 4] - raw[i]))
    nontera = torch.cat((raw[:4], raw[8:]))
    if torch.isfinite(nontera).any():
        diag["max_tera_minus_max_nontera"].append(float(raw[4:8].max() - nontera.max()))
    diag["tera_probability_t1"].append(float(raw.softmax(-1)[4:8].sum()))
