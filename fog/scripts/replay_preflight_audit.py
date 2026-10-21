"""Inspect replay parsing, model inputs, and belief-loss backward passes.

Unlike ``preflight_audit`` (fresh direct-Showdown mechanics), this drives the actual public replay
parser, shard tensorization, replay dataset masks, model forward, censored marginal/chain losses,
and backward pass. It never creates an optimizer or changes a parameter.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fog.constants.schema import (
    N_EV_UNITS,
    N_NATURES,
    NO_ITEM,
    SET_EV_COLS,
    SET_NATURE_COL,
    SET_TERA_COL,
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
from fog.encoding.events import to_id
from fog.scripts.audit_support import parameter_digest_with_metadata as _parameter_digest

SCHEMA_VERSION = 4
FAMILIES = ("species", "item", "ability", "move", "nature", "ev", "tera")
SCALAR_COLUMNS = {
    "species": 0, "item": 1, "ability": 2,
    "nature": SET_NATURE_COL, "tera": SET_TERA_COL,
}
MOVE_COLUMNS = (3, 4, 5, 6)


def _family_counts(values: np.ndarray) -> dict[str, int]:
    return {
        family: int((values[..., list(columns)] >= 0).sum())
        for family, columns in {
            "species": (0,),
            "item": (1,),
            "ability": (2,),
            "move": MOVE_COLUMNS,
            "nature": (SET_NATURE_COL,),
            "ev": SET_EV_COLS,
            "tera": (SET_TERA_COL,),
        }.items()
    }


def _protocol_preview_species(replay: dict) -> dict[str, list[int]]:
    """Team-preview species in protocol slot order for both sides."""
    previews: dict[str, list[int]] = {"p1": [], "p2": []}
    for line in str(replay.get("log", "")).splitlines():
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "poke" and parts[2] in previews:
            previews[parts[2]].append(
                TABLES.species_index(to_id(parts[3].split(",")[0]))
            )
    return previews


def _perspective_issues(rows: list[dict], replay: dict, replay_id: str) -> list[str]:
    """Verify each role's opponent, self, and entity slots point at the correct protocol side."""
    issues: list[str] = []
    previews = _protocol_preview_species(replay)
    role_rows = {
        role: next((row for row in rows if row["role"] == role), None)
        for role in ("p1", "p2")
    }
    for role, other in (("p1", "p2"), ("p2", "p1")):
        row = role_rows[role]
        if row is None:
            continue
        expected_opp = previews[other]
        expected_self = previews[role]
        actual_target = np.asarray(row["belief"])[: len(expected_opp), 0].tolist()
        actual_input = np.asarray(row["obs"].opp_revealed["tokens"])[
            : len(expected_opp), 0
        ].tolist()
        actual_self = np.asarray(row["self_belief"])[: len(expected_self), 0].tolist()
        actual_entities = np.asarray(row["obs"].entity_ids)[: len(expected_self), 0].tolist()
        for name, actual, expected in (
            ("opponent target", actual_target, expected_opp),
            ("opponent input", actual_input, expected_opp),
            ("self target", actual_self, expected_self),
            ("own entity preview", actual_entities, expected_self),
        ):
            if actual != expected:
                issues.append(
                    f"{replay_id}/{role}: {name} species {actual} != protocol {expected}"
                )

    p1, p2 = role_rows["p1"], role_rows["p2"]
    if (
        p1 is not None
        and p2 is not None
        and previews["p1"] != previews["p2"]
        and np.array_equal(p1["belief"], p2["belief"])
    ):
        issues.append(
            f"{replay_id}: distinct protocol teams produced identical perspective targets"
        )
    return issues


def _private_leak_issues(row: dict, prefix: str) -> list[str]:
    obs = row["obs"]
    entities = obs.entities[:6]
    ids = obs.entity_ids[:6]
    epistemic = entities[:, L.epistemic : L.epistemic + DIMS.epistemic_flags]
    issues: list[str] = []
    checks = (
        (0, 1, TABLES.n_items, "item"),
        (1, 2, TABLES.n_abilities, "ability"),
        (2, 3, TABLES.n_moves, "move0"),
        (3, 4, TABLES.n_moves, "move1"),
        (4, 5, TABLES.n_moves, "move2"),
        (5, 6, TABLES.n_moves, "move3"),
        (6, NATURE_ID_FIELD, N_NATURES, "nature"),
        *(
            (7 + stat, column, N_EV_UNITS, f"ev{stat}")
            for stat, column in enumerate(EV_ID_FIELDS)
        ),
    )
    for bit, column, unknown, name in checks:
        leaked = (epistemic[:, bit] == 0) & (ids[:, column] != unknown)
        if bool(leaked.any()):
            issues.append(f"{prefix}: {name} private id survived an unknown epistemic flag")
    if bool((ids[:, ORIGINAL_ITEM_ID_FIELD] != TABLES.n_items).any()):
        issues.append(f"{prefix}: unavailable original private items entered a public replay row")
    if bool((ids[:, ORIGINAL_ABILITY_ID_FIELD] != TABLES.n_abilities).any()):
        issues.append(f"{prefix}: unavailable original private abilities entered a public replay row")
    hidden_tera = epistemic[:, 13] == 0
    if bool(
        (
            entities[
                hidden_tera,
                L.tera_type : L.tera_type + DIMS.tera_type,
            ]
            != 0
        ).any()
    ):
        issues.append(f"{prefix}: a hidden private tera type survived replay redaction")
    if bool(
        (
            entities[:, L.move_pp : L.move_pp + DIMS.move_pp]
            != 0
        ).any()
    ):
        issues.append(f"{prefix}: private PP bookkeeping survived replay redaction")
    return issues


def _public_target_issues(row: dict, prefix: str) -> list[str]:
    tokens = np.asarray(row["obs"].opp_revealed["tokens"])
    target = np.asarray(row["belief"])
    issues: list[str] = []
    no_item = TABLES.item_index(NO_ITEM)
    if bool((tokens[:, 1] == no_item).any()):
        issues.append(f"{prefix}: current <NO_ITEM> entered submitted-item belief tokens")
    if bool((target[:, 1] == no_item).any()):
        issues.append(f"{prefix}: current <NO_ITEM> became an original-item label")
    for slot in range(6):
        for family, column in SCALAR_COLUMNS.items():
            public = int(tokens[slot, column])
            if public >= 0 and public != int(target[slot, column]):
                issues.append(
                    f"{prefix}: public {family}[{slot}]={public} conflicts with terminal "
                    f"submitted label {int(target[slot, column])}"
                )
        public_moves = {int(value) for value in tokens[slot, MOVE_COLUMNS] if value >= 0}
        target_moves = {int(value) for value in target[slot, MOVE_COLUMNS] if value >= 0}
        if not public_moves.issubset(target_moves):
            issues.append(
                f"{prefix}: public moves for slot {slot} are not a subset of terminal labels"
            )
    return issues


def _monotonic_issues(rows: list[dict], replay_id: str, role: str) -> list[str]:
    issues: list[str] = []
    previous: np.ndarray | None = None
    for row in sorted(rows, key=lambda item: int(item["turn"])):
        current = np.asarray(row["obs"].opp_revealed["tokens"])
        if previous is not None:
            for slot in range(6):
                for family, column in SCALAR_COLUMNS.items():
                    old, new = int(previous[slot, column]), int(current[slot, column])
                    if old >= 0 and new != old:
                        issues.append(
                            f"{replay_id}/{role}/turn{row['turn']}: submitted {family}[{slot}] "
                            "was forgotten or changed"
                        )
                old_moves = {
                    int(value) for value in previous[slot, MOVE_COLUMNS] if value >= 0
                }
                new_moves = {
                    int(value) for value in current[slot, MOVE_COLUMNS] if value >= 0
                }
                if not old_moves.issubset(new_moves):
                    issues.append(
                        f"{replay_id}/{role}/turn{row['turn']}: submitted move evidence regressed "
                        f"for slot {slot}"
                    )
        previous = current
    return issues


def _selected_queries(output, batch: dict) -> tuple[dict[str, int], list[int]]:
    counts: Counter[str] = Counter()
    slots: set[int] = set()
    if output.loss_valid is None:
        return dict(counts), []
    for family, mask in output.loss_valid.items():
        use = mask & batch["belief_valid"][family]
        counts[family] += int(use.sum())
        if family == "move":
            used_slots = torch.nonzero(use.any(dim=-1).any(dim=0), as_tuple=True)[0]
        else:
            used_slots = torch.nonzero(use.any(dim=0), as_tuple=True)[0]
        slots.update(int(slot) for slot in used_slots)
    return dict(counts), sorted(slots)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from fog.config import load_config
    from fog.models.agent import FogAgent
    from fog.scripts.parse_replays import parse_replay, replay_files, rows_to_arrays
    from fog.training.dataset import _build_batch
    from fog.training.losses import belief_loss, belief_output_loss
    from fog.training.ppo import _chain_probability_metrics
    from fog.training.preflight import source_fingerprint

    cfg = load_config(args.config)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    files = replay_files(args.raw, args.games)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "rows.jsonl"

    agent = FogAgent.from_tables(cfg).to(device)
    compatibility = {"loaded": [], "initialized": []}
    init_path = cfg.get("bc", {}).get("init_ckpt")
    if init_path:
        state = torch.load(init_path, map_location=device, weights_only=False)
        if cfg.get("bc", {}).get("init_compatible", False):
            compatibility = agent.load_compatible_state_dict(state)
        else:
            agent.load_state_dict(state)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.belief.parameters():
        parameter.requires_grad_(True)
    agent.train()

    digest_before = _parameter_digest(agent)
    issues: list[str] = []
    row_records: list[dict[str, Any]] = []
    dryruns: list[dict[str, Any]] = []
    game_records: list[dict[str, Any]] = []
    labels_total: Counter[str] = Counter()
    valid_total: Counter[str] = Counter()
    selected_total: Counter[str] = Counter()
    selected_slots: set[int] = set()
    context_depths: list[int] = []
    last_queries = 0
    ratings: list[int] = []

    for game_index, path in enumerate(files):
        replay_id = path.stem
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            rating = int(replay.get("rating") or 0)
            ratings.append(rating)
            parsed = parse_replay(replay)
            rows = parsed["rows"]
            if not rows:
                issues.append(f"{replay_id}: parser produced no decision rows")
                continue
            arrays = rows_to_arrays(rows)
            all_indices = np.arange(len(rows))
            cpu_batch = _build_batch(arrays, all_indices, TABLES, None)
            for family, mask in cpu_batch["belief_valid"].items():
                valid_total[family] += int(mask.sum())

            roles = sorted({str(row["role"]) for row in rows})
            if roles != ["p1", "p2"]:
                issues.append(f"{replay_id}: expected both perspectives, got {roles}")
            issues.extend(_perspective_issues(rows, replay, replay_id))
            for role in roles:
                role_rows = [row for row in rows if row["role"] == role]
                issues.extend(_monotonic_issues(role_rows, replay_id, role))

            forced_support = 0
            for row_index, row in enumerate(rows):
                prefix = f"{replay_id}/{row['role']}/turn{row['turn']}"
                row_issues: list[str] = []
                try:
                    row["obs"].validate()
                except Exception as error:  # noqa: BLE001 - audit records exact row
                    row_issues.append(f"{prefix}: observation validation failed: {error}")
                expected_history = min(
                    DIMS.history_cap, max(0, int(row["turn"]) - 1)
                )
                if int(row["obs"].history_len) != expected_history:
                    row_issues.append(
                        f"{prefix}: history_len={row['obs'].history_len}, "
                        f"expected {expected_history}"
                    )
                expected_rows = max(1, int(row["obs"].history_len))
                if len(row["obs"].history) != expected_rows:
                    row_issues.append(f"{prefix}: stored history rows disagree with history_len")
                for name in ("entities", "field", "history", "history_state"):
                    value = getattr(row["obs"], name)
                    if value is not None and not bool(np.isfinite(value).all()):
                        row_issues.append(f"{prefix}: {name} contains a non-finite value")
                action = int(row["action"])
                raw_mask = np.asarray(row["mask"], dtype=bool)
                raw_action_supported = 0 <= action < 14 and bool(raw_mask[action])
                if not 0 <= action < 14:
                    row_issues.append(f"{prefix}: action {action} is out of range")
                if not raw_action_supported:
                    forced_support += 1
                row_issues.extend(_private_leak_issues(row, prefix))
                row_issues.extend(_public_target_issues(row, prefix))
                issues.extend(row_issues)

                known = _family_counts(np.asarray(row["obs"].opp_revealed["tokens"]))
                valid = {
                    family: int(cpu_batch["belief_valid"][family][row_index].sum())
                    for family in FAMILIES
                }
                row_records.append(
                    {
                        "game": game_index,
                        "replay_id": replay_id,
                        "rating": rating,
                        "role": row["role"],
                        "turn": int(row["turn"]),
                        "history_len": int(row["obs"].history_len),
                        "action": action,
                        "raw_action_supported": raw_action_supported,
                        "reconstructed_legal_actions": (
                            int(raw_mask.sum()) + int(not raw_action_supported)
                        ),
                        "known_submitted_facts": known,
                        "hidden_training_targets": valid,
                        "issues": row_issues,
                    }
                )

            terminal_by_role: dict[str, dict[str, int]] = {}
            for role in roles:
                role_row = next(row for row in rows if row["role"] == role)
                family_counts = _family_counts(np.asarray(role_row["belief"]))
                terminal_by_role[role] = family_counts
                labels_total.update(family_counts)

                role_indices = np.asarray(
                    [index for index, row in enumerate(rows) if row["role"] == role],
                    dtype=np.int64,
                )
                batch = _build_batch(arrays, role_indices, TABLES, device)
                agent.zero_grad(set_to_none=True)
                output = agent(batch)
                marginal_stats: dict[str, int] = {}
                chain_stats: dict[str, int] = {}
                marginal = belief_loss(
                    output.belief.logits,
                    batch["belief_targets"],
                    batch["belief_valid"],
                    stats=marginal_stats,
                )
                chain = belief_output_loss(
                    output.belief_chain,
                    batch["belief_targets"],
                    batch["belief_valid"],
                    stats=chain_stats,
                )
                total = (
                    float(cfg.get("loss", {}).get("belief_coef", 1.0)) * marginal
                    + float(cfg.get("loss", {}).get("belief_chain_coef", 1.0)) * chain
                )
                total.backward()
                gradients = [
                    parameter.grad.detach()
                    for parameter in agent.belief.parameters()
                    if parameter.grad is not None
                ]
                finite_gradients = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                gradient_norm = float(
                    torch.sqrt(sum((gradient.float() ** 2).sum() for gradient in gradients))
                ) if gradients else 0.0
                query_counts, query_slots = _selected_queries(output.belief_chain, batch)
                selected_total.update(query_counts)
                selected_slots.update(query_slots)
                context = output.belief_chain.context_count
                route_length = output.belief_chain.route_length
                if context is not None and route_length is not None:
                    active = route_length > 0
                    context_depths.extend(int(value) for value in context[active].tolist())
                    last_queries += int((context[active] == route_length[active] - 1).sum())
                probability = _chain_probability_metrics(
                    output.belief_chain, batch["belief_valid"]
                )
                dryrun_issues = []
                if not bool(torch.isfinite(total)):
                    dryrun_issues.append("loss was non-finite")
                if not gradients or gradient_norm <= 0:
                    dryrun_issues.append("belief model received no gradient")
                if not finite_gradients:
                    dryrun_issues.append("belief gradient was non-finite")
                if int(chain_stats.get("valid", 0)) <= 0:
                    dryrun_issues.append("arbitrary-query route selected no trainable target")
                if int(chain_stats.get("dropped_nonfinite", 0)) > 0:
                    dryrun_issues.append("a chain target had non-finite/masked true logits")
                for message in dryrun_issues:
                    issues.append(f"{replay_id}/{role}: {message}")
                dryruns.append(
                    {
                        "replay_id": replay_id,
                        "role": role,
                        "rows": len(role_indices),
                        "marginal_loss": float(marginal.detach()),
                        "chain_loss": float(chain.detach()),
                        "total_loss": float(total.detach()),
                        "marginal_valid": int(marginal_stats.get("valid", 0)),
                        "marginal_dropped": int(marginal_stats.get("dropped_nonfinite", 0)),
                        "chain_valid": int(chain_stats.get("valid", 0)),
                        "chain_dropped": int(chain_stats.get("dropped_nonfinite", 0)),
                        "selected_queries_by_family": query_counts,
                        "selected_slots": query_slots,
                        "gradient_tensors": len(gradients),
                        "gradient_norm": gradient_norm,
                        "gradients_finite": finite_gradients,
                        **probability,
                        "issues": dryrun_issues,
                    }
                )
                agent.zero_grad(set_to_none=True)

            game_records.append(
                {
                    "game": game_index,
                    "id": replay_id,
                    "rating": rating,
                    "players": replay.get("players", []),
                    "rows": len(rows),
                    "p1_rows": sum(row["role"] == "p1" for row in rows),
                    "p2_rows": sum(row["role"] == "p2" for row in rows),
                    "max_turn": max(int(row["turn"]) for row in rows),
                    "actions_force_added_to_public_mask": forced_support,
                    "protocol_preview_species": _protocol_preview_species(replay),
                    "terminal_labels_by_role": terminal_by_role,
                }
            )
        except Exception as error:  # noqa: BLE001 - an audit must preserve the failing game
            issues.append(f"{replay_id}: {type(error).__name__}: {error}")

    digest_after = _parameter_digest(agent)
    if digest_after != digest_before:
        issues.append("parameters changed during a zero-optimizer replay inspection")

    with trace_path.open("w", encoding="utf-8") as stream:
        for record in row_records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    passed = (
        len(game_records) == args.games
        and len(dryruns) >= args.games * 2
        and not issues
        and digest_before == digest_after
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_fingerprint": source_fingerprint(),
        "backend": "public_showdown_replay",
        "config": str(args.config),
        "device": str(device),
        "games_requested": args.games,
        "games_completed": len(game_records),
        "decision_rows": len(row_records),
        "perspective_dryruns": len(dryruns),
        "optimizer_steps": 0,
        "rating_min": min(ratings) if ratings else None,
        "rating_max": max(ratings) if ratings else None,
        "final_labels_by_family": dict(labels_total),
        "hidden_targets_by_family": dict(valid_total),
        "selected_queries_by_family": dict(selected_total),
        "selected_slots": sorted(selected_slots),
        "context_depth_min": min(context_depths) if context_depths else None,
        "context_depth_max": max(context_depths) if context_depths else None,
        "last_context_queries": last_queries,
        "parameter_digest_before": digest_before,
        "parameter_digest_after": digest_after,
        "compatibility": {
            "loaded": len(compatibility.get("loaded", [])),
            "initialized": len(compatibility.get("initialized", [])),
        },
        "passed": passed,
        "error_count": len(issues),
        "issues": issues,
        "games": game_records,
        "dryruns": dryruns,
        "trace": str(trace_path.resolve()),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    latest = Path("runs/replay_preflight_audit/latest.json")
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw_replays"))
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--config", default="configs/set_ar_replay.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/replay_preflight_audit/latest_run"),
    )
    args = parser.parse_args()
    report = run(args)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
