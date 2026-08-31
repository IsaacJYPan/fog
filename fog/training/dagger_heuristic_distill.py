"""DAgger-style policy correction on states reached by the live policy.

The actor executes its own deterministic policy in authoritative direct Showdown.  At each accepted
decision either SimpleHeuristics or a separately configured checkpoint is queried only as a
supervised label.  This avoids the teacher-forced state distribution that made ordinary behavior
cloning brittle at gameplay time.  A checkpoint teacher receives the exact same player-visible
Observation as the actor; it never executes the battle and never receives simulator-private data.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from fog.training.heuristic_distill import _POLICY_KEYS
from fog.training.outcome_heuristic_distill import team_holdout


def correction_weight(executed: int, target: int, *, disagreement: float = 2.0) -> float:
    """Give correction states more weight while retaining teacher agreement examples."""
    if disagreement < 1.0:
        raise ValueError("disagreement weight must be at least one")
    return float(disagreement if int(executed) != int(target) else 1.0)


def _policy_output_parameters(policy_value) -> list:
    """Return the parameters that directly map policy features to one action logit.

    Typed agents use the candidate-query scorer instead of ``policy``.  Keeping this
    routing in one helper prevents output-only corrections from accidentally
    unfreezing the unused legacy policy layer.
    """
    output = (
        policy_value.action_query_scorer[-1]
        if policy_value.action_query_policy
        else policy_value.policy[-1]
    )
    return list(output.parameters())


def action_family(action: int) -> str:
    """Return the stable project action family for a 14-way action index."""
    if 0 <= action < 4:
        return "ordinary_move"
    if action < 8:
        return "tera_move"
    if action < 13:
        return "switch"
    if action == 13:
        return "forced"
    raise ValueError(f"action index out of range: {action}")


def summarize_corrections(rows: list[dict], *, team_count: int) -> dict:
    """Summarize where donor actions and heuristic labels disagree without training."""
    families = ("ordinary_move", "tera_move", "switch", "forced")

    def empty() -> dict:
        return {
            "rows": 0,
            "disagreements": 0,
            "executed_by_slot": [0] * 14,
            "target_by_slot": [0] * 14,
            "executed_by_family": {family: 0 for family in families},
            "target_by_family": {family: 0 for family in families},
            "family_confusion": {
                source: {target: 0 for target in families} for source in families
            },
        }

    overall = empty()
    by_opponent: dict[str, dict] = {}
    for row in rows:
        executed, target = int(row["executed"]), int(row["target"])
        executed_family, target_family = action_family(executed), action_family(target)
        for bucket in (overall, by_opponent.setdefault(str(row["opponent"]), empty())):
            bucket["rows"] += 1
            bucket["disagreements"] += int(executed != target)
            bucket["executed_by_slot"][executed] += 1
            bucket["target_by_slot"][target] += 1
            bucket["executed_by_family"][executed_family] += 1
            bucket["target_by_family"][target_family] += 1
            bucket["family_confusion"][executed_family][target_family] += 1

    for bucket in (overall, *by_opponent.values()):
        bucket["disagreement_rate"] = (
            bucket["disagreements"] / bucket["rows"] if bucket["rows"] else None
        )
    return {
        "games": len({int(row["game"]) for row in rows}),
        "rows": len(rows),
        "team_count": team_count,
        "teams_covered": len({int(row["team"]) for row in rows}),
        "overall": overall,
        "by_opponent": by_opponent,
    }


def within_family_eligible(executed: int, target: int) -> bool:
    """Keep only comparisons within teacher-supported strategic families."""
    executed_family, target_family = action_family(executed), action_family(target)
    return executed_family == target_family and target_family in {"ordinary_move", "switch"}


def conditional_family_log_probs(log_probs, labels):
    """Renormalize each row inside its ordinary-move or switch family."""
    import torch

    if log_probs.ndim != 2 or log_probs.shape[1] != 14:
        raise ValueError("expected [batch, 14] log probabilities")
    mask = torch.zeros_like(log_probs, dtype=torch.bool)
    ordinary = labels < 4
    switches = (labels >= 8) & (labels < 13)
    mask[ordinary, :4] = True
    mask[switches, 8:13] = True
    if not bool((ordinary | switches).all()):
        raise ValueError("conditional labels must be ordinary moves or switches")
    masked = log_probs.masked_fill(~mask, float("-inf"))
    return masked - torch.logsumexp(masked, dim=-1, keepdim=True), mask


def summarize_policy_shift(rows: list[dict], predictions: list[int]) -> dict:
    """Compare checkpoint argmaxes with accepted donor actions on identical observations."""
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have equal length")

    def empty() -> dict:
        return {
            "rows": 0,
            "changed": 0,
            "teacher_agreement_before": 0,
            "teacher_agreement_after": 0,
            "changed_toward_teacher": 0,
            "changed_away_from_teacher": 0,
            "changed_other": 0,
            "action_transitions": [[0] * 14 for _ in range(14)],
        }

    overall = empty()
    groupings = {
        "by_opponent": {},
        "by_executed_family": {},
        "by_target_family": {},
        "by_outcome": {},
    }

    for row, predicted_value in zip(rows, predictions, strict=True):
        executed, target, predicted = (
            int(row["executed"]), int(row["target"]), int(predicted_value)
        )
        if not 0 <= predicted < 14:
            raise ValueError(f"predicted action out of range: {predicted}")
        outcome = float(row["outcome"])
        keys = {
            "by_opponent": str(row["opponent"]),
            "by_executed_family": action_family(executed),
            "by_target_family": action_family(target),
            "by_outcome": "win" if outcome > 0 else "loss" if outcome < 0 else "draw",
        }
        buckets = [overall]
        for grouping, key in keys.items():
            buckets.append(groupings[grouping].setdefault(key, empty()))
        for bucket in buckets:
            changed = predicted != executed
            bucket["rows"] += 1
            bucket["changed"] += int(changed)
            bucket["teacher_agreement_before"] += int(executed == target)
            bucket["teacher_agreement_after"] += int(predicted == target)
            bucket["changed_toward_teacher"] += int(changed and predicted == target)
            bucket["changed_away_from_teacher"] += int(changed and executed == target)
            bucket["changed_other"] += int(
                changed and predicted != target and executed != target
            )
            bucket["action_transitions"][executed][predicted] += 1

    for bucket in (overall, *(item for group in groupings.values() for item in group.values())):
        count = bucket["rows"]
        bucket["changed_rate"] = bucket["changed"] / count if count else None
        bucket["teacher_agreement_before_rate"] = (
            bucket["teacher_agreement_before"] / count if count else None
        )
        bucket["teacher_agreement_after_rate"] = (
            bucket["teacher_agreement_after"] / count if count else None
        )
    return {"overall": overall, **groupings}


def summarize_sampling_probabilities(
    log_probs,
    legality_mask,
    temperatures: list[float],
    tera_logit_bonuses: list[float] | None = None,
) -> dict:
    """Report how much probability positive temperatures assign to each action family.

    This is a read-only calibration on retained observations.  It avoids guessing that a higher
    temperature actually reaches a suppressed family such as Tera moves before paying for a
    training run.
    """
    from fog.env.player import _behavior_log_probs

    if log_probs.ndim != 2 or log_probs.shape[1] != 14:
        raise ValueError("expected [rows, 14] log probabilities")
    if legality_mask.shape != log_probs.shape:
        raise ValueError("legality mask must match log probabilities")
    greedy = log_probs.argmax(dim=-1)
    tera_legal = legality_mask[:, 4:8].any(dim=-1)
    report: dict[str, dict[str, float | int]] = {}
    settings = [(float(value), 0.0) for value in temperatures]
    settings.extend((1.0, float(value)) for value in (tera_logit_bonuses or []))
    for temperature, bonus in settings:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("comparison temperatures must be finite and positive")
        if not np.isfinite(bonus) or bonus < 0:
            raise ValueError("comparison Tera bonuses must be finite and non-negative")
        probabilities = _behavior_log_probs(
            log_probs,
            temperature=temperature,
            tera_logit_bonus=bonus,
        ).exp()
        greedy_probability = probabilities.gather(1, greedy[:, None]).squeeze(1)
        tera_probability = probabilities[:, 4:8].sum(dim=-1)
        key = f"temperature_{temperature:g}" if not bonus else f"tera_bonus_{bonus:g}"
        report[key] = {
            "temperature": temperature,
            "tera_logit_bonus": bonus,
            "rows": int(log_probs.shape[0]),
            "tera_legal_rows": int(tera_legal.sum()),
            "expected_ordinary_move_rate": float(probabilities[:, :4].sum(dim=-1).mean()),
            "expected_tera_move_rate": float(tera_probability.mean()),
            "expected_switch_rate": float(probabilities[:, 8:13].sum(dim=-1).mean()),
            "expected_forced_rate": float(probabilities[:, 13].mean()),
            "expected_non_greedy_rate": float((1.0 - greedy_probability).mean()),
            "expected_tera_when_legal_rate": (
                float(tera_probability[tera_legal].mean()) if bool(tera_legal.any()) else 0.0
            ),
        }
    return report


def summarize_teacher_agreement(rows: list[dict]) -> dict:
    """Compare checkpoint and scripted teachers on identical player-visible states."""
    eligible = [row for row in rows if "simple_target" in row and "maxdamage_target" in row]
    if not eligible:
        return {"rows": 0}

    def comparison(left: str, right: str) -> dict:
        exact = sum(int(row[left]) == int(row[right]) for row in eligible)
        family = sum(
            action_family(int(row[left])) == action_family(int(row[right])) for row in eligible
        )
        return {
            "exact_agreement": exact / len(eligible),
            "family_agreement": family / len(eligible),
        }

    return {
        "rows": len(eligible),
        "checkpoint_vs_simple": comparison("target", "simple_target"),
        "checkpoint_vs_maxdamage": comparison("target", "maxdamage_target"),
        "simple_vs_maxdamage": comparison("simple_target", "maxdamage_target"),
    }


def consensus_rows(rows: list[dict], teacher: str) -> list[dict]:
    """Select high-confidence rows where the checkpoint and scripted teacher agree exactly."""
    key = {"simple": "simple_target", "maxdamage": "maxdamage_target"}.get(teacher)
    if key is None:
        raise ValueError(f"unsupported consensus teacher {teacher!r}")
    missing = sum(key not in row for row in rows)
    if missing:
        raise ValueError(f"{missing} rows lack {key}; enable --audit-heuristic-labels")
    return [row for row in rows if int(row["target"]) == int(row[key])]


def compare_checkpoints_on_rows(
    cfg: dict,
    rows: list[dict],
    checkpoints: list[tuple[str, str]],
    *,
    batch_size: int,
    temperatures: list[float] | None = None,
    tera_logit_bonuses: list[float] | None = None,
) -> dict:
    """Forward checkpoints on retained donor states; this is read-only analysis."""
    import torch

    from fog.models.agent import FogAgent
    from fog.training.rollout import _stack_frags

    if batch_size < 1:
        raise ValueError("comparison batch size must be positive")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    report: dict[str, dict] = {}
    for label, checkpoint in checkpoints:
        agent = FogAgent.from_tables(cfg).to(device)
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        agent.eval()
        predictions: list[int] = []
        probability_rows: list[torch.Tensor] = []
        legality_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                part = rows[start : start + batch_size]
                batch = _stack_frags([row["frag"] for row in part], device=device)
                output = agent(batch)
                predictions.extend(output.log_probs.argmax(-1).cpu().tolist())
                if temperatures or tera_logit_bonuses:
                    probability_rows.append(output.log_probs.detach().cpu())
                    legality_rows.append(batch["legality_mask"].detach().cpu())
        report[label] = {
            "checkpoint": checkpoint,
            **summarize_policy_shift(rows, predictions),
        }
        if temperatures or tera_logit_bonuses:
            report[label]["sampling_probabilities"] = summarize_sampling_probabilities(
                torch.cat(probability_rows),
                torch.cat(legality_rows),
                temperatures or [],
                tera_logit_bonuses,
            )
        del agent
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def _checkpoint_arg(value: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, path


def _collection_opponent_selector(game_number: int, workers: int) -> int:
    """Balance opponent types within every statically partitioned worker.

    ``partition_games`` assigns round-robin game numbers to workers. Selecting the opponent with
    ``game_number % 4`` therefore correlates workers with opponent types whenever four workers are
    used and leaves the self-play worker as a severe straggler. Each worker instead advances
    through the same four-way opponent cycle as it advances through its own assignment.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    return (game_number // workers) % 4


def collect_rows(
    cfg: dict,
    checkpoint: str,
    *,
    games: int,
    workers: int,
    actor_threads: int,
    seed: int,
    retain_public_action_features: bool = False,
    teacher_cfg: dict | None = None,
    teacher_checkpoint: str | None = None,
    teacher_compatible: bool = False,
    audit_heuristic_labels: bool = False,
    outcome_arbitration_repeats: int = 0,
    outcome_arbitration_max_per_game: int = 0,
    outcome_arbitration_stride: int = 1,
    outcome_min_simple_better: int = 2,
) -> tuple[list[dict], int]:
    """Collect teacher corrections without letting truth or the teacher choose live actions."""
    import torch

    from fog.env.actions import map_order_to_action
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import _scripted_order, bind_fog_player, play_local_battle
    from fog.env.teams import load_pool, to_packed
    from fog.models.agent import FogAgent
    from fog.training.direct_parallel import WorkerGame, partition_games, run_workers
    from fog.training.rollout import RolloutBuffer

    pool = load_pool(cfg["env"]["team_pool"])
    packed = [to_packed(paste) for paste in pool]
    schedule_rng = random.Random(seed)
    jobs = [
        WorkerGame(
            worker_id=-1,
            game_number=index,
            seed=schedule_rng.getrandbits(256),
            p1_index=index % len(pool),
            p2_index=schedule_rng.randrange(len(pool)),
        )
        for index in range(games)
    ]
    assignments = partition_games(jobs, workers)
    showdown_root = cfg.get("env", {}).get("showdown_root")

    def worker(worker_id: int, assigned: list[WorkerGame]) -> list[dict]:
        torch.set_num_threads(max(1, actor_threads))
        agent = FogAgent.from_tables(cfg).cpu()
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        agent.eval()
        teacher = None
        if (teacher_cfg is None) != (teacher_checkpoint is None):
            raise ValueError("teacher config and checkpoint must be supplied together")
        if teacher_cfg is not None and teacher_checkpoint is not None:
            teacher = FogAgent.from_tables(teacher_cfg).cpu()
            teacher_state = torch.load(
                teacher_checkpoint, map_location="cpu", weights_only=False
            )
            if teacher_compatible:
                teacher.load_compatible_state_dict(teacher_state)
            else:
                teacher.load_state_dict(teacher_state)
            teacher.eval()
        actor = bind_fog_player(FogPlayer(agent, temperature=0.0))
        actor.record_mirror = False
        actor.record_policy_reference = True
        donor_opponent = bind_fog_player(FogPlayer(agent, temperature=0.0))
        donor_opponent.record_mirror = False
        buffer = RolloutBuffer()
        actor.attach_buffer(buffer)
        teacher_rng = random.Random(seed ^ (worker_id + 1) * 0x9E3779B1)
        simple_audit_rng = random.Random(seed ^ (worker_id + 1) * 0xD1B54A32D192ED03)
        maxdamage_audit_rng = random.Random(seed ^ (worker_id + 1) * 0x94D049BB133111EB)
        game_audit_labels: list[dict[str, int]] = []
        game_outcome_labels: list[dict] = []
        game_arbitrated = 0
        game_disagreements = 0
        game_decision = 0

        def label(battle, observation, _output) -> int:
            nonlocal game_arbitrated, game_disagreements, game_decision
            executed = int(_output.log_probs[0].argmax().item())
            if teacher is not None:
                from fog.encoding.batch import observation_to_batch

                with torch.no_grad():
                    teacher_output = teacher(observation_to_batch(observation, "cpu"))
                target = int(teacher_output.log_probs[0].argmax().item())
                if not bool(observation.legality_mask[target]):
                    raise ValueError(f"checkpoint teacher returned illegal action {target}")
            else:
                target = map_order_to_action(
                    battle, _scripted_order("simpleheuristics", battle, teacher_rng)
                )
            outcome_row = {
                "outcome_arbitrated": False,
                "outcome_selected": False,
                "outcome_pairs": [],
            }
            if outcome_arbitration_repeats > 0 and executed != target:
                from fog.env.actions import map_action_to_order
                from fog.env.showdown_local_runner import LocalOrderFactory, _choice
                from fog.eval.policy_disagreement_rollout import (
                    _branch_views,
                    _rollout,
                    select_simple_outcome_label,
                )

                game_disagreements += 1
                selected_stride = (
                    (game_disagreements - 1) % max(1, outcome_arbitration_stride) == 0
                )
                within_cap = (
                    outcome_arbitration_max_per_game <= 0
                    or game_arbitrated < outcome_arbitration_max_per_game
                )
                if selected_stride and within_cap and executed < 4 and target < 4:
                    local = getattr(battle, "_fog_local_battle", None)
                    if local is None:
                        raise RuntimeError("outcome arbitration requires direct local battle")
                    role = getattr(battle, "player_role", None) or "p1"
                    opponent_role = "p2" if role == "p1" else "p1"

                    def action_choice(action: int) -> str:
                        order = map_action_to_order(battle, action, LocalOrderFactory())
                        return str(getattr(order, "message", order))

                    opponent_choice = _choice(
                        "simpleheuristics",
                        local.views[opponent_role],
                        local.update.requests.get(opponent_role),
                        teacher_rng,
                    )
                    root_state = local.state
                    root_views = _branch_views(local)
                    pairs = []
                    for repeat in range(outcome_arbitration_repeats):
                        pair_seed = (
                            seed + int(game_decision) * 101 + repeat
                            + int(getattr(battle, "turn", 0) or 0) * 100_003
                        )
                        pair = {"seed": pair_seed}
                        try:
                            policy_branch = _rollout(
                                client,
                                state=root_state,
                                views=root_views,
                                requests=local.update.requests,
                                learner_role=role,
                                root_learner_choice=action_choice(executed),
                                root_opponent_choice=opponent_choice,
                                seed=pair_seed,
                                max_turns=160,
                            )
                            simple_branch = _rollout(
                                client,
                                state=root_state,
                                views=root_views,
                                requests=local.update.requests,
                                learner_role=role,
                                root_learner_choice=action_choice(target),
                                root_opponent_choice=opponent_choice,
                                seed=pair_seed,
                                max_turns=160,
                            )
                            pair.update(
                                policy_outcome=policy_branch["outcome"],
                                simple_outcome=simple_branch["outcome"],
                                error=None,
                            )
                        except Exception as exc:  # noqa: BLE001 - bounded offline evaluator
                            pair.update(
                                policy_outcome=None,
                                simple_outcome=None,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        pairs.append(pair)
                    selected = select_simple_outcome_label(
                        executed,
                        target,
                        pairs,
                        min_simple_better=outcome_min_simple_better,
                    )
                    outcome_row = {
                        "outcome_arbitrated": True,
                        "outcome_selected": bool(selected),
                        "outcome_pairs": pairs,
                    }
                    game_arbitrated += 1
                    if not selected:
                        target = executed
            game_outcome_labels.append(outcome_row)
            game_decision += 1
            if audit_heuristic_labels:
                game_audit_labels.append(
                    {
                        "simple_target": map_order_to_action(
                            battle,
                            _scripted_order("simpleheuristics", battle, simple_audit_rng),
                        ),
                        "maxdamage_target": map_order_to_action(
                            battle,
                            _scripted_order("maxdamage", battle, maxdamage_audit_rng),
                        ),
                    }
                )
            return target

        actor.supervised_label_provider = label
        rows: list[dict] = []
        client_kwargs = {"showdown_root": showdown_root} if showdown_root else {}
        with ShowdownDirectClient(**client_kwargs) as client:
            for game in assigned:
                game_audit_labels.clear()
                game_outcome_labels.clear()
                game_arbitrated = 0
                game_disagreements = 0
                game_decision = 0
                selector = _collection_opponent_selector(game.game_number, workers)
                opponent_name = (
                    "simpleheuristics" if selector < 2 else "maxdamage" if selector == 2 else "donor"
                )
                opponent = donor_opponent if opponent_name == "donor" else opponent_name
                play_local_battle(
                    client,
                    {
                        "formatid": cfg["env"].get("battle_format", "gen9ou"),
                        "seed": f"sodium,{game.seed:064x}",
                        "strictChoices": True,
                        "p1": {"name": f"Actor{worker_id}", "team": packed[game.p1_index]},
                        "p2": {"name": "Opponent", "team": packed[game.p2_index]},
                    },
                    actor,
                    opponent,
                    seed=game.seed & (2**31 - 1),
                    battle_tag=game.battle_tag,
                    max_turns=160,
                )
                if len(buffer._games) != 1:
                    raise RuntimeError("expected exactly one finalized DAgger game")
                finalized = next(iter(buffer._games.values()))
                if finalized.outcome is None:
                    raise RuntimeError("DAgger game was not finalized")
                audit_labels = game_audit_labels if audit_heuristic_labels else [None] * len(finalized.frags)
                if len(game_outcome_labels) != len(finalized.frags):
                    raise RuntimeError("outcome arbitration rows do not align with decisions")
                for frag, executed, audit_label, outcome_label in zip(
                    finalized.frags,
                    finalized.actions,
                    audit_labels,
                    game_outcome_labels,
                    strict=True,
                ):
                    if "supervised_label" not in frag or "reference_log_probs" not in frag:
                        raise RuntimeError("accepted DAgger row lacks label or policy reference")
                    rows.append(
                        {
                            "game": int(game.game_number),
                            "team": int(game.p1_index),
                            "opponent": opponent_name,
                            "outcome": float(finalized.outcome),
                            "executed": int(executed),
                            "target": int(frag["supervised_label"]),
                            "reference_log_probs": np.asarray(
                                frag["reference_log_probs"], dtype=np.float32
                            ),
                            "frag": {
                                key: frag[key]
                                for key in _POLICY_KEYS
                                if key in frag and key != "action_features"
                            },
                        }
                    )
                    if audit_label is not None:
                        rows[-1].update(audit_label)
                    rows[-1].update(outcome_label)
                    if retain_public_action_features:
                        from fog.encoding.public_action_features import public_action_features

                        if "action_features" not in frag:
                            raise RuntimeError("public action features were requested but absent")
                        rows[-1]["frag"]["action_features"] = public_action_features(
                            frag["action_features"]
                        )
                buffer.clear()
                completed = sum(g.game_number <= game.game_number for g in assigned)
                if completed <= 5 or completed % 10 == 0:
                    print(
                        f"[model-reached-collect] worker={worker_id} "
                        f"games={completed} through_game={game.game_number} rows={len(rows)}",
                        flush=True,
                    )
        return rows

    nested = run_workers(assignments, worker)
    rows = [row for worker_rows in nested for row in worker_rows]
    rows.sort(key=lambda row: (row["game"], row["team"]))
    return rows, len(pool)


def train_policy(
    cfg: dict,
    checkpoint: str,
    rows: list[dict],
    *,
    team_count: int,
    epochs: int,
    batch_size: int,
    lr: float,
    kl_coef: float,
    disagreement_weight: float,
    seed: int,
    out_dir: Path,
    within_family_output_only: bool = False,
) -> dict:
    """Conservatively update only the policy head on on-policy correction states."""
    import torch

    from fog.models.agent import FogAgent
    from fog.training.ppo import _masked_kl
    from fog.training.rollout import _stack_frags

    if not rows:
        raise ValueError("no DAgger rows collected")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    donor_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student = FogAgent.from_tables(cfg).to(device)
    student.load_state_dict(donor_state)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    if within_family_output_only:
        parameters = _policy_output_parameters(student.policy_value)
    else:
        from fog.training.bc import _policy_modules

        parameters = [
            parameter for module in _policy_modules(student) for parameter in module.parameters()
        ]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)

    holdout_teams = team_holdout(team_count, 0.1, seed)
    eligible = np.asarray(
        [
            (not within_family_output_only)
            or within_family_eligible(row["executed"], row["target"])
            for row in rows
        ],
        dtype=bool,
    )
    train_indices = np.asarray(
        [i for i, row in enumerate(rows) if int(row["team"]) not in holdout_teams],
        dtype=np.int64,
    )
    train_indices = train_indices[eligible[train_indices]]
    hold_indices = np.asarray(
        [i for i, row in enumerate(rows) if int(row["team"]) in holdout_teams],
        dtype=np.int64,
    )
    hold_indices = hold_indices[eligible[hold_indices]]
    if not len(train_indices) or not len(hold_indices):
        raise ValueError("conditional DAgger split has no eligible train or holdout rows")
    weights = np.asarray(
        [
            correction_weight(
                row["executed"], row["target"], disagreement=disagreement_weight
            )
            for row in rows
        ],
        dtype=np.float32,
    )

    def make_batch(indices: np.ndarray):
        batch = _stack_frags([rows[int(i)]["frag"] for i in indices], device=device)
        labels = torch.as_tensor(
            [rows[int(i)]["target"] for i in indices], dtype=torch.long, device=device
        )
        reference = torch.as_tensor(
            np.stack([rows[int(i)]["reference_log_probs"] for i in indices]),
            dtype=torch.float32,
            device=device,
        )
        batch_weights = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
        return batch, labels, reference, batch_weights

    def losses(output_log_probs, labels, reference):
        if not within_family_output_only:
            nll = -output_log_probs.gather(1, labels[:, None]).squeeze(1)
            return nll, _masked_kl(output_log_probs, reference)
        conditional, mask = conditional_family_log_probs(output_log_probs, labels)
        reference_conditional, _ = conditional_family_log_probs(reference, labels)
        nll = -conditional.gather(1, labels[:, None]).squeeze(1)
        finite_mask = mask & torch.isfinite(reference_conditional) & torch.isfinite(conditional)
        reference_probs = torch.where(finite_mask, reference_conditional.exp(), 0.0)
        difference = torch.where(
            finite_mask,
            torch.where(
                finite_mask,
                reference_conditional,
                torch.zeros_like(reference_conditional),
            )
            - torch.where(finite_mask, conditional, torch.zeros_like(conditional)),
            0.0,
        )
        return nll, (reference_probs * difference).sum(-1)

    @torch.no_grad()
    def evaluate(indices: np.ndarray) -> dict[str, float]:
        student.eval()
        nll_sum = correct = kl_sum = 0.0
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch, labels, reference, _ = make_batch(part)
            output = student(batch)
            nll, kl = losses(output.log_probs, labels, reference)
            nll_sum += float(nll.sum())
            predictions = (
                conditional_family_log_probs(output.log_probs, labels)[0].argmax(-1)
                if within_family_output_only
                else output.log_probs.argmax(-1)
            )
            correct += float((predictions == labels).sum())
            kl_sum += float(kl.sum())
        return {
            "nll": nll_sum / len(indices),
            "top1": correct / len(indices),
            "kl_to_actor": kl_sum / len(indices),
        }

    baseline = evaluate(hold_indices)
    history: list[dict] = []
    best_nll = baseline["nll"]
    best_epoch = 0
    generator = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        student.train()
        order = generator.permutation(train_indices)
        train_sum = 0.0
        for start in range(0, len(order), batch_size):
            part = order[start : start + batch_size]
            batch, labels, reference, batch_weights = make_batch(part)
            output = student(batch)
            nll, row_kl = losses(output.log_probs, labels, reference)
            weighted_nll = (batch_weights * nll).sum() / batch_weights.sum().clamp_min(1e-8)
            kl = row_kl.mean()
            loss = weighted_nll + kl_coef * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite DAgger loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite DAgger gradient at epoch {epoch}")
            optimizer.step()
            train_sum += float(loss.detach()) * len(part)
        metrics = evaluate(hold_indices)
        metrics.update(epoch=epoch, train_loss=train_sum / len(train_indices))
        history.append(metrics)
        print(
            f"[dagger-distill] epoch={epoch} train={metrics['train_loss']:.5f} "
            f"hold_nll={metrics['nll']:.5f} hold_top1={metrics['top1']:.4f} "
            f"hold_kl={metrics['kl_to_actor']:.5f}",
            flush=True,
        )
        torch.save(student.state_dict(), out_dir / "student_latest.pt")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            torch.save(student.state_dict(), out_dir / "student_best.pt")

    if best_epoch == 0:
        raise RuntimeError("whole-team heldout DAgger NLL did not improve")
    best_state = torch.load(out_dir / "student_best.pt", map_location="cpu", weights_only=False)
    trainable_ids = {id(parameter) for parameter in parameters}
    allowed_names = {
        name
        for name, parameter in student.named_parameters(remove_duplicate=False)
        if id(parameter) in trainable_ids
    }
    changed = [
        name
        for name, donor_tensor in donor_state.items()
        if name not in allowed_names
        and not torch.equal(donor_tensor.cpu(), best_state[name].cpu())
    ]
    if changed:
        raise RuntimeError(f"protected tensors changed during DAgger: {changed[:8]}")

    disagreements = sum(row["executed"] != row["target"] for row in rows)
    report = {
        "games": len({row["game"] for row in rows}),
        "rows": len(rows),
        "team_count": team_count,
        "holdout_teams": sorted(holdout_teams),
        "train_rows": len(train_indices),
        "holdout_rows": len(hold_indices),
        "disagreements": disagreements,
        "disagreement_rate": disagreements / len(rows),
        "lr": lr,
        "kl_coef": kl_coef,
        "disagreement_weight": disagreement_weight,
        "baseline": baseline,
        "history": history,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "outside_policy_byte_identical": True,
        "checkpoint": str(out_dir / "student_best.pt"),
        "within_family_output_only": within_family_output_only,
        "eligible_rows": int(eligible.sum()),
        "outside_output_byte_identical": bool(within_family_output_only),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    from fog.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-config", default=None,
                        help="architecture config for an optional checkpoint teacher")
    parser.add_argument("--teacher-checkpoint", default=None,
                        help="optional stronger checkpoint used only to label actor-reached states")
    parser.add_argument(
        "--teacher-compatible", action="store_true",
        help="explicitly use behavior-preserving compatible loading for the historical teacher",
    )
    parser.add_argument(
        "--audit-heuristic-labels", action="store_true",
        help="record SimpleHeuristics and MaxDamage labels on the same visible actor states",
    )
    parser.add_argument(
        "--consensus-with", choices=("simple", "maxdamage"), default=None,
        help="train only on exact checkpoint/scripted-teacher agreement rows",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actor-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kl-coef", type=float, default=0.5)
    parser.add_argument("--disagreement-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--outcome-arbitration-repeats", type=int, default=0)
    parser.add_argument("--outcome-arbitration-max-per-game", type=int, default=0)
    parser.add_argument("--outcome-arbitration-stride", type=int, default=1)
    parser.add_argument("--outcome-min-simple-better", type=int, default=2)
    parser.add_argument(
        "--outcome-selected-only",
        action="store_true",
        help="train only on ordinary-move labels selected by paired offline outcomes",
    )
    parser.add_argument("--collect-only", action="store_true",
                        help="write action-family correction diagnostics without training")
    parser.add_argument("--within-family-output-only", action="store_true",
                        help="learn conditional move/switch choices in the final output layer only")
    parser.add_argument("--compare-checkpoint", action="append", type=_checkpoint_arg, default=[],
                        metavar="LABEL=PATH",
                        help="in collect-only mode, forward a checkpoint on every retained row")
    parser.add_argument("--compare-batch-size", type=int, default=512)
    parser.add_argument(
        "--compare-temperature",
        action="append",
        type=float,
        default=[],
        help="read-only sampling-probability calibration on retained rows (repeatable)",
    )
    parser.add_argument(
        "--compare-tera-logit-bonus",
        action="append",
        type=float,
        default=[],
        help="read-only Tera-only exploration calibration on retained rows (repeatable)",
    )
    parser.add_argument(
        "--retain-public-action-features",
        action="store_true",
        help=(
            "retain answer-free public action features for exact collect-only checkpoint "
            "comparisons; never retains SimpleHeuristics answer columns"
        ),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    if (args.teacher_config is None) != (args.teacher_checkpoint is None):
        parser.error("--teacher-config and --teacher-checkpoint must be supplied together")
    teacher_cfg = load_config(args.teacher_config) if args.teacher_config else None
    started = time.monotonic()
    rows, team_count = collect_rows(
        cfg,
        args.checkpoint,
        games=args.games,
        workers=args.workers,
        actor_threads=args.actor_threads,
        seed=args.seed,
        retain_public_action_features=args.retain_public_action_features,
        teacher_cfg=teacher_cfg,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_compatible=args.teacher_compatible,
        audit_heuristic_labels=args.audit_heuristic_labels,
        outcome_arbitration_repeats=args.outcome_arbitration_repeats,
        outcome_arbitration_max_per_game=args.outcome_arbitration_max_per_game,
        outcome_arbitration_stride=args.outcome_arbitration_stride,
        outcome_min_simple_better=args.outcome_min_simple_better,
    )
    print(
        f"[dagger-distill] collected rows={len(rows)} games={args.games} "
        f"teams={team_count} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    if args.collect_only:
        report = summarize_corrections(rows, team_count=team_count)
        report["checkpoint"] = args.checkpoint
        report["seed"] = args.seed
        report["validation_used"] = False
        report["trained"] = False
        report["teacher_checkpoint"] = args.teacher_checkpoint
        report["teacher_compatible"] = args.teacher_compatible
        report["teacher_agreement"] = summarize_teacher_agreement(rows)
        report["outcome_arbitrated_rows"] = sum(
            bool(row.get("outcome_arbitrated", False)) for row in rows
        )
        report["outcome_selected_rows"] = sum(
            bool(row.get("outcome_selected", False)) for row in rows
        )
        report["outcome_arbitration_errors"] = sum(
            bool(pair.get("error"))
            for row in rows
            for pair in row.get("outcome_pairs", [])
        )
        if args.compare_checkpoint:
            if len(dict(args.compare_checkpoint)) != len(args.compare_checkpoint):
                parser.error("comparison checkpoint labels must be unique")
            report["checkpoint_comparisons"] = compare_checkpoints_on_rows(
                cfg,
                rows,
                args.compare_checkpoint,
                batch_size=args.compare_batch_size,
                temperatures=args.compare_temperature,
                tera_logit_bonuses=args.compare_tera_logit_bonus,
            )
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    training_rows = (
        consensus_rows(rows, args.consensus_with) if args.consensus_with is not None else rows
    )
    if args.outcome_selected_only:
        training_rows = [row for row in training_rows if row.get("outcome_selected", False)]
        # Outcome arbitration is the expensive part of this workflow.  Persist the
        # compact selected subset before optimization so a technical training failure
        # never forces the authoritative paired rollouts to be collected again.
        import torch

        torch.save(training_rows, args.out / "outcome_selected_rows.pt")
    report = train_policy(
        cfg,
        args.checkpoint,
        training_rows,
        team_count=team_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        disagreement_weight=args.disagreement_weight,
        seed=args.seed,
        out_dir=args.out,
        within_family_output_only=args.within_family_output_only,
    )
    report["teacher_checkpoint"] = args.teacher_checkpoint
    report["teacher_compatible"] = args.teacher_compatible
    report["validation_used"] = False
    report["collected_rows"] = len(rows)
    report["consensus_with"] = args.consensus_with
    report["consensus_rows"] = len(training_rows)
    report["outcome_selected_only"] = args.outcome_selected_only
    report["outcome_arbitrated_rows"] = sum(
        bool(row.get("outcome_arbitrated", False)) for row in rows
    )
    report["outcome_selected_rows"] = sum(
        bool(row.get("outcome_selected", False)) for row in rows
    )
    report["teacher_agreement"] = summarize_teacher_agreement(rows)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
