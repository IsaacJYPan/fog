"""Measure critic error against repeated continuation values on censored actor states.

The accepted policy plays training-team games against SimpleHeuristics.  At a sparse, spread set
of its decision states, the evaluator snapshots authoritative simulator state *offline* and runs
independent continuations of the same accepted policy versus Simple.  The actor and every branch
policy receive only their ordinary player-visible views.  Repeated outcomes estimate the expected
continuation target and its stochastic variance; no weights are trained or saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _normalize_choice(choice: str | None) -> str:
    if choice is None:
        return ""
    value = str(choice).strip()
    if value.startswith("/choose "):
        return value[8:]
    if value.startswith("/team "):
        return "team " + value[6:]
    return value


def _sample_seed(seed: int, step: int) -> str:
    digest = hashlib.sha256(f"{seed}:{step}".encode()).hexdigest()
    return f"sodium,{digest}"


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def split_team_indices(
    pool_size: int, *, holdout_fraction: float, seed: int, split: str
) -> list[int]:
    """Return a deterministic disjoint subset of the training-team pool."""
    if pool_size < 2:
        raise ValueError("team split requires at least two teams")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    if split not in {"all", "train", "holdout"}:
        raise ValueError(f"unknown team split {split!r}")
    indices = list(range(pool_size))
    if split == "all":
        return indices
    random.Random(seed).shuffle(indices)
    count = max(1, round(pool_size * float(holdout_fraction)))
    count = min(count, pool_size - 1)
    holdout = sorted(indices[:count])
    train = sorted(indices[count:])
    return train if split == "train" else holdout


def summarize_value_targets(states: list[dict[str, Any]]) -> dict[str, Any]:
    valid = []
    for state in states:
        outcomes = [
            float(row["outcome"])
            for row in state.get("continuations", [])
            if row.get("outcome") is not None
        ]
        if len(outcomes) >= 2 and state.get("value_prediction") is not None:
            valid.append((state, outcomes))
    if not valid:
        return {"states": 0, "valid_continuations": 0}

    soft_errors = []
    hard_errors = []
    floors = []
    live_errors = []
    half_left: list[float] = []
    half_right: list[float] = []
    sign_disagreements = 0
    half_states = 0
    progress_predictions: list[float] = []
    progress_targets: list[float] = []
    for state, outcomes in valid:
        prediction = float(state["value_prediction"])
        target = sum(outcomes) / len(outcomes)
        soft_errors.append((prediction - target) ** 2)
        hard_errors.append(sum((prediction - outcome) ** 2 for outcome in outcomes) / len(outcomes))
        outcome_mean = target
        floors.append(
            sum((outcome - outcome_mean) ** 2 for outcome in outcomes) / (len(outcomes) - 1)
        )
        if state.get("live_outcome") is not None:
            live_errors.append((prediction - float(state["live_outcome"])) ** 2)
        if state.get("progress_score") is not None:
            progress_predictions.append(float(state["progress_score"]))
            progress_targets.append(target)
        midpoint = len(outcomes) // 2
        if midpoint and len(outcomes) - midpoint:
            left = sum(outcomes[:midpoint]) / midpoint
            right = sum(outcomes[midpoint:]) / (len(outcomes) - midpoint)
            half_left.append(left)
            half_right.append(right)
            half_states += 1
            sign_disagreements += (left > 0) != (right > 0)

    repeated_hard_mse = sum(hard_errors) / len(hard_errors)
    stochastic_floor = sum(floors) / len(floors)
    summary = {
        "states": len(valid),
        "valid_continuations": sum(len(outcomes) for _, outcomes in valid),
        "critic_mse_to_continuation_mean": sum(soft_errors) / len(soft_errors),
        "critic_mse_to_repeated_hard_outcomes": repeated_hard_mse,
        "unbiased_stochastic_outcome_variance": stochastic_floor,
        "reducible_excess_mse": repeated_hard_mse - stochastic_floor,
        "critic_mse_to_live_single_outcome": (
            sum(live_errors) / len(live_errors) if live_errors else None
        ),
        "split_half_mean_mse": (
            sum((a - b) ** 2 for a, b in zip(half_left, half_right)) / half_states
            if half_states else None
        ),
        "split_half_correlation": _correlation(half_left, half_right),
        "split_half_sign_disagreement_rate": (
            sign_disagreements / half_states if half_states else None
        ),
    }
    if progress_targets:
        summary.update({
            "progress_states": len(progress_targets),
            "progress_mse_to_continuation_mean": sum(
                (prediction - target) ** 2
                for prediction, target in zip(progress_predictions, progress_targets)
            ) / len(progress_targets),
            "progress_correlation_with_continuation_mean": _correlation(
                progress_predictions, progress_targets
            ),
            "progress_sign_accuracy": sum(
                (prediction > 0) == (target > 0)
                for prediction, target in zip(progress_predictions, progress_targets)
            ) / len(progress_targets),
        })
    return summary


def _branch_views(local: Any) -> dict[str, Any]:
    return {role: deepcopy(local.views[role], {id(local): None}) for role in ("p1", "p2")}


def _apply_sample(views: dict[str, Any], sampled: dict[str, Any]) -> None:
    from fog.env.showdown_local import LocalShowdownBattle

    for role in ("p1", "p2"):
        LocalShowdownBattle._apply_view_update(
            views[role], sampled["player_logs"][role], sampled["requests"].get(role)
        )


def _rollout(
    client: Any,
    *,
    state: dict[str, Any],
    views: dict[str, Any],
    learner_role: str,
    root_choices: dict[str, str | None],
    agent: Any,
    opponent_controller: str,
    seed: int,
    max_turns: int,
) -> dict[str, Any]:
    from fog.env.player import FogPlayer
    from fog.env.showdown_local_runner import _choice, bind_fog_player
    from fog.eval.policy_disagreement_rollout import _outcome_for_player

    branch_views = {role: deepcopy(view) for role, view in views.items()}
    branch_actor = bind_fog_player(FogPlayer(agent, temperature=0.0))
    sampled = client.sample(
        state,
        tuple(_normalize_choice(root_choices[role]) for role in ("p1", "p2")),
        seed=_sample_seed(seed, 0),
        include_state=True,
    )
    _apply_sample(branch_views, sampled)
    rng = random.Random(seed)
    steps = 1
    try:
        while not sampled["ended"] and int(sampled["turn"]) < int(max_turns):
            controllers = {
                learner_role: branch_actor,
                "p2" if learner_role == "p1" else "p1": opponent_controller,
            }
            choices = {
                role: _choice(
                    controllers[role], branch_views[role], sampled["requests"].get(role), rng
                )
                for role in ("p1", "p2")
            }
            if choices == {"p1": None, "p2": None}:
                raise RuntimeError("continuation branch has no pending choice")
            sampled = client.sample(
                sampled["state"],
                tuple(_normalize_choice(choices[role]) for role in ("p1", "p2")),
                seed=_sample_seed(seed, steps),
                include_state=True,
            )
            _apply_sample(branch_views, sampled)
            steps += 1
    finally:
        finished = getattr(branch_actor, "_battle_finished", None)
        if finished is not None:
            finished(branch_views[learner_role])
    winner = sampled.get("winner")
    username = str(getattr(branch_views[learner_role], "player_username", ""))
    outcome = None
    if sampled["ended"]:
        outcome = _outcome_for_player(winner, username)
    return {
        "outcome": outcome,
        "winner": winner,
        "turn": int(sampled["turn"]),
        "steps": steps,
        "truncated": not bool(sampled["ended"]),
    }


def main() -> None:
    import torch

    from fog.config import load_config_with_args
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.showdown_local_runner import _choice, bind_fog_player
    from fog.env.teams import load_pool, to_packed
    from fog.eval.policy_disagreement_rollout import _outcome_for_player
    from fog.eval.showdown_direct_selection import _options, matched_schedule
    from fog.models.agent import FogAgent
    from fog.training.progress_reward import battle_progress_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-states", type=int, default=32)
    parser.add_argument("--max-states-per-game", type=int, default=2)
    parser.add_argument("--state-stride", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261340)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--opponent",
        choices=("simpleheuristics", "maxdamage"),
        default="simpleheuristics",
    )
    parser.add_argument("--team-split", choices=("all", "train", "holdout"), default="all")
    parser.add_argument("--team-split-seed", type=int, default=20261346)
    parser.add_argument("--team-holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    agent.eval()

    latest_value: dict[str, Any] = {}
    latest_feature: dict[str, Any] = {}

    def capture_value_feature(_module: Any, inputs: tuple[Any, ...]) -> None:
        latest_feature["feature"] = inputs[0].detach().reshape(inputs[0].shape[0], -1)[0].cpu()

    value_feature_handle = agent.policy_value.value[-1].register_forward_pre_hook(
        capture_value_feature
    )

    def observe(_tag: str, _fragment: dict, output: Any) -> None:
        latest_value["value"] = float(output.value.reshape(-1)[0])
        if "feature" not in latest_feature:
            raise RuntimeError("value feature hook did not capture the actor forward")
        latest_value["feature"] = latest_feature["feature"].tolist()

    actor = bind_fog_player(FogPlayer(agent, temperature=0.0))
    actor.decision_observer = observe
    all_teams = load_pool(args.pool)
    selected_team_indices = split_team_indices(
        len(all_teams),
        holdout_fraction=args.team_holdout_fraction,
        seed=args.team_split_seed,
        split=args.team_split,
    )
    teams = [all_teams[index] for index in selected_team_indices]
    packed = [to_packed(team) for team in teams]
    schedule = matched_schedule(len(teams), args.games, args.seed)
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic_only": True,
        "trained": False,
        "validation_used_for_training": False,
        "truth_used_for_actor": False,
        "authoritative_state_used_offline_only": True,
        "continuation_policy": f"accepted censored policy versus {args.opponent}",
        "opponent": args.opponent,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "pool": str(args.pool),
        "team_split": args.team_split,
        "team_split_seed": args.team_split_seed,
        "team_holdout_fraction": args.team_holdout_fraction,
        "selected_team_indices": selected_team_indices,
        "games_target": args.games,
        "max_states": args.max_states,
        "max_states_per_game": args.max_states_per_game,
        "state_stride": args.state_stride,
        "repeats": args.repeats,
        "seed": args.seed,
        "games": [],
        "states": [],
        "errors": [],
    }
    _atomic_json(report_path, report)
    client_kwargs = {}
    showdown_root = cfg.get("env", {}).get("showdown_root")
    if showdown_root:
        client_kwargs["showdown_root"] = showdown_root
    started = time.monotonic()

    with ShowdownDirectClient(**client_kwargs) as client:
        for game_index, game in enumerate(schedule):
            learner_role = "p1" if game.p1_is_a else "p2"
            local = LocalShowdownBattle.create(
                client,
                _options("gen9ou", packed, game),
                battle_tag=f"value-continuation-control-{game_index}",
            )
            actual_rng = random.Random(game.controller_seed)
            game_rows: list[dict[str, Any]] = []
            decision_index = 0
            try:
                truncated = False
                while not local.ended:
                    if int(local.p1.turn) >= args.max_turns:
                        truncated = True
                        break
                    controllers = {
                        learner_role: actor,
                        "p2" if learner_role == "p1" else "p1": args.opponent,
                    }
                    latest_value.clear()
                    choices = {
                        role: _choice(
                            controllers[role], local.views[role],
                            local.update.requests.get(role), actual_rng,
                        )
                        for role in ("p1", "p2")
                    }
                    learner_request = local.update.requests.get(learner_role)
                    is_action = bool(
                        learner_request and not learner_request.get("wait")
                        and not learner_request.get("teamPreview")
                    )
                    eligible = (
                        is_action
                        and len(report["states"]) < args.max_states
                        and len(game_rows) < args.max_states_per_game
                        and decision_index % max(1, args.state_stride) == 0
                    )
                    if eligible:
                        if "value" not in latest_value:
                            raise RuntimeError("actor decision observer did not record a value")
                        state_row: dict[str, Any] = {
                            "game": game_index,
                            "turn": int(local.p1.turn),
                            "role": learner_role,
                            "value_prediction": latest_value["value"],
                            "value_feature": latest_value["feature"],
                            # Public HP/alive progress from the same censored view used by the
                            # actor.  This is a diagnostic target, never private simulator truth.
                            "progress_score": battle_progress_score(local.views[learner_role]),
                            "learner_team_index": selected_team_indices[
                                game.p1_team if game.p1_is_a else game.p2_team
                            ],
                            "opponent_team_index": selected_team_indices[
                                game.p2_team if game.p1_is_a else game.p1_team
                            ],
                            "live_outcome": None,
                            "continuations": [],
                        }
                        root_state = local.state
                        root_views = _branch_views(local)
                        for repeat in range(args.repeats):
                            repeat_seed = (
                                args.seed + game_index * 100_003 + decision_index * 101 + repeat
                            )
                            try:
                                continuation = _rollout(
                                    client,
                                    state=root_state,
                                    views=root_views,
                                    learner_role=learner_role,
                                    root_choices=choices,
                                    agent=agent,
                                    opponent_controller=args.opponent,
                                    seed=repeat_seed,
                                    max_turns=args.max_turns,
                                )
                                continuation["seed"] = repeat_seed
                                continuation["error"] = None
                            except Exception as exc:  # noqa: BLE001 - retain partial evidence
                                continuation = {
                                    "seed": repeat_seed,
                                    "outcome": None,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            state_row["continuations"].append(continuation)
                        report["states"].append(state_row)
                        game_rows.append(state_row)
                        report["summary"] = summarize_value_targets(report["states"])
                        _atomic_json(report_path, report)
                    update = local.step(choices["p1"], choices["p2"], include_state=True)
                    if update.choice_errors:
                        raise RuntimeError(f"live choice errors: {update.choice_errors}")
                    if is_action:
                        decision_index += 1

                username = str(getattr(local.views[learner_role], "player_username", ""))
                live_outcome = None
                if not truncated:
                    live_outcome = _outcome_for_player(local.winner, username)
                for state_row in game_rows:
                    state_row["live_outcome"] = live_outcome
                report["games"].append({
                    "game": game_index,
                    "winner": None if truncated else local.winner,
                    "learner_role": learner_role,
                    "turns": int(local.p1.turn),
                    "truncated": truncated,
                    "error": None,
                })
            except Exception as exc:  # noqa: BLE001 - retain partial causal evidence
                error = f"{type(exc).__name__}: {exc}"
                report["games"].append({
                    "game": game_index,
                    "winner": None,
                    "learner_role": learner_role,
                    "turns": int(local.p1.turn),
                    "truncated": False,
                    "error": error,
                })
                report["errors"].append(error)
            finally:
                finished = getattr(actor, "_battle_finished", None)
                if finished is not None:
                    finished(local.views[learner_role])
                local.close()
            report["summary"] = summarize_value_targets(report["states"])
            _atomic_json(report_path, report)
            print(
                f"[value-continuation] game={game_index + 1}/{args.games} "
                f"states={len(report['states'])} "
                f"rollouts={report['summary'].get('valid_continuations', 0)}",
                flush=True,
            )

    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = (
        "complete" if len(report["games"]) == args.games and not report["errors"] else "failed"
    )
    report["summary"] = summarize_value_targets(report["states"])
    _atomic_json(report_path, report)
    value_feature_handle.remove()
    print(f"[value-continuation] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
