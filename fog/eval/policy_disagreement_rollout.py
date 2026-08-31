"""Offline outcome arbitration for policy-versus-heuristic disagreement states.

The accepted policy plays ordinary validation games from its censored observation.  When its
action differs from SimpleHeuristics on the *same* player-visible view, this diagnostic forks the
authoritative simulator state offline, forces each candidate action, and finishes both branches
under the same Simple-vs-Simple continuation and paired RNG seeds.  Authoritative state is never
fed to the acting policy and no counterfactual action is executed in the live game.

This is an evaluation diagnostic, not a teacher or training-data generator.  It asks whether the
heuristic's disagreed-with action has better long-horizon outcome evidence than the policy action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def _outcome_for_player(winner: str | None, player_name: str) -> float:
    """Map Showdown display names to poke-env's normalized usernames safely."""
    if winner is None:
        return 0.0

    def normalized(value: str) -> str:
        return "".join(character for character in str(value).casefold() if character.isalnum())

    winner_id = normalized(winner)
    player_id = normalized(player_name)
    if not winner_id or not player_id:
        raise ValueError(f"cannot resolve winner {winner!r} for player {player_name!r}")
    return 1.0 if winner_id == player_id else -1.0


def summarize_pairs(states: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(pair["policy_outcome"] - pair["simple_outcome"])
              for state in states for pair in state["pairs"]
              if pair.get("policy_outcome") is not None
              and pair.get("simple_outcome") is not None]
    policy_better = sum(delta > 0 for delta in deltas)
    simple_better = sum(delta < 0 for delta in deltas)
    tied = sum(delta == 0 for delta in deltas)
    return {
        "paired_rollouts": len(deltas),
        "policy_action_better": policy_better,
        "simple_action_better": simple_better,
        "tied": tied,
        "mean_policy_minus_simple_outcome": sum(deltas) / len(deltas) if deltas else None,
        "policy_better_rate_non_tied": (
            policy_better / (policy_better + simple_better)
            if policy_better + simple_better else None
        ),
    }


def select_simple_outcome_label(
    executed: int,
    simple_target: int,
    pairs: list[dict[str, Any]],
    *,
    min_simple_better: int = 2,
) -> bool:
    """Conservatively accept a heuristic move only when paired outcomes support it.

    Cross-family corrections are deliberately excluded: the causal audits found the reproducible
    weakness inside ordinary-move ranking, while switching was neutral or favored the parent.
    """
    if not (0 <= int(executed) < 4 and 0 <= int(simple_target) < 4):
        return False
    deltas = [
        float(pair["policy_outcome"] - pair["simple_outcome"])
        for pair in pairs
        if pair.get("policy_outcome") is not None and pair.get("simple_outcome") is not None
    ]
    simple_better = sum(delta < 0 for delta in deltas)
    policy_better = sum(delta > 0 for delta in deltas)
    return bool(
        len(deltas) >= min_simple_better
        and simple_better >= min_simple_better
        and policy_better == 0
        and sum(deltas) / len(deltas) <= -0.5
    )


def _branch_views(local: Any) -> dict[str, Any]:
    return {
        role: deepcopy(local.views[role], {id(local): None})
        for role in ("p1", "p2")
    }


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
    requests: dict[str, Any],
    learner_role: str,
    root_learner_choice: str,
    root_opponent_choice: str | None,
    seed: int,
    max_turns: int,
) -> dict[str, Any]:
    from fog.env.showdown_local_runner import _choice

    branch_views = {role: deepcopy(view) for role, view in views.items()}
    learner_index = 0 if learner_role == "p1" else 1
    choices = [root_opponent_choice, root_opponent_choice]
    choices[learner_index] = root_learner_choice
    choices[1 - learner_index] = root_opponent_choice
    sampled = client.sample(
        state,
        tuple(_normalize_choice(choice) for choice in choices),
        seed=_sample_seed(seed, 0),
        include_state=True,
    )
    _apply_sample(branch_views, sampled)
    rng = random.Random(seed)
    steps = 1
    while not sampled["ended"] and int(sampled["turn"]) < int(max_turns):
        next_choices = [
            _choice("simpleheuristics", branch_views[role], sampled["requests"].get(role), rng)
            for role in ("p1", "p2")
        ]
        if next_choices == [None, None]:
            raise RuntimeError("live sampled branch has no pending choice")
        sampled = client.sample(
            sampled["state"],
            tuple(_normalize_choice(choice) for choice in next_choices),
            seed=_sample_seed(seed, steps),
            include_state=True,
        )
        _apply_sample(branch_views, sampled)
        steps += 1
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
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.showdown_local_runner import _choice, bind_fog_player
    from fog.env.teams import load_pool, to_packed
    from fog.eval.showdown_direct_selection import _options, matched_schedule
    from fog.models.agent import FogAgent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--max-states-per-game", type=int, default=0)
    parser.add_argument("--disagreement-stride", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261321)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    from fog.env.player import FogPlayer

    actor = bind_fog_player(FogPlayer(agent, temperature=0.0))
    teams = load_pool(args.pool)
    packed = [to_packed(team) for team in teams]
    schedule = matched_schedule(len(teams), args.games, args.seed)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "pool": str(args.pool),
        "games_target": args.games,
        "max_states": args.max_states,
        "max_states_per_game": args.max_states_per_game,
        "disagreement_stride": args.disagreement_stride,
        "repeats": args.repeats,
        "seed": args.seed,
        "trained": False,
        "validation_used_for_training": False,
        "truth_used_for_actor": False,
        "authoritative_state_used_offline_only": True,
        "continuation_policy": "SimpleHeuristics for both sides",
        "games": [],
        "states": [],
        "errors": [],
    }
    _atomic_json(report_path, report)
    client_kwargs = {}
    showdown_root = cfg.get("env", {}).get("showdown_root")
    if showdown_root:
        client_kwargs["showdown_root"] = showdown_root

    with ShowdownDirectClient(**client_kwargs) as client:
        for game_index, game in enumerate(schedule):
            learner_role = "p1" if game.p1_is_a else "p2"
            local = LocalShowdownBattle.create(
                client,
                _options("gen9ou", packed, game),
                battle_tag=f"policy-disagreement-rollout-{game_index}",
            )
            actual_rng = random.Random(game.controller_seed)
            try:
                truncated = False
                decision_index = 0
                disagreement_index = 0
                game_state_count = 0
                while not local.ended:
                    if int(local.p1.turn) >= args.max_turns:
                        truncated = True
                        break
                    controllers = {
                        learner_role: actor,
                        "p2" if learner_role == "p1" else "p1": "simpleheuristics",
                    }
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
                    if is_action and len(report["states"]) < args.max_states:
                        label_rng = random.Random(args.seed * 1_000_003 + game_index * 997 + decision_index)
                        simple_choice = _choice(
                            "simpleheuristics", local.views[learner_role], learner_request, label_rng
                        )
                        policy_choice = choices[learner_role]
                        if _normalize_choice(simple_choice) != _normalize_choice(policy_choice):
                            disagreement_index += 1
                            within_game_cap = (
                                args.max_states_per_game <= 0
                                or game_state_count < args.max_states_per_game
                            )
                            selected_stride = (
                                (disagreement_index - 1) % max(1, args.disagreement_stride) == 0
                            )
                            if not within_game_cap or not selected_stride:
                                decision_index += 1
                                update = local.step(
                                    choices["p1"], choices["p2"], include_state=True
                                )
                                if update.choice_errors:
                                    raise RuntimeError(f"live choice errors: {update.choice_errors}")
                                continue
                            state_row = {
                                "game": game_index,
                                "turn": int(local.p1.turn),
                                "role": learner_role,
                                "policy_choice": _normalize_choice(policy_choice),
                                "simple_choice": _normalize_choice(simple_choice),
                                "opponent_choice": _normalize_choice(
                                    choices["p2" if learner_role == "p1" else "p1"]
                                ),
                                "pairs": [],
                            }
                            root_state = local.state
                            root_views = _branch_views(local)
                            opponent_choice = choices["p2" if learner_role == "p1" else "p1"]
                            for repeat in range(args.repeats):
                                pair_seed = args.seed + game_index * 100_003 + decision_index * 101 + repeat
                                pair: dict[str, Any] = {"seed": pair_seed}
                                try:
                                    policy_branch = _rollout(
                                        client, state=root_state, views=root_views,
                                        requests=local.update.requests, learner_role=learner_role,
                                        root_learner_choice=policy_choice,
                                        root_opponent_choice=opponent_choice, seed=pair_seed,
                                        max_turns=args.max_turns,
                                    )
                                    simple_branch = _rollout(
                                        client, state=root_state, views=root_views,
                                        requests=local.update.requests, learner_role=learner_role,
                                        root_learner_choice=simple_choice,
                                        root_opponent_choice=opponent_choice, seed=pair_seed,
                                        max_turns=args.max_turns,
                                    )
                                    pair.update({
                                        "policy_outcome": policy_branch["outcome"],
                                        "simple_outcome": simple_branch["outcome"],
                                        "policy_branch": policy_branch,
                                        "simple_branch": simple_branch,
                                        "error": None,
                                    })
                                except Exception as exc:  # noqa: BLE001 - bounded diagnostic
                                    pair.update({
                                        "policy_outcome": None,
                                        "simple_outcome": None,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    })
                                state_row["pairs"].append(pair)
                            report["states"].append(state_row)
                            game_state_count += 1
                            report["summary"] = summarize_pairs(report["states"])
                            _atomic_json(report_path, report)
                    update = local.step(choices["p1"], choices["p2"], include_state=True)
                    if update.choice_errors:
                        raise RuntimeError(f"live choice errors: {update.choice_errors}")
                    decision_index += 1
                report["games"].append({
                    "game": game_index,
                    "winner": None if truncated else local.winner,
                    "learner_role": learner_role,
                    "turns": int(local.p1.turn),
                    "truncated": truncated,
                    "error": None,
                })
            except Exception as exc:  # noqa: BLE001 - retain partial causal evidence
                report["games"].append({
                    "game": game_index,
                    "winner": None,
                    "learner_role": learner_role,
                    "turns": int(local.p1.turn),
                    "truncated": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                report["errors"].append(report["games"][-1]["error"])
            finally:
                finished = getattr(actor, "_battle_finished", None)
                if finished is not None:
                    finished(local.views[learner_role])
                local.close()
            report["summary"] = summarize_pairs(report["states"])
            _atomic_json(report_path, report)
            print(
                f"[counterfactual] game={game_index + 1}/{args.games} "
                f"states={len(report['states'])} pairs={report['summary']['paired_rollouts']}",
                flush=True,
            )

    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete" if len(report["games"]) == args.games and not report["errors"] else "failed"
    report["summary"] = summarize_pairs(report["states"])
    _atomic_json(report_path, report)
    print(f"[counterfactual] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
