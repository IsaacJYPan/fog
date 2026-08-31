"""Small, truth-audited direct-Showdown trace for root determinized search.

Ground truth is used only after belief worlds have been sampled and scored.  It is never inserted
into the agent batch, simulator, action scores, or chosen action.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fog.constants.schema import ACTIONS, ActionType
from fog.encoding.reveals import FAMILY_COLS


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def world_accuracy(
    worlds: np.ndarray, truth: np.ndarray, revealed: np.ndarray
) -> dict[str, Any]:
    """Compare samples to truth without making truth available to search."""
    worlds = np.asarray(worlds, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    revealed = np.asarray(revealed, dtype=np.int64)
    valid = truth >= 0
    hidden = valid & (revealed < 0)
    shown = valid & (revealed >= 0)

    def sample_score(sample: np.ndarray, mask: np.ndarray) -> float | None:
        count = int(mask.sum())
        return float((sample[mask] == truth[mask]).mean()) if count else None

    hidden_scores = [sample_score(sample, hidden) for sample in worlds]
    numeric = [score for score in hidden_scores if score is not None]
    family: dict[str, Any] = {}
    for name, cols in FAMILY_COLS.items():
        mask = np.zeros_like(hidden)
        mask[:, list(cols)] = hidden[:, list(cols)]
        scores = [sample_score(sample, mask) for sample in worlds]
        present = [score for score in scores if score is not None]
        family[name] = {
            "hidden_fields": int(mask.sum()),
            "per_sample": scores,
            "mean": float(np.mean(present)) if present else None,
            "variance": float(np.var(present)) if present else None,
        }

    exact = [
        bool(np.array_equal(sample[hidden], truth[hidden])) if hidden.any() else True
        for sample in worlds
    ]
    keys = [sample.tobytes() for sample in worlds]
    return {
        "hidden_fields": int(hidden.sum()),
        "revealed_or_clamped_fields": int(shown.sum()),
        "per_sample_hidden_correctness": hidden_scores,
        "hidden_correctness_mean": float(np.mean(numeric)) if numeric else None,
        "hidden_correctness_variance": float(np.var(numeric)) if numeric else None,
        "hidden_correctness_min": float(np.min(numeric)) if numeric else None,
        "hidden_correctness_max": float(np.max(numeric)) if numeric else None,
        "exact_true_hidden_world_per_sample": exact,
        "exact_true_hidden_world_sampled": any(exact),
        "revealed_correctness_per_sample": [sample_score(sample, shown) for sample in worlds],
        "unique_samples": len(set(keys)),
        "duplicate_samples": len(keys) - len(set(keys)),
        "per_family": family,
    }


def _decoded_world(world: np.ndarray) -> list[dict[str, Any]]:
    from fog.constants.tables import TABLES
    from fog.env.engine_battle import _decode_world_slot

    decoded = []
    for row in world:
        value = _decode_world_slot(row, 100, TABLES)
        decoded.append(
            {
                "species": value.species,
                "item": value.item,
                "ability": value.ability,
                "moves": list(value.moves),
                "nature": value.nature,
                "evs": list(value.evs),
                "tera": value.tera_type,
            }
        )
    return decoded


def _action_label(battle: Any, action: int) -> str:
    from fog.env.actions import active_move_slots, move_key, pokemon_key, team_order_minus_active

    kind, slot = ACTIONS[action]
    if kind is ActionType.FORCED:
        return "forced/default"
    if kind in (ActionType.MOVE, ActionType.MOVE_TERA):
        moves = active_move_slots(battle)
        move = move_key(moves[slot]) if slot < len(moves) else f"slot-{slot + 1}"
        return f"{'tera+' if kind is ActionType.MOVE_TERA else ''}{move}"
    bench = team_order_minus_active(battle)
    target = pokemon_key(bench[slot]) if slot < len(bench) else f"slot-{slot + 1}"
    return f"switch:{target}"


class DiagnosticStepper:
    """Showdown reconstruction stepper that exposes the reply and leaf mechanics."""

    def __init__(self, battle: Any, tracker: Any, role: str, seed: int) -> None:
        from fog.search.engine_step import ShowdownWorldStepper

        self.base = ShowdownWorldStepper(battle, tracker, role=role, seed=seed)
        self.role = role
        self.opp_role = "p2" if role == "p1" else "p1"
        self.failures = 0
        self._bases = self.base._bases
        self.last: dict[str, Any] = {}

    def __call__(self, action_idx: int, world_tokens: np.ndarray):
        from fog.encoding.observation import Observation

        try:
            fork = self.base._fork(np.asarray(world_tokens, dtype=np.int64))
            pending = fork.pending_sides()
            if self.role not in pending:
                raise RuntimeError(f"searcher not pending: {pending}")
            my_order = fork.engine_action(self.role, action_idx)
            moves = {self.role: my_order}
            reply_idx = reply_order = None
            if self.opp_role in pending:
                reply_idx = int(self.base.opponent_policy.choose(fork, self.opp_role))
                reply_order = fork.engine_action(self.opp_role, reply_idx)
                moves[self.opp_role] = reply_order
            fork.step(moves)
            forced_replies: list[dict[str, Any]] = []
            guard = 0
            while not fork.finished and self.role not in fork.pending_sides():
                guard += 1
                if guard > self.base.max_forced_steps:
                    raise RuntimeError("forced-replacement loop exceeded limit")
                idx = int(self.base.opponent_policy.choose(fork, self.opp_role))
                order = fork.engine_action(self.opp_role, idx)
                forced_replies.append({"action_idx": idx, "order": order})
                fork.step({self.opp_role: order})
            if fork.finished:
                result: Any = (
                    1.0 if fork.winner == self.role else (-1.0 if fork.winner else 0.0)
                )
                result_kind = "terminal"
            else:
                result = Observation.from_battle(
                    fork.view(self.role), tracker=fork.trackers[self.role]
                )
                result_kind = "fog_leaf"
            self.last = {
                "searcher_order": my_order,
                "opponent_reply_idx": reply_idx,
                "opponent_reply": reply_order,
                "forced_replies": forced_replies,
                "result_kind": result_kind,
                "error": None,
            }
            return result
        except Exception as exc:  # noqa: BLE001 - trace object records exact failure details
            self.failures += 1
            self.last = {
                "result_kind": "failure",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return None


@torch.no_grad()
def traced_action(
    agent: Any,
    batch: dict[str, Any],
    *,
    battle: Any,
    tracker: Any,
    truth: np.ndarray,
    k: int,
    prior_weight: float,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    from fog.search.determinized import _stack_observations

    started = time.perf_counter()
    agent.eval()
    legal = batch["legality_mask"][0].nonzero(as_tuple=True)[0].tolist()
    out = agent(batch)
    log_probs = out.log_probs[0]
    root_value = float(out.value[0])
    policy_action = int(log_probs[legal].argmax()) if legal else 13
    if legal:
        policy_action = legal[policy_action]

    generator = torch.Generator(device=batch["legality_mask"].device)
    generator.manual_seed(seed)
    from fog.search.determinized import root_belief_context

    contexts = agent.encode_contexts(batch)
    belief_context = root_belief_context(agent, batch, contexts)
    worlds = agent.belief.sample_team(
        belief_context, batch["opp_revealed"], k=k, generator=generator
    )[0].cpu().numpy()
    revealed = batch["opp_revealed"]["tokens"][0].detach().cpu().numpy()
    accuracy = world_accuracy(worlds, truth, revealed)
    role = getattr(battle, "player_role", None) or "p1"
    stepper = DiagnosticStepper(battle, tracker, role, seed)

    values_by_action: dict[int, list[float]] = {action: [] for action in legal}
    simulations: list[dict[str, Any]] = []
    leaves: list[Any] = []
    leaf_refs: list[int] = []
    for action in legal:
        for world_index, world in enumerate(worlds):
            result = stepper(action, world)
            simulation = {
                "action_idx": action,
                "action": _action_label(battle, action),
                "world_index": world_index,
                **stepper.last,
                "leaf_value": None,
            }
            simulations.append(simulation)
            ref = len(simulations) - 1
            if result is None:
                continue
            if isinstance(result, (int, float)):
                value = float(result)
                values_by_action[action].append(value)
                simulations[ref]["leaf_value"] = value
            else:
                leaves.append(result)
                leaf_refs.append(ref)

    device = batch["legality_mask"].device
    if leaves:
        leaf_values = agent(_stack_observations(leaves, device)).value.tolist()
        for ref, value in zip(leaf_refs, leaf_values):
            value = float(value)
            simulations[ref]["leaf_value"] = value
            values_by_action[simulations[ref]["action_idx"]].append(value)

    action_rows = []
    best_action = policy_action
    best_score = -math.inf
    for action in legal:
        values = values_by_action[action]
        mean = float(np.mean(values)) if values else root_value
        std = float(np.std(values)) if values else None
        prior = prior_weight * float(log_probs[action])
        score = mean + prior
        if score > best_score:
            best_score, best_action = score, action
        action_rows.append(
            {
                "action_idx": action,
                "action": _action_label(battle, action),
                "policy_probability": float(log_probs[action].exp()),
                "policy_log_probability": float(log_probs[action]),
                "search_mean": mean,
                "search_std": std,
                "valid_worlds": len(values),
                "failed_worlds": k - len(values),
                "prior_contribution": prior,
                "blended_score": score,
            }
        )
    ranked = sorted(action_rows, key=lambda row: row["blended_score"], reverse=True)
    margin = (
        ranked[0]["blended_score"] - ranked[1]["blended_score"]
        if len(ranked) > 1
        else None
    )
    trace = {
        "turn": int(getattr(battle, "turn", 0) or 0),
        "role": role,
        "root_value": root_value,
        "legal_actions": [_action_label(battle, action) for action in legal],
        "policy_action_idx": policy_action,
        "policy_action": _action_label(battle, policy_action),
        "search_action_idx": best_action,
        "search_action": _action_label(battle, best_action),
        "search_overrode_policy": best_action != policy_action,
        "score_margin": margin,
        "actions": action_rows,
        "worlds": [
            {
                "index": index,
                "decoded": _decoded_world(world),
                "hidden_correctness": accuracy["per_sample_hidden_correctness"][index],
                "exact_true_hidden_world": accuracy["exact_true_hidden_world_per_sample"][index],
            }
            for index, world in enumerate(worlds)
        ],
        "belief_accuracy": accuracy,
        "simulations": simulations,
        "simulation_attempts": len(legal) * k,
        "simulation_failures": stepper.failures,
        "elapsed_seconds": time.perf_counter() - started,
        "truth_used_for_action": False,
    }
    return best_action, trace


class TraceFogPlayer:
    """Composition wrapper around FogPlayer with a traced search override."""

    def __init__(self, agent: Any, trace_path: Path, *, k: int, prior_weight: float, seed: int):
        from fog.env.player import FogPlayer

        class _Player(FogPlayer):
            pass

        self.adapter = _Player(agent, temperature=0.0)
        self.adapter.search = {"k": k, "prior_weight": prior_weight, "seed": seed}
        self.adapter._search_action = self._search_action  # type: ignore[method-assign]
        self.trace_path = trace_path
        self.k = k
        self.prior_weight = prior_weight
        self.seed = seed
        self.truth: np.ndarray | None = None
        self.game: dict[str, Any] = {}
        self.pending: dict[str, Any] | None = None

    def _search_action(self, battle, obs, tracker, batch, device):
        if self.truth is None:
            raise RuntimeError("trace truth was not configured for this game")
        action, trace = traced_action(
            self.adapter.agent,
            batch,
            battle=battle,
            tracker=tracker,
            truth=self.truth,
            k=self.k,
            prior_weight=self.prior_weight,
            seed=self.seed + len(self.game.get("decisions", [])),
        )
        trace["game"] = {k: v for k, v in self.game.items() if k != "decisions"}
        self.pending = trace
        return action

    def flush(self, events: list[str], choice: str | None) -> None:
        if self.pending is None:
            return
        self.pending["authoritative_choice"] = choice
        self.pending["ensuing_authoritative_events"] = list(events)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(self.pending)) + "\n")
        self.game.setdefault("decisions", []).append(self.pending)
        self.pending = None

    def discard(self) -> None:
        self.pending = None


def _play_traced_game(client, options, player: TraceFogPlayer, opponent, *, seed, max_turns):
    import random

    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.showdown_local_runner import _choice

    local = LocalShowdownBattle.create(client, options, battle_tag=player.game["tag"])
    rng = random.Random(seed)
    learner_role = "p1" if player.game["p1_is_learner"] else "p2"
    controllers = (
        (player.adapter, opponent)
        if learner_role == "p1"
        else (opponent, player.adapter)
    )
    try:
        truncated = False
        for _ in range(1_000):
            if local.ended:
                break
            if int(local.p1.turn) >= max_turns:
                truncated = True
                break
            views = (local.p1, local.p2)
            choices = [
                _choice(controllers[0], views[0], local.update.requests["p1"], rng),
                _choice(controllers[1], views[1], local.update.requests["p2"], rng),
            ]
            for _retry in range(4):
                update = local.step(choices[0], choices[1], include_state=False)
                if not update.choice_errors:
                    player.flush(update.messages[learner_role], choices[0 if learner_role == "p1" else 1])
                    break
                if learner_role in update.choice_errors:
                    player.discard()
                for index, role in enumerate(("p1", "p2")):
                    if role in update.choice_errors:
                        choices[index] = _choice(
                            controllers[index], views[index], update.requests[role], rng
                        )
            else:
                raise RuntimeError("four unavailable-choice retries")
        player.flush([], None)
        return {
            "winner": None if truncated else local.winner,
            "p1_won": None if truncated else local.p1.won,
            "turns": int(local.p1.turn),
            "truncated": truncated,
        }
    finally:
        for controller, view in zip(controllers, (local.p1, local.p2)):
            finished = getattr(controller, "_battle_finished", None)
            if finished is not None:
                finished(view)
        local.close()


def main() -> None:
    from fog.config import load_config_with_args
    from fog.constants.tables import TABLES
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import load_pool, to_packed
    from fog.eval.showdown_direct_selection import _options, matched_schedule
    from fog.models.agent import FogAgent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games-per-opponent", type=int, default=2)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    trace_path = args.out / "trace.jsonl"
    results_path = args.out / "results.json"
    if results_path.exists() or trace_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing trace in {args.out}")
    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    teams = load_pool(args.pool)
    packed = [to_packed(team) for team in teams]
    truths = [truth_tokens_from_paste(team, TABLES) for team in teams]
    schedule_size = max(4, 4 * math.ceil(args.games_per_opponent / 4))
    schedule = matched_schedule(len(teams), schedule_size, args.seed)
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "pool": str(args.pool),
        "k": args.k,
        "prior_weight": args.prior_weight,
        "seed": args.seed,
        "games": [],
        "complete": False,
    }
    _atomic_json(results_path, report)
    player = TraceFogPlayer(
        agent, trace_path, k=args.k, prior_weight=args.prior_weight, seed=args.seed
    )
    bind_fog_player(player.adapter)

    client_kwargs = {}
    showdown_root = cfg.get("env", {}).get("showdown_root")
    if showdown_root:
        client_kwargs["showdown_root"] = showdown_root
    with ShowdownDirectClient(**client_kwargs) as client:
        game_number = 0
        for opponent in ("simpleheuristics", "maxdamage"):
            for game in schedule[: args.games_per_opponent]:
                if time.monotonic() >= deadline:
                    break
                learner_team = game.p1_team if game.p1_is_a else game.p2_team
                opponent_team = game.p2_team if game.p1_is_a else game.p1_team
                tag = f"search-trace-{opponent}-{game_number}"
                player.truth = truths[opponent_team]
                player.adapter.private_state_truth_provider = (
                    lambda _battle, value=truths[learner_team]: value
                )
                player.game = {
                    "tag": tag,
                    "opponent": opponent,
                    "p1_is_learner": game.p1_is_a,
                    "learner_team_index": learner_team,
                    "opponent_team_index": opponent_team,
                    "decisions": [],
                }
                error = None
                try:
                    outcome = _play_traced_game(
                        client,
                        _options("gen9ou", packed, game),
                        player,
                        opponent,
                        seed=game.controller_seed,
                        max_turns=args.max_turns,
                    )
                except Exception as exc:  # noqa: BLE001 - game report retains exact failure
                    outcome = None
                    error = f"{type(exc).__name__}: {exc}"
                game_report = {
                    **{k: v for k, v in player.game.items() if k != "decisions"},
                    "decisions": len(player.game["decisions"]),
                    "outcome": outcome,
                    "error": error,
                }
                report["games"].append(game_report)
                _atomic_json(results_path, report)
                print(
                    f"[search-trace] {opponent} game {game_number}: "
                    f"decisions={game_report['decisions']} error={error}",
                    flush=True,
                )
                game_number += 1
            if time.monotonic() >= deadline:
                break

    report["elapsed_seconds"] = time.monotonic() - started
    report["complete"] = len(report["games"]) == 2 * args.games_per_opponent and all(
        game["error"] is None for game in report["games"]
    )
    _atomic_json(results_path, report)


if __name__ == "__main__":
    main()
