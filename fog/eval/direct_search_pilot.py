"""Bounded, progress-checkpointed direct-Showdown search-on/search-off pilot.

This diagnostic uses the in-memory authoritative Showdown worker, not poke-env's websocket
harness. Every policy/search pair receives the same four-way-crossed team/side schedule. Results
are atomically rewritten after each four-game block so a hard deadline leaves useful evidence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def wilson95(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    point = wins / games
    denominator = 1.0 + z * z / games
    centre = (point + z * z / (2.0 * games)) / denominator
    half = (
        z
        * (point * (1.0 - point) / games + z * z / (4.0 * games * games)) ** 0.5
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def row_payload(
    label: str,
    opponent: str,
    search: bool,
    wins: int,
    games: int,
    elapsed: float,
    adapter: Any | None,
    *,
    losses: int = 0,
    draws: int = 0,
    truncations: int = 0,
    failures: int = 0,
    turns: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    decisive = wins + losses
    low, high = wilson95(wins, decisive)
    return {
        "label": label,
        "opponent": opponent,
        "search": search,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "decisive_games": decisive,
        "decisive_win_rate": wins / decisive if decisive else None,
        "wilson95": [low, high],
        "truncations": truncations,
        "failures": failures,
        "turns": turns,
        "mean_turns": turns / games if games else None,
        "elapsed_seconds": elapsed,
        "search_stats": dict(getattr(adapter, "search_stats", {})) if adapter else {},
        "error": error,
    }


def run_row(
    *,
    client: Any,
    agent: Any,
    opponent: str,
    packed: list[str],
    schedule: list[Any],
    search_cfg: dict | None,
    report: dict[str, Any],
    results_path: Path,
    deadline: float,
    max_turns: int,
) -> dict[str, Any]:
    from fog.env.player import FogPlayer
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.eval.showdown_direct_selection import _options

    adapter = FogPlayer(agent, temperature=0.0)
    adapter.search = dict(search_cfg) if search_cfg else None
    learner = bind_fog_player(adapter)
    started = time.monotonic()
    wins = losses = draws = truncations = failures = turns = 0
    error = None
    label = f"{'search' if search_cfg else 'policy'} vs {opponent}"
    for index, game in enumerate(schedule):
        if time.monotonic() >= deadline:
            break
        p1, p2 = (learner, opponent) if game.p1_is_a else (opponent, learner)
        try:
            battle = play_local_battle(
                client,
                _options("gen9ou", packed, game),
                p1,
                p2,
                seed=game.controller_seed,
                battle_tag=f"search-pilot-{label.replace(' ', '-')}-{index}",
                max_turns=max_turns,
            )
        except Exception as exc:  # noqa: BLE001 - row records and bounds every failed battle
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            if failures > max(1, len(schedule) // 5):
                break
            continue
        turns += battle.turns
        truncations += int(battle.truncated)
        if battle.p1_won is None:
            draws += 1
        elif battle.p1_won == game.p1_is_a:
            wins += 1
        else:
            losses += 1
        completed = wins + losses + draws
        if (index + 1) % 4 == 0:
            progress = row_payload(
                label,
                opponent,
                search_cfg is not None,
                wins,
                completed,
                time.monotonic() - started,
                adapter,
                losses=losses,
                draws=draws,
                truncations=truncations,
                failures=failures,
                turns=turns,
                error=error,
            )
            report["current"] = progress
            atomic_json(results_path, report)
            print(
                f"[search-pilot] {label}: {wins}-{losses}, draws={draws}, "
                f"fallbacks={progress['search_stats'].get('fallbacks', 0)}",
                flush=True,
            )
    return row_payload(
        label,
        opponent,
        search_cfg is not None,
        wins,
        wins + losses + draws,
        time.monotonic() - started,
        adapter,
        losses=losses,
        draws=draws,
        truncations=truncations,
        failures=failures,
        turns=turns,
        error=error,
    )


def main() -> None:
    import torch

    from fog.config import load_config_with_args
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.teams import load_pool, to_packed
    from fog.eval.showdown_direct_selection import matched_schedule
    from fog.models.agent import FogAgent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--search-mode", choices=("belief_sampled", "direct_two_sided"),
                        default="direct_two_sided")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--opponent-model", choices=("maxdamage", "policy"), default="maxdamage")
    parser.add_argument("--min-override-margin", type=float, default=0.0)
    parser.add_argument("--uncertainty-z", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.games < 4 or args.games % 4:
        parser.error("--games must be a positive multiple of four")

    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    teams = load_pool(args.pool)
    packed = [to_packed(team) for team in teams]
    schedule = matched_schedule(len(teams), args.games, args.seed)
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    if results_path.exists():
        raise FileExistsError(f"refusing to overwrite prior pilot results: {results_path}")
    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "pool": str(args.pool),
        "teams": len(teams),
        "seed": args.seed,
        "games_target_per_row": args.games,
        "block_size": 4,
        "search": {
            "mode": args.search_mode,
            "k": args.k,
            "prior_weight": args.prior_weight,
            "opponent_model": args.opponent_model,
            "min_override_margin": args.min_override_margin,
            "uncertainty_z": args.uncertainty_z,
        },
        "deadline_seconds": args.deadline_seconds,
        "max_turns": args.max_turns,
        "rows": [],
        "current": None,
        "complete": False,
    }
    atomic_json(results_path, report)

    client_kwargs = {}
    showdown_root = cfg.get("env", {}).get("showdown_root")
    if showdown_root:
        client_kwargs["showdown_root"] = showdown_root
    with ShowdownDirectClient(**client_kwargs) as client:
        # Establish both cheap policy baselines, then spend the bounded budget on search.
        search_cfg = {
            "k": args.k,
            "iters": 256,
            "floor": 0.02,
            "prior_weight": args.prior_weight,
            "seed": args.seed,
            "opponent_model": args.opponent_model,
            "min_override_margin": args.min_override_margin,
            "uncertainty_z": args.uncertainty_z,
        }
        if args.search_mode == "direct_two_sided":
            search_cfg["mode"] = "direct_two_sided"
        for opponent, row_search_cfg in (
            ("simpleheuristics", None),
            ("maxdamage", None),
            ("simpleheuristics", search_cfg),
            ("maxdamage", search_cfg),
        ):
            if time.monotonic() >= deadline:
                break
            row = run_row(
                client=client,
                agent=agent,
                opponent=opponent,
                packed=packed,
                schedule=schedule,
                search_cfg=row_search_cfg,
                report=report,
                results_path=results_path,
                deadline=deadline,
                max_turns=args.max_turns,
            )
            report["rows"].append(row)
            report["current"] = None
            atomic_json(results_path, report)

    report["elapsed_seconds"] = time.monotonic() - started
    report["complete"] = len(report["rows"]) == 4 and all(
        row["games"] == args.games and row["error"] is None for row in report["rows"]
    )
    atomic_json(results_path, report)
    print(f"[search-pilot] report -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
