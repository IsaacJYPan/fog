"""Matched checkpoint selection on the authoritative, in-memory Showdown simulator.

This is the policy-strength gate for checkpoints produced by the public-belief loop.  It uses
Showdown directly (no server or WebSocket) and crosses both nuisance variables that otherwise
dominate small Pokemon evaluations:

* team assignment -- each policy plays each member of a sampled team pair; and
* player position -- each policy plays as both p1 and p2 with each assignment.

One matched block therefore contains four games.  Every row receives the same team-index blocks,
Showdown seeds, and controller seeds.  Outcomes can still differ because a changed action changes
the later random-request sequence, but the exogenous schedule is identical.

Example:
    python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a \
      --checkpoint g0=runs/pb_loop/gen_00/agent_in.pt \
      --checkpoint gen0=runs/pb_loop/gen_00/agent.pt \
      --checkpoint gen1=runs/pb_loop/gen_01/agent.pt --games 256 \
      --opponent maxdamage --opponent simpleheuristics --head-to-head \
      --out runs/pb_loop/showdown_selection.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScheduledGame:
    """One game, expressed relative to candidate A and opponent B."""

    p1_is_a: bool
    p1_team: int
    p2_team: int
    showdown_seed: int
    controller_seed: int


@dataclass
class MatchResult:
    label: str
    a: str
    b: str
    requested_games: int
    completed_games: int = 0
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    truncated: int = 0
    failures: int = 0
    turns: int = 0
    wall_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        from fog.eval.stats import wilson_interval

        decisive = self.a_wins + self.b_wins
        interval = wilson_interval(self.a_wins, decisive)
        return {
            **asdict(self),
            "decisive_games": decisive,
            "a_win_rate_decisive": interval.point,
            "wilson_95_low": interval.low,
            "wilson_95_high": interval.high,
            "mean_turns": self.turns / self.completed_games if self.completed_games else None,
            "games_per_second": self.completed_games / self.wall_s if self.wall_s else None,
        }


def match_result_from_dict(data: dict[str, Any]) -> MatchResult:
    """Restore one checkpointed row while ignoring its derived reporting fields."""
    names = {field.name for field in fields(MatchResult)}
    return MatchResult(**{key: value for key, value in data.items() if key in names})


def greedy_action_with_tera_bonus(log_probs: Any, tera_logit_bonus: float = 0.0) -> int:
    """Deterministic evaluation action after an explicit Tera-family logit intervention.

    This is deliberately an evaluation-only causal screen.  Illegal actions remain ``-inf`` and
    the bonus changes only slots 4..7, so it can reveal whether a policy has useful but suppressed
    Tera preferences without adding global sampling noise or changing model parameters.
    """
    import torch

    bonus = float(tera_logit_bonus)
    if not math.isfinite(bonus) or bonus < 0:
        raise ValueError("tera logit bonus must be finite and non-negative")
    scores = log_probs
    if bonus:
        scores = log_probs.clone()
        scores[..., 4:8] = scores[..., 4:8] + bonus
    return int(torch.argmax(scores).item())


def validate_evaluation_temperature(value: float) -> float:
    """Validate the global policy temperature used by a read-only gameplay screen."""
    temperature = float(value)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("evaluation temperature must be finite and non-negative")
    return temperature


def _label_float_arg(value: str) -> tuple[str, float]:
    label, separator, raw = value.partition("=")
    if not separator or not label or not raw:
        raise argparse.ArgumentTypeError("expected LABEL=FLOAT")
    try:
        number = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=FLOAT") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Tera bonus must be finite and non-negative")
    return label, number


def matched_schedule(pool_size: int, games: int, seed: int) -> list[ScheduledGame]:
    """Create fully crossed four-game blocks for team assignment and player position."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if games < 4 or games % 4:
        raise ValueError("games must be a positive multiple of four")
    rng = random.Random(seed)
    schedule: list[ScheduledGame] = []
    for _ in range(games // 4):
        x, y = rng.randrange(pool_size), rng.randrange(pool_size)
        # A has X/p1, Y/p1, Y/p2, X/p2.  B receives the complementary assignment.
        layouts = ((True, x, y), (True, y, x), (False, x, y), (False, y, x))
        for p1_is_a, p1_team, p2_team in layouts:
            schedule.append(
                ScheduledGame(
                    p1_is_a=p1_is_a,
                    p1_team=p1_team,
                    p2_team=p2_team,
                    showdown_seed=rng.getrandbits(256),
                    controller_seed=rng.randrange(2**31),
                )
            )
    return schedule


def _options(fmt: str, packed: list[str], game: ScheduledGame) -> dict[str, Any]:
    return {
        "formatid": fmt,
        "seed": f"sodium,{game.showdown_seed:064x}",
        "strictChoices": True,
        "p1": {"name": "Player A" if game.p1_is_a else "Player B",
               "team": packed[game.p1_team]},
        "p2": {"name": "Player B" if game.p1_is_a else "Player A",
               "team": packed[game.p2_team]},
    }


def run_match(
    client: Any,
    packed: list[str],
    schedule: list[ScheduledGame],
    make_a: Callable[[], Any],
    make_b: Callable[[], Any],
    *,
    label: str,
    a_name: str,
    b_name: str,
    fmt: str = "gen9ou",
    progress_every: int = 4,
    max_turns: int = 200,
    initial: MatchResult | None = None,
    progress_callback: Callable[[MatchResult], None] | None = None,
) -> MatchResult:
    """Run one matched row; more than 20% failed games is a defect and aborts the row."""
    from fog.env.showdown_local_runner import play_local_battle

    a, b = make_a(), make_b()
    result = initial or MatchResult(
        label=label, a=a_name, b=b_name, requested_games=len(schedule)
    )
    if (result.label, result.a, result.b, result.requested_games) != (
        label, a_name, b_name, len(schedule)
    ):
        raise ValueError("saved match row does not match the requested row")
    start_index = result.completed_games + result.failures
    if start_index > len(schedule) or start_index % 4:
        raise ValueError("saved match row must end at a complete four-game block")
    previous_wall_s = result.wall_s
    started = time.perf_counter()
    for index in range(start_index, len(schedule)):
        game = schedule[index]
        p1, p2 = (a, b) if game.p1_is_a else (b, a)
        try:
            battle = play_local_battle(
                client,
                _options(fmt, packed, game),
                p1,
                p2,
                seed=game.controller_seed,
                battle_tag=f"battle-select-{label.replace(' ', '-')}-{index}",
                max_turns=max_turns,
            )
        except Exception as exc:
            result.failures += 1
            print(f"[showdown-selection] {label} game {index} failed: {exc!r}", flush=True)
            if result.failures > max(1, len(schedule) // 5):
                raise RuntimeError(
                    f"{result.failures}/{len(schedule)} games failed in {label!r}; "
                    "this is a defect, not evaluation noise"
                ) from exc
            continue
        result.completed_games += 1
        result.turns += battle.turns
        result.truncated += int(battle.truncated)
        if battle.p1_won is None:
            result.draws += 1
        elif battle.p1_won == game.p1_is_a:
            result.a_wins += 1
        else:
            result.b_wins += 1
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"[showdown-selection] {label}: {index + 1}/{len(schedule)} "
                f"A={result.a_wins} B={result.b_wins} draws={result.draws} "
                f"failures={result.failures}",
                flush=True,
            )
        result.wall_s = previous_wall_s + time.perf_counter() - started
        if progress_callback is not None and (index + 1) % 4 == 0:
            progress_callback(result)
    result.wall_s = previous_wall_s + time.perf_counter() - started
    return result


def _checkpoint_arg(value: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, path


def _load_native_agent(
    cfg: dict, checkpoint: str | Path, device: Any, *, compatible: bool = False
) -> Any:
    """Load a checkpoint under its own architecture.

    ``compatible`` is explicit because it may initialize newly added, behavior-preserving
    tensors.  It is useful for evaluating historical anchors after a zero-compatible head was
    added, while keeping strict loading as the default for current candidates.
    """
    import torch

    from fog.models.agent import FogAgent

    agent = FogAgent.from_tables(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if compatible:
        manifest = agent.load_compatible_state_dict(state)
        print(
            f"[load] compatible checkpoint {checkpoint}: "
            f"loaded={len(manifest['loaded'])} initialized={len(manifest['initialized'])}",
            flush=True,
        )
    else:
        agent.load_state_dict(state)
    return agent


def main() -> None:
    import torch

    from fog.config import load_config_with_args
    from fog.env.player import FogPlayer, _sample_action
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player
    from fog.env.teams import load_pool, to_packed
    from fog.training.value_finetune import load_posterior_agent

    parser = argparse.ArgumentParser(description="Matched direct-Showdown checkpoint selection")
    parser.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", action="append", type=_checkpoint_arg, required=True,
                        metavar="LABEL=PATH")
    parser.add_argument(
        "--checkpoint-config", action="append", type=_checkpoint_arg, default=[],
        metavar="LABEL=CONFIG",
        help="load LABEL natively with this architecture config; required for mixed architectures",
    )
    parser.add_argument(
        "--compatible-checkpoint", action="append", default=[], metavar="LABEL",
        help="explicitly use behavior-preserving compatible loading for this checkpoint label",
    )
    parser.add_argument(
        "--checkpoint-tera-logit-bonus",
        action="append",
        type=_label_float_arg,
        default=[],
        metavar="LABEL=FLOAT",
        help=(
            "evaluation-only deterministic bonus for legal Tera slots 4..7 on one checkpoint; "
            "default zero preserves the deployed policy exactly"
        ),
    )
    parser.add_argument("--opponent", action="append", default=[],
                        choices=("random", "maxdamage", "simpleheuristics"))
    parser.add_argument("--head-to-head", action="store_true")
    parser.add_argument("--games", type=int, default=256,
                        help="games per row; must be a multiple of four")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="checkpoint action temperature; zero is the deployed greedy policy",
    )
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=200,
                        help="declare longer games draws instead of allowing pathological stalls")
    parser.add_argument("--action-stats", action="store_true",
                        help="record passive selected/legal action-family frequencies")
    parser.add_argument("--resume", action="store_true",
                        help="resume --out from its last complete four-game block")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        args.temperature = validate_evaluation_temperature(args.temperature)
    except ValueError as exc:
        parser.error(str(exc))

    if len(dict(args.checkpoint)) != len(args.checkpoint):
        parser.error("checkpoint labels must be unique")
    if not args.opponent and not args.head_to_head:
        parser.error("request at least one --opponent or --head-to-head")

    cfg = load_config_with_args(args.config) if args.config else {}
    checkpoint_configs = dict(args.checkpoint_config)
    checkpoint_tera_bonuses = dict(args.checkpoint_tera_logit_bonus)
    if len(checkpoint_tera_bonuses) != len(args.checkpoint_tera_logit_bonus):
        parser.error("--checkpoint-tera-logit-bonus labels must be unique")
    compatible_labels = set(args.compatible_checkpoint)
    unknown_compatible_labels = compatible_labels - set(dict(args.checkpoint))
    if unknown_compatible_labels:
        parser.error(
            "--compatible-checkpoint labels must also appear in --checkpoint: "
            + ", ".join(sorted(unknown_compatible_labels))
        )
    unknown_config_labels = set(checkpoint_configs) - set(dict(args.checkpoint))
    if unknown_config_labels:
        parser.error(
            "--checkpoint-config labels must also appear in --checkpoint: "
            + ", ".join(sorted(unknown_config_labels))
        )
    unknown_bonus_labels = set(checkpoint_tera_bonuses) - set(dict(args.checkpoint))
    if unknown_bonus_labels:
        parser.error(
            "--checkpoint-tera-logit-bonus labels must also appear in --checkpoint: "
            + ", ".join(sorted(unknown_bonus_labels))
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teams = load_pool(args.pool)
    packed = [to_packed(team) for team in teams]
    schedule = matched_schedule(len(teams), args.games, args.seed)

    agents = {}
    for label, checkpoint in args.checkpoint:
        if label in checkpoint_configs:
            checkpoint_cfg = load_config_with_args(checkpoint_configs[label])
            agent = _load_native_agent(
                checkpoint_cfg, checkpoint, device, compatible=label in compatible_labels
            )
        else:
            agent = load_posterior_agent(cfg, checkpoint, device)
        agent.eval()
        agents[label] = agent

    action_stats: dict[str, dict[str, Any]] = {}

    def checkpoint_factory(label: str, stats_key: str | None = None) -> Callable[[], Any]:
        def make() -> Any:
            tera_bonus = checkpoint_tera_bonuses.get(label, 0.0)
            player = FogPlayer(
                agents[label],
                temperature=args.temperature,
                tera_logit_bonus=tera_bonus,
            )
            if args.action_stats and stats_key is not None:
                stats = action_stats.setdefault(
                    stats_key,
                    {
                        "decisions": 0,
                        "selected_by_slot": [0] * 14,
                        "legal_opportunities_by_slot": [0] * 14,
                        "selected_by_family": {
                            "ordinary_move": 0,
                            "tera_move": 0,
                            "switch": 0,
                            "forced": 0,
                        },
                        "legal_decisions_by_family": {
                            "ordinary_move": 0,
                            "tera_move": 0,
                            "switch": 0,
                            "forced": 0,
                        },
                    },
                )

                def select_and_count(_battle: Any, obs: Any, out: Any) -> int:

                    action = _sample_action(
                        out.log_probs[0],
                        temperature=args.temperature,
                        tera_logit_bonus=tera_bonus,
                    )
                    legal = [bool(value) for value in obs.legality_mask]
                    stats["decisions"] += 1
                    stats["selected_by_slot"][action] += 1
                    for index, is_legal in enumerate(legal):
                        stats["legal_opportunities_by_slot"][index] += int(is_legal)
                    families = {
                        "ordinary_move": range(4),
                        "tera_move": range(4, 8),
                        "switch": range(8, 13),
                        "forced": range(13, 14),
                    }
                    for family, indices in families.items():
                        stats["legal_decisions_by_family"][family] += int(
                            any(legal[index] for index in indices)
                        )
                        stats["selected_by_family"][family] += int(action in indices)
                    return action

                player.action_override = select_and_count
            return bind_fog_player(player)

        return make

    def scripted_factory(name: str) -> Callable[[], str]:
        return lambda: name

    protocol = {
        "simulator": "pinned Pokemon Showdown, direct in-memory worker",
        "format": args.format,
        "pool": str(args.pool),
        "pool_size": len(teams),
        "games_per_row": args.games,
        "block_size": 4,
        "block_design": "both team assignments crossed with both player positions",
        "seed": args.seed,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "device": str(device),
        "checkpoint_configs": checkpoint_configs,
        "checkpoint_tera_logit_bonuses": checkpoint_tera_bonuses,
        "draws_excluded_from_win_rate": True,
    }
    checkpoint_map = dict(args.checkpoint)
    rows: list[dict[str, Any]] = []
    current: MatchResult | None = None
    if args.resume and args.out.exists():
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        if saved.get("protocol") != protocol or saved.get("checkpoints") != checkpoint_map:
            raise ValueError(
                "resume report protocol/checkpoints differ from this command; use the original "
                "arguments or a new --out path"
            )
        rows = list(saved.get("rows", []))
        if saved.get("current_row") is not None:
            current = match_result_from_dict(saved["current_row"])

    def write_report(status: str, active: MatchResult | None = None) -> None:
        report = {
            "status": status,
            "protocol": protocol,
            "checkpoints": checkpoint_map,
            "rows": rows,
        }
        if args.action_stats:
            report["action_stats"] = action_stats
        if active is not None:
            report["current_row"] = active.as_dict()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    completed = {
        row["label"] for row in rows
        if int(row.get("completed_games", 0)) + int(row.get("failures", 0)) >= args.games
    }

    def finish(match: MatchResult) -> None:
        nonlocal current
        rows[:] = [row for row in rows if row.get("label") != match.label]
        rows.append(match.as_dict())
        completed.add(match.label)
        current = None
        write_report("in_progress")
        row = rows[-1]
        print(
            f"[showdown-selection] RESULT {match.label}: "
            f"{row['a_win_rate_decisive']:.3f} "
            f"[{row['wilson_95_low']:.3f}, {row['wilson_95_high']:.3f}] "
            f"({match.a_wins}-{match.b_wins}, draws={match.draws})",
            flush=True,
        )

    def run_row(client: Any, make_a: Callable[[], Any], make_b: Callable[[], Any], *,
                label: str, a_name: str, b_name: str) -> None:
        nonlocal current
        if label in completed:
            print(f"[showdown-selection] resume: skipping completed row {label}", flush=True)
            return
        initial = current if current is not None and current.label == label else None
        if current is not None and initial is None:
            raise ValueError(
                f"saved current row is {current.label!r}, but schedule expects {label!r}; "
                "the resume command changed row order"
            )

        def progress(match: MatchResult) -> None:
            nonlocal current
            current = match
            write_report("in_progress", match)

        match = run_match(
            client, packed, schedule, make_a, make_b,
            label=label, a_name=a_name, b_name=b_name,
            fmt=args.format, progress_every=args.progress_every,
            max_turns=args.max_turns, initial=initial, progress_callback=progress,
        )
        finish(match)

    client_kwargs = {}
    showdown_root = cfg.get("env", {}).get("showdown_root")
    if showdown_root:
        client_kwargs["showdown_root"] = showdown_root
    with ShowdownDirectClient(**client_kwargs) as client:
        for label in agents:
            for opponent in args.opponent:
                run_row(
                    client,
                    checkpoint_factory(label, f"{label} vs {opponent}"),
                    scripted_factory(opponent),
                    label=f"{label} vs {opponent}", a_name=label, b_name=opponent,
                )

        if args.head_to_head:
            labels = list(agents)
            for i, a_label in enumerate(labels):
                for b_label in labels[i + 1:]:
                    run_row(
                        client,
                        checkpoint_factory(a_label, f"{a_label} vs {b_label}:a"),
                        checkpoint_factory(b_label, f"{a_label} vs {b_label}:b"),
                        label=f"{a_label} vs {b_label}", a_name=a_label, b_name=b_label,
                    )

    write_report("complete")
    print(f"[showdown-selection] report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
