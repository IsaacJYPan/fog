"""Matched checkpoint selection on the authoritative, in-memory Showdown simulator.

This is the policy-strength gate for checkpoints produced by the public-belief loop.  It uses
Showdown directly (no server or WebSocket) and crosses both nuisance variables that otherwise
dominate small Pokemon evaluations:

* team assignment -- each policy plays each member of a sampled team pair; and
* player position -- each policy plays as both p1 and p2 with each assignment.

One matched block therefore contains four games.  Every row receives the same team-index blocks,
Showdown seeds, and controller seeds.  Outcomes can still differ because a changed action changes
the later random-request sequence, but the exogenous schedule is identical.

Any controller can be an entrant: ``--controller LABEL=module:factory`` imports ``factory`` and
calls it (with ``--controller-kwargs LABEL=JSON`` as keyword arguments) once per row. The returned
object either has the local runner's ``act(battle) -> order`` contract or a ``choose(battle)``
method returning a FOG action index or an order. Optional hooks: ``take_game_counters()`` (per-game
counters, recorded after every game), ``discard_last_uncommitted_decision(battle)`` (Showdown
rejected the last choice) and ``_battle_finished(battle)``. A controller label named in
``--opponent`` is an opponent; any other controller is a candidate like a checkpoint.

``--sequential`` stops each row at the first ``BlockSPRT`` decision on per-block mean scores.
Each finished row prints one Markdown results-ledger row.

Example:
    python -m fog.eval.showdown_direct_selection --pool data/teams/pool_b \
      --controller mcts=fog.search.engine_mcts:showdown_controller \
      --controller-kwargs 'mcts={"k": 4, "budget_ms": 400}' \
      --opponent simpleheuristics --games 100 --sequential --out runs/mcts_vs_sh.json

    python -m fog.eval.showdown_direct_selection --pool data/teams/pool_a \
      --checkpoint g0=runs/pb_loop/gen_00/agent_in.pt \
      --checkpoint gen0=runs/pb_loop/gen_00/agent.pt \
      --checkpoint gen1=runs/pb_loop/gen_01/agent.pt --games 256 \
      --opponent maxdamage --opponent simpleheuristics --head-to-head \
      --out runs/pb_loop/showdown_selection.json
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
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
    game_scores: list[float | None] = field(default_factory=list)
    # Per-game controller counters ({"a": {...}, "b": {...}}) for controllers that expose them.
    game_counters: list[dict[str, Any]] = field(default_factory=list)
    sequential: dict[str, Any] | None = None

    def counter_totals(self, side: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        for game in self.game_counters:
            for key, value in (game.get(side) or {}).items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
        return totals

    def block_means(self) -> list[float]:
        """Mean score of each complete four-game block (unresolved games excluded)."""
        means = []
        for start in range(0, len(self.game_scores) - len(self.game_scores) % 4, 4):
            scores = [s for s in self.game_scores[start:start + 4] if s is not None]
            if scores:
                means.append(sum(scores) / len(scores))
        return means

    def as_dict(self) -> dict[str, Any]:
        from fog.eval.stats import wilson_interval

        decisive = self.a_wins + self.b_wins
        interval = wilson_interval(self.a_wins, decisive)
        finished = self.completed_games - self.truncated
        true_draws = self.draws - self.truncated
        return {
            **asdict(self),
            "decisive_games": decisive,
            "a_win_rate_decisive": interval.point,
            "wilson_95_low": interval.low,
            "wilson_95_high": interval.high,
            "finished_games": finished,
            "true_draws": true_draws,
            "a_score": (self.a_wins + 0.5 * true_draws) / finished if finished else None,
            "mean_turns": self.turns / self.completed_games if self.completed_games else None,
            "games_per_second": self.completed_games / self.wall_s if self.wall_s else None,
            "a_counter_totals": self.counter_totals("a"),
            "b_counter_totals": self.counter_totals("b"),
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
    from fog.env.player import _sample_action

    return _sample_action(log_probs, temperature=0, tera_logit_bonus=tera_logit_bonus)


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
    stop_after_block: Callable[[MatchResult], bool] | None = None,
) -> MatchResult:
    """Run one matched row; more than 20% failed games is a defect and aborts the row.

    ``stop_after_block`` is called after every complete four-game block; returning ``True`` ends
    the row early (sequential stopping).
    """
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
    if len(result.game_scores) != start_index:
        raise ValueError("legacy evaluation lacks per-game outcomes; use a new report path")
    previous_wall_s = result.wall_s
    started = time.perf_counter()

    def take_counters() -> None:
        counters = {}
        for side, controller in (("a", a), ("b", b)):
            take = getattr(controller, "take_game_counters", None)
            if take is not None:
                counters[side] = take()
        if counters:
            result.game_counters.append(counters)

    def stop_now(index: int) -> bool:
        return (stop_after_block is not None and (index + 1) % 4 == 0
                and bool(stop_after_block(result)))

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
            result.game_scores.append(None)
            take_counters()
            print(f"[showdown-selection] {label} game {index} failed: {exc!r}", flush=True)
            if result.failures > max(1, len(schedule) // 5):
                raise RuntimeError(
                    f"{result.failures}/{len(schedule)} games failed in {label!r}; "
                    "this is a defect, not evaluation noise"
                ) from exc
            result.wall_s = previous_wall_s + time.perf_counter() - started
            if progress_callback is not None and (index + 1) % 4 == 0:
                progress_callback(result)
            if stop_now(index):
                break
            continue
        take_counters()
        result.completed_games += 1
        result.turns += battle.turns
        result.truncated += int(battle.truncated)
        if battle.truncated:
            result.draws += 1  # legacy aggregate; true_draws explicitly excludes these
            result.game_scores.append(None)
        elif battle.p1_won is None:
            result.draws += 1
            result.game_scores.append(0.5)
        elif battle.p1_won == game.p1_is_a:
            result.a_wins += 1
            result.game_scores.append(1.0)
        else:
            result.b_wins += 1
            result.game_scores.append(0.0)
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"[showdown-selection] {label}: {index + 1}/{len(schedule)} "
                f"A={result.a_wins} B={result.b_wins} draws={result.draws} "
                f"failures={result.failures}",
                flush=True,
            )
        result.wall_s = previous_wall_s + time.perf_counter() - started
        stop = stop_now(index)
        if progress_callback is not None and (index + 1) % 4 == 0:
            progress_callback(result)
        if stop:
            break
    result.wall_s = previous_wall_s + time.perf_counter() - started
    return result


class ChooseAdapter:
    """Give a ``choose(battle)`` controller the local runner's ``act(battle)`` contract.

    ``choose`` may return a FOG action index (mapped through ``map_action_to_order``) or an order.
    Optional hooks of the wrapped object are forwarded.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def act(self, battle: Any) -> Any:
        choice = self.inner.choose(battle)
        if isinstance(choice, (int,)) and not isinstance(choice, bool):
            from fog.env.actions import map_action_to_order
            from fog.env.showdown_local_runner import LocalOrderFactory

            return map_action_to_order(battle, int(choice), LocalOrderFactory())
        return choice

    def __getattr__(self, name: str) -> Any:
        if name in ("take_game_counters", "discard_last_uncommitted_decision",
                    "_battle_finished", "choose_teampreview"):
            return getattr(self.inner, name)
        raise AttributeError(name)


def load_controller_factory(spec: str, kwargs: dict[str, Any] | None = None) -> Callable[[], Any]:
    """``module:factory`` -> a zero-argument maker of local-runner controllers."""
    module_name, separator, attr = spec.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError(f"controller spec must be module:factory, got {spec!r}")
    factory = getattr(importlib.import_module(module_name), attr)
    options = dict(kwargs or {})

    def make() -> Any:
        controller = factory(**options)
        if not hasattr(controller, "act") and hasattr(controller, "choose"):
            controller = ChooseAdapter(controller)
        if not (hasattr(controller, "act") or callable(controller)):
            raise TypeError(f"{spec} returned neither an act() nor a choose() controller")
        return controller

    return make


def sequential_stop(min_blocks: int, max_blocks: int, **sprt: Any) -> Callable[[MatchResult], bool]:
    """A ``run_match`` stop rule: ``BlockSPRT`` over the row's block means, recomputed each block
    (so it resumes correctly). Stores the decision in ``result.sequential``."""
    from dataclasses import asdict as dc_asdict

    from fog.eval.stats import BlockSPRT

    min_blocks = max(2, min(min_blocks, max_blocks))

    def stop(result: MatchResult) -> bool:
        decision = BlockSPRT.evaluate(
            result.block_means(), min_blocks=min_blocks, max_blocks=max_blocks, **sprt
        )
        result.sequential = {**dc_asdict(decision), "min_blocks": min_blocks,
                             "max_blocks": max_blocks, **sprt}
        return decision.decision != "continue"

    return stop


def git_short_hash() -> str:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True, cwd=Path(__file__).resolve().parent)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parent).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ledger_row(row: dict[str, Any], *, backend: str, commit: str, command: str,
               date: str | None = None) -> str:
    """One results-ledger Markdown row for a finished match row (candidate = A)."""
    totals = row.get("a_counter_totals") or {}
    decisions = totals.get("decisions") or 0
    seconds = totals.get("decision_seconds")
    per_decision = f"{seconds / decisions:.3f}" if decisions and seconds is not None else "n/a"
    fallbacks = (f"{int(totals['fallbacks'])}/{int(decisions)}"
                 if "fallbacks" in totals and decisions else "n/a")
    point = row.get("a_win_rate_decisive")
    if point is None or point != point:
        rate = "n/a"
    else:
        rate = f"{point:.3f} [{row['wilson_95_low']:.3f}, {row['wilson_95_high']:.3f}]"
    games = f"{row['completed_games']}"
    if row.get("failures"):
        games += f" (+{row['failures']} failed)"
    sequential = row.get("sequential")
    if sequential:
        games += f", SPRT {sequential['decision']} at {sequential['blocks']} blocks"
    cells = [date or datetime.date.today().isoformat(), row["a"], row["b"], backend, games,
             f"{rate} ({row['a_wins']}-{row['b_wins']}-{row['draws'] - row['truncated']})",
             per_decision, fallbacks, f"`{commit}`", f"`{command}`"]
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


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
    parser.add_argument("--checkpoint", action="append", type=_checkpoint_arg, default=[],
                        metavar="LABEL=PATH")
    parser.add_argument(
        "--controller", action="append", type=_checkpoint_arg, default=[],
        metavar="LABEL=MODULE:FACTORY",
        help="an arbitrary controller entrant; a label also named in --opponent is an opponent",
    )
    parser.add_argument(
        "--controller-kwargs", action="append", type=_checkpoint_arg, default=[],
        metavar="LABEL=JSON", help="keyword arguments (a JSON object) for a controller factory",
    )
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
                        help="random, maxdamage, simpleheuristics, or a --controller label")
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
    parser.add_argument("--policy-diagnostics", action="store_true",
                        help="save raw Tera margins and temperature-one mass with action stats")
    parser.add_argument("--resume", action="store_true",
                        help="resume --out from its last complete four-game block")
    parser.add_argument("--sequential", action="store_true",
                        help="stop each row at the first BlockSPRT decision on block means")
    parser.add_argument("--sprt-p0", type=float, default=0.50)
    parser.add_argument("--sprt-p1", type=float, default=0.55)
    parser.add_argument("--sprt-alpha", type=float, default=0.05)
    parser.add_argument("--sprt-beta", type=float, default=0.05)
    parser.add_argument("--sprt-min-blocks", type=int, default=100,
                        help="capped at the row's block count (--games / 4), the maximum")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import shlex
    import sys

    command = "python -m fog.eval.showdown_direct_selection " + shlex.join(sys.argv[1:])

    try:
        args.temperature = validate_evaluation_temperature(args.temperature)
    except ValueError as exc:
        parser.error(str(exc))

    if len(dict(args.checkpoint)) != len(args.checkpoint):
        parser.error("checkpoint labels must be unique")
    controller_specs = dict(args.controller)
    if len(controller_specs) != len(args.controller) or set(controller_specs) & set(
        dict(args.checkpoint)
    ):
        parser.error("controller labels must be unique and distinct from checkpoint labels")
    controller_kwargs: dict[str, dict[str, Any]] = {}
    for label, raw in args.controller_kwargs:
        if label not in controller_specs:
            parser.error(f"--controller-kwargs label {label!r} is not a --controller label")
        try:
            controller_kwargs[label] = json.loads(raw)
        except json.JSONDecodeError as exc:
            parser.error(f"--controller-kwargs {label}: {exc}")
        if not isinstance(controller_kwargs[label], dict):
            parser.error(f"--controller-kwargs {label} must be a JSON object")
    scripted_names = ("random", "maxdamage", "simpleheuristics")
    for opponent in args.opponent:
        if opponent not in scripted_names and opponent not in controller_specs:
            parser.error(f"--opponent {opponent!r} is neither scripted nor a --controller label")
    candidate_controllers = [label for label in controller_specs if label not in args.opponent]
    if not args.checkpoint and not candidate_controllers:
        parser.error("request at least one --checkpoint or candidate --controller")
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
    controller_makers = {
        label: load_controller_factory(spec, controller_kwargs.get(label))
        for label, spec in controller_specs.items()
    }
    stop_rule = None
    if args.sequential:
        stop_rule = sequential_stop(
            args.sprt_min_blocks, args.games // 4, p0=args.sprt_p0, p1=args.sprt_p1,
            alpha=args.sprt_alpha, beta=args.sprt_beta,
        )

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
                from fog.eval.action_diagnostics import new_action_stats, record_action

                stats = action_stats.setdefault(stats_key, new_action_stats())

                def select_and_count(battle: Any, obs: Any, out: Any) -> int:
                    action = _sample_action(
                        out.log_probs[0], temperature=args.temperature,
                        tera_logit_bonus=tera_bonus,
                    )
                    record_action(stats, battle, obs.legality_mask, out.log_probs[0], action,
                                  policy_diagnostics=args.policy_diagnostics)
                    return action

                player.action_override = select_and_count
            return bind_fog_player(player)

        return make

    def scripted_factory(name: str) -> Callable[[], Any]:
        if name in controller_makers:
            return controller_makers[name]
        return lambda: name

    def candidate_factory(label: str, stats_key: str) -> Callable[[], Any]:
        if label in controller_makers:
            return controller_makers[label]
        return checkpoint_factory(label, stats_key)

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
        "score_schema_version": 2,
        "action_stats_schema_version": 2,
        "action_stats": args.action_stats,
        "policy_diagnostics": args.policy_diagnostics,
        "primary_score": "win=1, genuine draw=0.5, loss=0; truncations unresolved",
        "draws_excluded_from_win_rate": True,  # legacy decisive diagnostic only
    }
    if controller_specs:
        protocol["controllers"] = controller_specs
        protocol["controller_kwargs"] = controller_kwargs
    if args.sequential:
        protocol["sequential"] = {
            "test": "BlockSPRT on four-game block mean scores", "p0": args.sprt_p0,
            "p1": args.sprt_p1, "alpha": args.sprt_alpha, "beta": args.sprt_beta,
            "min_blocks": min(args.sprt_min_blocks, args.games // 4),
            "max_blocks": args.games // 4,
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
        action_stats.update(saved.get("action_stats", {}))
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
        or (row.get("sequential") or {}).get("decision", "continue") != "continue"
    }
    ledger_rows: list[str] = []
    commit = git_short_hash()

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
        if match.sequential:
            print(f"[showdown-selection] SPRT {match.label}: {match.sequential}", flush=True)
        ledger_rows.append(ledger_row(row, backend="showdown_direct", commit=commit,
                                      command=command))

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
            stop_after_block=stop_rule,
        )
        finish(match)

    client_kwargs = {}
    showdown_root = cfg.get("env", {}).get("showdown_root")
    if showdown_root:
        client_kwargs["showdown_root"] = showdown_root
    with ShowdownDirectClient(**client_kwargs) as client:
        for label in [*agents, *candidate_controllers]:
            for opponent in args.opponent:
                run_row(
                    client,
                    candidate_factory(label, f"{label} vs {opponent}"),
                    scripted_factory(opponent),
                    label=f"{label} vs {opponent}", a_name=label, b_name=opponent,
                )

        if args.head_to_head:
            labels = [*agents, *candidate_controllers]
            for i, a_label in enumerate(labels):
                for b_label in labels[i + 1:]:
                    run_row(
                        client,
                        candidate_factory(a_label, f"{a_label} vs {b_label}:a"),
                        candidate_factory(b_label, f"{a_label} vs {b_label}:b"),
                        label=f"{a_label} vs {b_label}", a_name=a_label, b_name=b_label,
                    )

    write_report("complete")
    print(f"[showdown-selection] report -> {args.out}", flush=True)
    if ledger_rows:
        print("\n".join(ledger_rows), flush=True)


if __name__ == "__main__":
    main()
