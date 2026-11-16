"""Search distillation (ROADMAP P3): the network plays, EngineMCTS labels every decision.

``collect`` plays games on pinned Showdown. At each decision of a recorded side the teacher
(``fog.search.engine_mcts.ShowdownEngineMCTS``) searches from that side's own battle view,
exactly as it does when it plays. With probability ``beta`` its best action is executed,
otherwise the network's sampled action. Every decision is stored with the observation the
network received, the teacher's visit distribution and value, and the game outcome. One
compressed ``.npz`` per game; a finished game is never replayed, so the command resumes.

    python -m fog.training.search_distill collect --config configs/search_distill_cpu.yaml \
        --pool data/teams/replay_v1/train --out runs/distill/round0 --games 2400 --beta 1 \
        --workers 4

``train`` fits the policy to the visit distributions (cross-entropy) and the value to a mix of
outcome and teacher value, without belief passes:

    python -m fog.training.search_distill train --config configs/search_distill_cpu.yaml \
        --data runs/distill/round0 --out runs/distill/net0

``agreement`` summarises the optional live label check (``collect --agreement-every N``).

The teacher sees what a player sees: no true opponent team reaches a stored model input.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

N_ACTIONS = 14
# What the policy and value read when the model is called with need_belief=False.
MODEL_KEYS = (
    "entities", "entity_ids", "field", "history", "history_ids", "history_state",
    "history_len", "active_idx", "legality_mask", "action_features",
)
_HISTORY_KEYS = ("history", "history_ids", "history_state")
# Opponent of the recorded side, by game number modulo four: two mirror games (both sides
# recorded), one against each scripted bot (one side recorded).
OPPONENT_CYCLE = ("mirror", "mirror", "simpleheuristics", "maxdamage")


# ------------------------------------------------------------------------------------ schedule
def game_plan(game: int, seed: int, pool_size: int) -> dict[str, Any]:
    """Teams, opponent kind and recorded seats for one game; a function of (seed, game) only."""
    if pool_size < 2:
        raise ValueError("the team pool needs at least two teams")
    rng = random.Random(f"search-distill:{seed}:{game}")
    first, second = rng.sample(range(pool_size), 2)
    opponent = OPPONENT_CYCLE[game % len(OPPONENT_CYCLE)]
    if opponent == "mirror":
        seats = ("p1", "p2")
    else:
        seats = ("p1",) if (game // len(OPPONENT_CYCLE)) % 2 == 0 else ("p2",)
    return {
        "game": int(game), "p1_team": first, "p2_team": second, "opponent": opponent,
        "seats": seats, "battle_seed": rng.getrandbits(256), "seat_seed": rng.getrandbits(62),
    }


def game_path(out: Path, game: int) -> Path:
    return out / f"game_{game:07d}.npz"


def claim_game(out: Path, game: int) -> bool:
    """True if this process may play the game: it is unfinished and nobody else holds it.

    Mirror games take several times longer than games against a scripted bot, so a fixed
    split of game numbers leaves some processes idle. Processes instead take the next free
    game. The claim is a lock file created atomically; ``collect`` clears stale ones first.
    """
    if game_path(out, game).exists():
        return False
    try:
        os.close(os.open(out / f"game_{game:07d}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


# ------------------------------------------------------------------------------- teacher hook
class TeacherSeat:
    """One recorded side: a FogPlayer whose decisions the teacher labels and may replace."""

    def __init__(self, agent, *, beta: float, temperature: float, teacher_kwargs: dict,
                 seed: int, agreement_every: int = 0, check_budget_ms: int = 1600) -> None:
        from fog.env.player import FogPlayer
        from fog.env.showdown_local_runner import bind_fog_player
        from fog.search.engine_mcts import ShowdownEngineMCTSController
        from fog.training.rollout import RolloutBuffer

        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.rng = random.Random(seed)
        self.teacher = ShowdownEngineMCTSController(**{**teacher_kwargs, "seed": seed & 0x7FFFFFFF})
        self.agreement_every = int(agreement_every)
        # Its own stream: which decisions are checked must not change which are played.
        self._check_rng = random.Random(seed ^ 0x5EED5EED)
        self._checks: dict[str, Any] = {}
        if self.agreement_every > 0:
            self._checks = {
                "repeat": ShowdownEngineMCTSController(
                    **{**teacher_kwargs, "seed": (seed + 1) & 0x7FFFFFFF}),
                "default": ShowdownEngineMCTSController(
                    **{**teacher_kwargs, "budget_ms": int(check_budget_ms),
                       "seed": (seed + 2) & 0x7FFFFFFF}),
            }
        self.player = bind_fog_player(FogPlayer(agent, temperature=temperature))
        self.player.record_mirror = False
        self.buffer = RolloutBuffer()
        self.player.attach_buffer(self.buffer)
        self.player.action_override = self._override
        self.player.decision_observer = self._observe
        self._pending: dict[str, Any] | None = None

    @staticmethod
    def _search(controller, battle) -> tuple[int, np.ndarray | None, float | None, bool]:
        role = getattr(battle, "player_role", None) or "p1"
        tracker = controller._tracker(battle)
        action = int(controller.search.choose(battle, tracker, role))
        searched = controller.search.last_root is not None
        policy = controller.search.last_policy if searched else None
        value = controller.search.last_value if searched else None
        return action, policy, value, searched

    def _override(self, battle, obs, out) -> int:
        from fog.env.player import _sample_action

        teacher_action, policy, value, searched = self._search(self.teacher, battle)
        network_action = _sample_action(out.log_probs[0], temperature=self.temperature)
        by_teacher = self.rng.random() < self.beta
        executed = teacher_action if by_teacher else int(network_action)
        check_repeat = check_default = -1
        if searched and self._checks and self._check_rng.random() < 1.0 / self.agreement_every:
            check_repeat = self._search(self._checks["repeat"], battle)[0]
            check_default = self._search(self._checks["default"], battle)[0]
        self._pending = {
            "teacher_policy": (np.asarray(policy, dtype=np.float32) if searched
                               else np.zeros(N_ACTIONS, dtype=np.float32)),
            # Engine score in [0, 1] for the searching side, mapped to the value head's range.
            "teacher_value": np.float32(2.0 * float(value) - 1.0 if value is not None else np.nan),
            "teacher_target": np.bool_(searched),
            "teacher_action": np.int64(teacher_action),
            "executed_by_teacher": np.bool_(by_teacher),
            "network_log_probs": out.log_probs[0].detach().cpu().numpy().astype(np.float32),
            "check_repeat_action": np.int64(check_repeat),
            "check_default_action": np.int64(check_default),
        }
        return executed

    def _observe(self, _tag: str, frag: dict, _out) -> None:
        # The fragment is the row the buffer keeps; a choice Showdown rejects is discarded
        # together with these labels.
        if self._pending is None:
            raise RuntimeError("decision recorded without a teacher label")
        frag.update(self._pending)
        self._pending = None

    def take_game(self) -> tuple[list[dict], list[int], float, dict]:
        """Rows, executed actions, outcome and teacher counters of the game just finished."""
        if len(self.buffer._games) != 1:
            raise RuntimeError("expected exactly one finished game in the buffer")
        game = next(iter(self.buffer._games.values()))
        if game.outcome is None:
            raise RuntimeError("the recorded game was not finalised")
        frags, actions, outcome = list(game.frags), list(game.actions), float(game.outcome)
        self.buffer.clear()
        counters = self.teacher.take_game_counters()
        for check in self._checks.values():
            check.take_game_counters()
        return frags, actions, outcome, counters


# ------------------------------------------------------------------------------ game storage
def _pad_histories(values: list[np.ndarray], dtype) -> np.ndarray:
    """[T_i, D] -> [N, T_max, D], zero-padded at the front (most recent turn last)."""
    length = max(1, max(int(v.shape[0]) for v in values))
    out = np.zeros((len(values), length, values[0].shape[1]), dtype=dtype)
    for index, value in enumerate(values):
        if value.shape[0]:
            out[index, length - value.shape[0]:] = value
    return out


def pack_rows(frags: list[dict], actions: list[int], outcome: float, seat: int) -> dict:
    """Stack one side's decisions into arrays; histories keep their own length per row."""
    if not frags:
        raise ValueError("a recorded side made no decision")
    arrays: dict[str, np.ndarray] = {}
    for key in MODEL_KEYS:
        values = [np.asarray(frag[key]) for frag in frags]
        if key in _HISTORY_KEYS:
            dtype = np.int32 if key == "history_ids" else np.float32
            arrays[key] = _pad_histories(values, dtype)
        else:
            stacked = np.stack(values)
            if stacked.dtype == np.float64:
                stacked = stacked.astype(np.float32)
            elif stacked.dtype == np.int64 and key != "history_len":
                stacked = stacked.astype(np.int32)
            arrays[key] = stacked
    arrays["history_rows"] = np.asarray([np.asarray(f["history"]).shape[0] for f in frags],
                                        dtype=np.int32)
    for key in ("teacher_policy", "teacher_value", "teacher_target", "teacher_action",
                "executed_by_teacher", "network_log_probs", "check_repeat_action",
                "check_default_action", "turn"):
        arrays[key] = np.stack([np.asarray(frag[key]) for frag in frags])
    arrays["executed"] = np.asarray(actions, dtype=np.int64)
    arrays["outcome"] = np.full(len(frags), outcome, dtype=np.float32)
    arrays["seat"] = np.full(len(frags), seat, dtype=np.int8)
    return arrays


def merge_seats(parts: list[dict]) -> dict:
    """Concatenate the recorded sides of one game, padding histories to a common length."""
    length = max(part["history"].shape[1] for part in parts)
    out: dict[str, np.ndarray] = {}
    for key in parts[0]:
        values = []
        for part in parts:
            value = part[key]
            if key in _HISTORY_KEYS and value.shape[1] < length:
                pad = np.zeros((value.shape[0], length - value.shape[1], value.shape[2]),
                               dtype=value.dtype)
                value = np.concatenate([pad, value], axis=1)
            values.append(value)
        out[key] = np.concatenate(values)
    return out


def save_game(path: Path, arrays: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, meta=np.asarray(json.dumps(meta)), **arrays)
    temporary.replace(path)


def load_game(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "meta"}
        meta = json.loads(str(data["meta"]))
    return arrays, meta


# ---------------------------------------------------------------------------------- collection
def build_agent(cfg: dict, checkpoint: str | None, seed: int):
    import torch

    from fog.models.agent import FogAgent

    torch.manual_seed(seed)
    agent = FogAgent.from_tables(cfg).cpu()
    if checkpoint:
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    agent.eval()
    return agent


def play_game(client, agent, plan: dict, packed: list[str], *, beta: float, temperature: float,
              teacher_kwargs: dict, max_turns: int, battle_format: str = "gen9ou",
              agreement_every: int = 0, check_budget_ms: int = 1600) -> tuple[dict, dict]:
    """Play one scheduled game and return (row arrays, metadata)."""
    import torch

    from fog.env.showdown_local_runner import play_local_battle

    # The network's sampled actions depend on this stream only, so a game is reproducible
    # whichever shard plays it.
    torch.manual_seed(plan["seat_seed"] & 0x7FFFFFFF)
    seats: dict[str, TeacherSeat] = {
        role: TeacherSeat(agent, beta=beta, temperature=temperature,
                          teacher_kwargs=teacher_kwargs,
                          seed=plan["seat_seed"] + index, agreement_every=agreement_every,
                          check_budget_ms=check_budget_ms)
        for index, role in enumerate(plan["seats"])
    }
    controllers = {role: (seats[role].player if role in seats else plan["opponent"])
                   for role in ("p1", "p2")}
    started = time.perf_counter()
    result = play_local_battle(
        client,
        {
            "formatid": battle_format,
            "seed": f"sodium,{plan['battle_seed']:064x}",
            "strictChoices": True,
            "p1": {"name": "SideOne", "team": packed[plan["p1_team"]]},
            "p2": {"name": "SideTwo", "team": packed[plan["p2_team"]]},
        },
        controllers["p1"], controllers["p2"],
        seed=plan["battle_seed"] & (2**31 - 1),
        battle_tag=f"battle-distill-{plan['game']}",
        max_turns=max_turns,
    )
    parts, counters = [], {}
    for role, seat in seats.items():
        frags, actions, outcome, seat_counters = seat.take_game()
        parts.append(pack_rows(frags, actions, outcome, 0 if role == "p1" else 1))
        counters[role] = {key: value for key, value in seat_counters.items()
                          if key != "failed_states"}
    meta = {
        **{key: plan[key] for key in ("game", "p1_team", "p2_team", "opponent")},
        "seats": list(plan["seats"]), "beta": beta, "temperature": temperature,
        "teacher": teacher_kwargs, "turns": int(result.turns),
        "truncated": bool(result.truncated), "winner": result.winner,
        "seconds": time.perf_counter() - started, "teacher_counters": counters,
    }
    return merge_seats(parts), meta


def collect_shard(args) -> int:
    """Play this shard's unfinished games. Returns the number of rows written."""
    import torch

    from fog.config import load_config
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.teams import load_pool, to_packed

    torch.set_num_threads(1)
    cfg = load_config(args.config)
    packed = [to_packed(paste) for paste in load_pool(args.pool)]
    agent = build_agent(cfg, args.checkpoint, args.seed)
    teacher_kwargs = {"k": args.teacher_k, "budget_ms": args.teacher_budget_ms, "threads": 1}
    rows = played = 0
    showdown_root = cfg.get("env", {}).get("showdown_root")
    client_kwargs = {"showdown_root": showdown_root} if showdown_root else {}
    with ShowdownDirectClient(**client_kwargs) as client:
        for game in range(args.games):
            if not claim_game(args.out, game):
                continue
            plan = game_plan(game, args.seed, len(packed))
            arrays, meta = play_game(
                client, agent, plan, packed, beta=args.beta, temperature=args.temperature,
                teacher_kwargs=teacher_kwargs, max_turns=args.max_turns,
                agreement_every=args.agreement_every, check_budget_ms=args.check_budget_ms,
            )
            meta.update(seed=args.seed, checkpoint=args.checkpoint, config=args.config,
                        pool=str(args.pool))
            save_game(game_path(args.out, game), arrays, meta)
            (args.out / f"game_{game:07d}.lock").unlink(missing_ok=True)
            rows += len(arrays["executed"])
            played += 1
            if played <= 3 or played % 10 == 0:
                print(f"[distill-collect] process={args.shard} played={played} game={game} "
                      f"rows={rows} last={meta['seconds']:.1f}s", flush=True)
    return rows


def collect(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    if args.shard is not None:
        collect_shard(args)
        return
    # Claims left by an interrupted run. Do not start two parent commands on one directory.
    for stale in args.out.glob("game_*.lock"):
        stale.unlink()
    if args.workers <= 1:
        args.shards, args.shard = 1, 0
        collect_shard(args)
    else:
        base = [sys.executable, "-m", "fog.training.search_distill", *sys.argv[1:],
                "--shards", str(args.workers)]
        processes = []
        for shard in range(args.workers):
            log = (args.out / f"collect_shard_{shard}.log").open("a", encoding="utf-8")
            processes.append((shard, log, subprocess.Popen(
                [*base, "--shard", str(shard)], stdout=log, stderr=subprocess.STDOUT)))
        failed = []
        for shard, log, process in processes:
            if process.wait() != 0:
                failed.append(shard)
            log.close()
        if failed:
            raise SystemExit(f"[distill-collect] shards failed (re-run to resume): {failed}")
    summary = summarize(args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def summarize(directory: Path) -> dict:
    """Counts over the finished games of one collection directory."""
    total = {"games": 0, "rows": 0, "target_rows": 0, "teacher_executed_rows": 0,
             "truncated_games": 0, "teacher_decisions": 0, "teacher_fallbacks": 0,
             "seconds": 0.0, "checked_rows": 0, "check_repeat_agree": 0,
             "check_default_agree": 0, "network_agrees_with_teacher": 0}
    by_opponent: dict[str, int] = {}
    for path in sorted(directory.glob("game_*.npz")):
        arrays, meta = load_game(path)
        target = arrays["teacher_target"].astype(bool)
        total["games"] += 1
        total["rows"] += len(target)
        total["target_rows"] += int(target.sum())
        total["teacher_executed_rows"] += int(arrays["executed_by_teacher"].sum())
        total["truncated_games"] += int(meta["truncated"])
        total["seconds"] += float(meta["seconds"])
        total["network_agrees_with_teacher"] += int(
            (arrays["network_log_probs"].argmax(-1) == arrays["teacher_action"])[target].sum())
        for counters in meta["teacher_counters"].values():
            total["teacher_decisions"] += int(counters.get("decisions", 0))
            total["teacher_fallbacks"] += int(counters.get("fallbacks", 0))
        checked = arrays["check_default_action"] >= 0
        total["checked_rows"] += int(checked.sum())
        total["check_repeat_agree"] += int(
            (arrays["check_repeat_action"] == arrays["teacher_action"])[checked].sum())
        total["check_default_agree"] += int(
            (arrays["check_default_action"] == arrays["teacher_action"])[checked].sum())
        by_opponent[meta["opponent"]] = by_opponent.get(meta["opponent"], 0) + 1
    total["games_by_opponent"] = by_opponent
    if total["checked_rows"]:
        total["top1_agreement_with_second_seed"] = (
            total["check_repeat_agree"] / total["checked_rows"])
        total["top1_agreement_with_check_budget"] = (
            total["check_default_agree"] / total["checked_rows"])
    return total


# ------------------------------------------------------------------------------------ training
class DistillData:
    """All rows of some collection directories in memory, split by game."""

    def __init__(self, directories: list[Path], *, holdout_fraction: float, seed: int,
                 limit_games: int | None = None) -> None:
        paths = sorted(path for directory in directories
                       for path in Path(directory).glob("game_*.npz"))
        if limit_games is not None:
            paths = paths[:limit_games]
        if not paths:
            raise ValueError("no collected games found")
        self.games: list[dict] = []
        index: list[tuple[int, int]] = []
        for number, path in enumerate(paths):
            arrays, _meta = load_game(path)
            self.games.append(arrays)
            index.extend((number, row) for row in range(len(arrays["executed"])))
        self.index = np.asarray(index, dtype=np.int64)
        order = np.random.default_rng(seed).permutation(len(paths))
        held = max(1, int(round(holdout_fraction * len(paths)))) if holdout_fraction > 0 else 0
        held_games = set(order[:held].tolist())
        is_held = np.asarray([game in held_games for game, _ in index], dtype=bool)
        self.holdout_rows = np.flatnonzero(is_held)
        self.train_rows = np.flatnonzero(~is_held)
        self.n_games, self.n_holdout_games = len(paths), held

    def batch(self, rows: np.ndarray) -> tuple[dict, dict]:
        """(model inputs, targets) for some row numbers, as the live path would stack them."""
        import torch

        from fog.training.rollout import _stack_frags

        frags, picked = [], []
        for game, row in self.index[rows]:
            arrays = self.games[game]
            length = int(arrays["history_rows"][row])
            frag = {}
            for key in MODEL_KEYS:
                value = arrays[key][row]
                if key in _HISTORY_KEYS:
                    value = value[value.shape[0] - length:] if length else value[:0]
                frag[key] = value
            frags.append(frag)
            picked.append((arrays, row))
        inputs = _stack_frags(frags)
        inputs["legality_mask"] = inputs["legality_mask"].bool()

        def column(key, dtype):
            return torch.as_tensor(np.stack([arrays[key][row] for arrays, row in picked]),
                                   dtype=dtype)

        targets = {
            "policy": column("teacher_policy", torch.float32),
            "has_policy": column("teacher_target", torch.bool),
            "teacher_value": column("teacher_value", torch.float32),
            "outcome": column("outcome", torch.float32),
        }
        return inputs, targets


def distill_loss(log_probs, value, value_logit, targets: dict, *, value_mix: float = 0.5,
                 value_coef: float = 0.5):
    """Cross-entropy to the visit distribution plus value regression to outcome and search.

    ``log_probs`` is the masked policy ([B, 14], ``-inf`` on illegal actions). Rows without a
    teacher distribution contribute nothing to the policy term. The value target is
    ``(1 - value_mix) * outcome + value_mix * teacher value`` where the teacher has a value and
    the outcome alone elsewhere; with a value logit it is fitted with the logistic loss.
    """
    import torch
    import torch.nn.functional as F

    policy_target = targets["policy"]
    has_policy = targets["has_policy"]
    supported = policy_target > 0
    row_ce = -(policy_target * torch.where(supported, log_probs, torch.zeros_like(log_probs))
               ).sum(-1)
    count = has_policy.sum()
    policy = (row_ce * has_policy).sum() / count.clamp(min=1)
    teacher_value = targets["teacher_value"]
    has_value = torch.isfinite(teacher_value)
    mixed = (1.0 - value_mix) * targets["outcome"] + value_mix * torch.nan_to_num(teacher_value)
    value_target = torch.where(has_value, mixed, targets["outcome"]).clamp(-1.0, 1.0)
    if value_logit is not None:
        value_loss = F.binary_cross_entropy_with_logits(2.0 * value_logit,
                                                        (value_target + 1.0) / 2.0)
    else:
        value_loss = F.mse_loss(value, value_target)
    total = policy + value_coef * value_loss
    with torch.no_grad():
        entropy = -(policy_target * torch.where(supported, policy_target.clamp_min(1e-12).log(),
                                                torch.zeros_like(policy_target))).sum(-1)
        agree = (log_probs.argmax(-1) == policy_target.argmax(-1)) & has_policy
        metrics = {
            "loss": float(total), "policy_ce": float(policy),
            "policy_kl": float(((row_ce - entropy) * has_policy).sum() / count.clamp(min=1)),
            "teacher_top1": float(agree.sum() / count.clamp(min=1)),
            "value_loss": float(value_loss),
            "value_outcome_mse": float(F.mse_loss(value, targets["outcome"])),
            "policy_rows": float(count), "rows": float(len(log_probs)),
        }
    return total, metrics


def _weighted_mean(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    policy_rows = sum(row["policy_rows"] for row in rows)
    all_rows = sum(row["rows"] for row in rows)
    for key in rows[0]:
        if key in ("policy_rows", "rows"):
            continue
        by_policy = key in ("policy_ce", "policy_kl", "teacher_top1")
        weight_key, denominator = ("policy_rows", policy_rows) if by_policy else ("rows", all_rows)
        out[key] = sum(row[key] * row[weight_key] for row in rows) / max(1.0, denominator)
    out["policy_rows"], out["rows"] = policy_rows, all_rows
    return out


def evaluate(agent, data: DistillData, rows: np.ndarray, batch_size: int, loss_kwargs: dict) -> dict:
    import torch

    agent.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs, targets = data.batch(rows[start:start + batch_size])
            out = agent(inputs, need_belief=False)
            results.append(distill_loss(out.log_probs, out.value, out.value_logit, targets,
                                        **loss_kwargs)[1])
    return _weighted_mean(results)


def train(args) -> dict:
    import torch

    from fog.config import load_config
    from fog.models.agent import FogAgent

    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    data = DistillData(args.data, holdout_fraction=args.holdout_fraction, seed=args.seed,
                       limit_games=args.limit_games)
    torch.manual_seed(args.seed)
    agent = FogAgent.from_tables(cfg).cpu()
    if args.init:
        agent.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False))
    loss_kwargs = {"value_mix": args.value_mix,
                   "value_coef": float(cfg.get("loss", {}).get("value_coef", 0.5))}
    steps_per_epoch = math.ceil(len(data.train_rows) / args.batch_size)
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps_per_epoch * args.epochs), eta_min=args.lr_final)
    state_path = out_dir / "state.pt"
    history: list[dict] = []
    start_epoch, best = 0, float("inf")
    settings = {"data": [str(path) for path in args.data], "games": data.n_games,
                "train_rows": len(data.train_rows), "holdout_rows": len(data.holdout_rows),
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "lr_final": args.lr_final, "seed": args.seed, "value_mix": args.value_mix,
                "config": args.config, "init": args.init}
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("cannot resume: settings differ from the saved run")
        agent.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, best = state["history"], state["epoch"], state["best"]
        print(f"[distill-train] resume after epoch {start_epoch}", flush=True)
    else:
        baseline = evaluate(agent, data, data.holdout_rows, args.batch_size, loss_kwargs)
        history.append({"epoch": 0, **{f"holdout_{k}": v for k, v in baseline.items()}})
        torch.save(agent.state_dict(), out_dir / "policy_best.pt")
        best = baseline["policy_ce"]
    print(f"[distill-train] games={data.n_games} train_rows={len(data.train_rows)} "
          f"holdout_rows={len(data.holdout_rows)} steps/epoch={steps_per_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        agent.train()
        order = np.random.default_rng(args.seed * 1000 + epoch).permutation(data.train_rows)
        started, seen, running = time.perf_counter(), 0, []
        for start in range(0, len(order), args.batch_size):
            inputs, targets = data.batch(order[start:start + args.batch_size])
            out = agent(inputs, need_belief=False)
            loss, metrics = distill_loss(out.log_probs, out.value, out.value_logit, targets,
                                         **loss_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running.append(metrics)
            seen += len(targets["outcome"])
        seconds = time.perf_counter() - started
        holdout = evaluate(agent, data, data.holdout_rows, args.batch_size, loss_kwargs)
        row = {"epoch": epoch + 1, "train_rows_per_second": seen / seconds,
               "seconds": seconds,
               **{f"train_{k}": v for k, v in _weighted_mean(running).items()},
               **{f"holdout_{k}": v for k, v in holdout.items()}}
        history.append(row)
        if holdout["policy_ce"] < best:
            best = holdout["policy_ce"]
            torch.save(agent.state_dict(), out_dir / "policy_best.pt")
            row["best"] = True
        print(f"[distill-train] epoch {epoch + 1}/{args.epochs} "
              f"train_ce={row['train_policy_ce']:.4f} holdout_ce={holdout['policy_ce']:.4f} "
              f"kl={holdout['policy_kl']:.4f} top1={holdout['teacher_top1']:.4f} "
              f"value_mse={holdout['value_outcome_mse']:.4f} rows/s={row['train_rows_per_second']:.0f}",
              flush=True)
        temporary = state_path.with_suffix(".tmp")
        torch.save({"settings": settings, "model": agent.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "epoch": epoch + 1, "best": best}, temporary)
        temporary.replace(state_path)
        (out_dir / "metrics.json").write_text(
            json.dumps({"settings": settings, "history": history}, indent=2) + "\n",
            encoding="utf-8")
    torch.save(agent.state_dict(), out_dir / "policy_final.pt")
    return {"settings": settings, "history": history}


# ----------------------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    c = commands.add_parser("collect", help="play games and store teacher-labelled decisions")
    c.add_argument("--config", default="configs/search_distill_cpu.yaml")
    c.add_argument("--checkpoint", default=None,
                   help="network that plays; omit for seeded random weights (use --beta 1)")
    c.add_argument("--pool", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--games", type=int, required=True)
    c.add_argument("--beta", type=float, required=True,
                   help="probability that the teacher's action is the one executed")
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--teacher-k", type=int, default=8)
    c.add_argument("--teacher-budget-ms", type=int, default=200)
    c.add_argument("--max-turns", type=int, default=300)
    c.add_argument("--agreement-every", type=int, default=0,
                   help="on one labelled decision in N, at random, also search with a second "
                        "seed and at --check-budget-ms, and store both answers")
    c.add_argument("--check-budget-ms", type=int, default=1600)
    c.add_argument("--workers", type=int, default=1, help="processes; each takes free games")
    c.add_argument("--shards", type=int, default=1, help="internal")
    c.add_argument("--shard", type=int, default=None, help="internal: process number")

    t = commands.add_parser("train", help="fit the policy and value to stored labels")
    t.add_argument("--config", default="configs/search_distill_cpu.yaml")
    t.add_argument("--data", type=Path, nargs="+", required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--init", default=None, help="initial weights; omit for seeded random")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--lr-final", type=float, default=3e-5)
    t.add_argument("--holdout-fraction", type=float, default=0.05)
    t.add_argument("--value-mix", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=20261201)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--limit-games", type=int, default=None)

    a = commands.add_parser("agreement", help="summarise a collection directory")
    a.add_argument("--data", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "train":
        train(args)
    else:
        print(json.dumps(summarize(args.data), indent=2))


if __name__ == "__main__":
    main()
