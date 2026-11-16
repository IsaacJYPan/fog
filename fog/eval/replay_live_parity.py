"""Compare saved model inputs in live self-play with public replay reconstruction.

CPU only. Each completed game is checkpointed; --compare-only reuses those games after a fix.
True own sheets are supplied exclusively here, never by the training conversion command.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def model_fields(batch: dict) -> dict[str, np.ndarray]:
    """Copy the actual batch passed to the model, stripping only the batch dimension."""
    fields = {key: value[0].detach().cpu().numpy().copy()
              for key, value in batch.items() if hasattr(value, "detach")}
    fields["opp_tokens"] = batch["opp_revealed"]["tokens"][0].cpu().numpy().copy()
    for key, value in batch["opp_revealed"]["masks"].items():
        fields[f"opp_mask.{key}"] = value[0].cpu().numpy().copy()
    return fields


def comparison_fields(fields: dict) -> dict:
    """Name every entity float block separately for each side; retain all other inputs."""
    from fog.encoding.entities import ENTITY_RAW_DIM, L

    from fog.encoding.dims import DIMS
    from fog.encoding.history import pad_or_truncate_front

    result = {key: np.asarray(value) for key, value in fields.items() if key != "entities"}
    # Front padding is masked by history_len in the model. Compare records in a common window,
    # preserving a short converter window's lost records rather than hiding them.
    for key in ("history", "history_ids", "history_state"):
        if key in result:
            result[key] = pad_or_truncate_front(result[key], cap=DIMS.history_cap)
    if "action_features" in result:
        from fog.encoding.action_features import ACTION_FEATURE_NAMES
        from fog.models.mechanical_features import MECHANICAL_ACTION_FEATURE_NAMES

        columns = [ACTION_FEATURE_NAMES.index(name) for name in MECHANICAL_ACTION_FEATURE_NAMES]
        result["action_features.mechanical"] = result["action_features"][:, columns]
    layout = list(vars(L).items())
    for index, (name, start) in enumerate(layout):
        end = layout[index + 1][1] if index + 1 < len(layout) else ENTITY_RAW_DIM
        for side, rows in (("own", slice(0, 6)), ("opponent", slice(6, 12))):
            result[f"entities.{side}.{name}"] = fields["entities"][rows, start:end]
    return result


def keyed_rows(rows: list[dict]) -> dict:
    """Align by side, turn and decision ordinal, never by label or approximate features."""
    counts: Counter = Counter()
    result = {}
    for row in rows:
        prefix = (row["role"], int(row["turn"]),
                  int(row["fields"].get("decision_phase", 0)))
        key = (*prefix, counts[prefix])
        counts[prefix] += 1
        result[key] = row
    return result


def live_history_diagnostics(rows: list[dict]) -> dict:
    """Count the two history facts independently of replay reconstruction."""
    from fog.encoding.dims import DIMS

    decisions = defaultdict(list)
    for row in rows:
        decisions[(row["role"], row["turn"])].append(int(row["fields"]["label"]))
    replaced = {key: actions for key, actions in decisions.items() if len(actions) > 1}
    affected = 0
    for row in rows:
        length = int(row["fields"]["history_len"])
        start = int(row["turn"]) - length
        for (role, turn), actions in replaced.items():
            if role != row["role"] or not start <= turn < row["turn"]:
                continue
            hist = row["fields"]["history"]
            action = hist[-length + turn - start,
                          DIMS.hist_snapshot:DIMS.hist_snapshot + DIMS.hist_my_action]
            if actions[0] != actions[-1] and action[actions[-1]] == 1:
                affected += 1
                break
    forced = sum(int(row["fields"].get("decision_phase", 0)) == 1 for row in rows)
    return {"forced_decisions": forced, "turns_with_multiple_decisions": len(replaced),
            "rows_with_overwritten_first_action": affected,
            "current_turn_excluded_at_replacement": forced}


def explain_difference(name, a, b, key, left, right) -> str | None:
    """Explain a residual only when its coordinates/values support the stated cause."""
    if name == "opp_mask.item" and (a is None or b is None):
        present = b if a is None else a
        if np.asarray(present).all():
            return "all-true item mask is explicit live and implicit in the training loader"
    if a is None or b is None or a.shape != b.shape:
        return None
    delta = np.abs(a.astype(float) - b.astype(float))
    if name == "entities.own.hp" and float(delta.max(initial=0)) <= 0.010001:
        return "public HP percentages versus exact own HP in the private request (difference <= 0.01)"
    if name == "action_features":
        from fog.encoding.action_features import ACTION_FEATURE_NAMES
        from fog.models.mechanical_features import MECHANICAL_ACTION_FEATURE_NAMES

        allowed = [ACTION_FEATURE_NAMES.index(n) for n in MECHANICAL_ACTION_FEATURE_NAMES]
        if np.array_equal(a[:, allowed], b[:, allowed]):
            return "all differences are outside the eleven candidate columns read by R4"
    if name == "history":
        from fog.encoding.dims import DIMS
        from fog.encoding.events import _OFF_RES, _R_DMG_DEALT, _R_DMG_TAKEN, _R_EXACT

        remaining = delta.copy()
        # The live per-turn snapshot and own damage use private HP; replay uses percentages.
        for column, limit in ((2, 0.010001), (_OFF_RES + _R_DMG_DEALT, 0.020001),
                              (_OFF_RES + _R_DMG_TAKEN, 0.020001),
                              (_OFF_RES + _R_EXACT, 2 / 255 + 1e-6)):
            remaining[remaining[:, column] <= limit, column] = 0
        quantized = not remaining.any()
        if quantized:
            return "history HP/damage columns differ only within public-HP quantization bounds"
        first = DIMS.hist_snapshot
        stop = first + DIMS.hist_my_action
        unobserved = False
        length = int(left[key]["fields"]["history_len"])
        for index in np.flatnonzero(remaining[:, first:stop].any(axis=1)):
            turn = key[1] - DIMS.history_cap + int(index)
            prior = (key[0], turn, 0, 0)
            if prior in left and prior not in right and turn >= key[1] - length:
                remaining[index, first:stop] = 0
                unobserved = True
        if not remaining.any() and unobserved:
            return "public-HP quantization plus a prior normal choice never disclosed in the log"
    return None


def compare_rows(live: list[dict], replay: list[dict]) -> dict:
    left, right = keyed_rows(live), keyed_rows(replay)
    aligned = sorted(left.keys() & right.keys())
    stats = defaultdict(lambda: {"rows": 0, "exact": 0, "max_difference": 0.0})
    differences = []
    for key in aligned:
        first = comparison_fields(left[key]["fields"])
        second = comparison_fields(right[key]["fields"])
        mismatch = {}
        for name in sorted(first.keys() | second.keys()):
            a, b = first.get(name), second.get(name)
            exact = a is not None and b is not None and np.array_equal(a, b)
            delta = (float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
                     if a is not None and b is not None and a.shape == b.shape else None)
            entry = stats[name]
            entry["rows"] += 1
            entry["exact"] += int(exact)
            if delta is not None:
                entry["max_difference"] = max(entry["max_difference"], delta)
            if not exact:
                explanation = explain_difference(name, a, b, key, left, right)
                mismatch[name] = {"max_difference": delta,
                                  "live_shape": None if a is None else list(a.shape),
                                  "replay_shape": None if b is None else list(b.shape),
                                  "explanation": explanation}
                if a is not None and b is not None and a.shape == b.shape:
                    coordinates = np.argwhere(a != b)[:12]
                    mismatch[name]["examples"] = [
                        {"index": coordinate.tolist(),
                         "live": np.asarray(a[tuple(coordinate)]).item(),
                         "replay": np.asarray(b[tuple(coordinate)]).item()}
                        for coordinate in coordinates]
        if mismatch:
            differences.append({"decision": list(key), "fields": mismatch})
    for entry in stats.values():
        entry["exact_share"] = entry["exact"] / entry["rows"]
    return {"aligned": len(aligned), "live_decisions": len(left), "replay_rows": len(right),
            "missing_replay": [list(k) for k in sorted(left.keys() - right.keys())],
            "missing_live": [list(k) for k in sorted(right.keys() - left.keys())],
            "fields": dict(stats), "differences": differences}


class RecordingSeat:
    """Use the production player and intercept the exact batch at its forward boundary."""

    def __init__(self, agent):
        from fog.env.player import FogPlayer
        from fog.env.showdown_local_runner import bind_fog_player

        self.player = bind_fog_player(FogPlayer(agent, temperature=1.0))
        self.player.record_mirror = False
        self.rows = []
        self.pending = None
        self.hook = agent.register_forward_pre_hook(self.capture)
        self.player.action_override = self.choose

    def capture(self, _module, args):
        self.pending = model_fields(args[0])

    def choose(self, battle, _obs, out):
        from fog.env.player import _sample_action

        action = int(_sample_action(out.log_probs[0], temperature=1.0))
        self.rows.append({"role": battle.player_role, "turn": int(battle.turn),
                          "fields": {**self.pending, "label": np.asarray(action)}})
        return action


def collect_game(client, agent, pastes, game: int, seed: int, max_turns: int) -> tuple:
    import torch

    from fog.constants.tables import TABLES
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.showdown_local_runner import _choice
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import to_packed

    rng = random.Random(f"replay-parity:{seed}:{game}")
    torch.manual_seed(rng.getrandbits(31))
    teams = rng.sample(pastes, 2)
    seats = [RecordingSeat(agent), RecordingSeat(agent)]
    options = {"formatid": "gen9ou", "seed": f"sodium,{rng.getrandbits(256):064x}",
               "strictChoices": True,
               "p1": {"name": "SideOne", "team": to_packed(teams[0])},
               "p2": {"name": "SideTwo", "team": to_packed(teams[1])}}
    log = []
    try:
        with LocalShowdownBattle.create(client, options, battle_tag=f"battle-parity-{game}") as local:
            log.extend(local.update.public_messages)
            while not local.ended and local.p1.turn < max_turns:
                choices = [_choice(seat.player, local.views[role], local.update.requests[role], rng)
                           for seat, role in zip(seats, ("p1", "p2"))]
                if choices == [None, None]:
                    raise RuntimeError("neither side owes a choice")
                update = local.step(*choices, include_state=False)
                if update.choice_errors:
                    raise RuntimeError(f"rejected choice: {update.choice_errors}")
                log.extend(update.public_messages)
            ended = local.ended
        rows = [row for seat in seats for row in seat.rows]
    finally:
        for seat in seats:
            seat.hook.remove()
    sheets = {role: truth_tokens_from_paste(paste, TABLES).tolist()
              for role, paste in zip(("p1", "p2"), teams)}
    return rows, {"log": "\n".join(log), "players": ["SideOne", "SideTwo"],
                  "sheets": sheets, "ended": ended}


def save_game(path: Path, rows: list, replay: dict) -> None:
    """One atomic per-game checkpoint; local NPZ never contains Python pickles."""
    arrays = {f"r{i}.{key}": value for i, row in enumerate(rows)
              for key, value in row["fields"].items()}
    meta = {"replay": replay, "rows": [{"role": r["role"], "turn": r["turn"]} for r in rows]}
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, meta=np.asarray(json.dumps(meta)), **arrays)
    path.with_suffix(".tmp").replace(path)


def load_game(path: Path) -> tuple:
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["meta"]))
        rows = [{**row, "fields": {key.split(".", 1)[1]: arrays[key].copy()
                                  for key in arrays.files if key.startswith(f"r{i}.")}}
                for i, row in enumerate(meta["rows"])]
    return rows, meta["replay"]


def replay_rows(replay: dict) -> list:
    from fog.constants.tables import TABLES
    from fog.models.shared_latent import SharedBattleLatent
    from fog.scripts.parse_replays import parse_replay, rows_to_arrays
    from fog.training.dataset import _build_batch

    result = parse_replay(replay, private_sets={role: np.asarray(sheet)
                                               for role, sheet in replay["sheets"].items()})
    rows = [row for row in result["rows"] if row["policy_valid"]]
    if not rows:
        return []
    arrays = rows_to_arrays(rows)
    out = []
    for i, row in enumerate(rows):
        batch = _build_batch(arrays, np.asarray([i]), TABLES, "cpu")
        # Match the production model's legacy phase inference when a shard has no phase column.
        if "decision_phase" not in batch:
            batch["decision_phase"] = SharedBattleLatent._decision_phase(
                batch, batch["legality_mask"]).long().argmax(dim=-1)
        wanted = {key: value for key, value in batch.items()
                  if key in ("entities", "entity_ids", "field", "history", "history_ids",
                             "history_state", "history_len", "active_idx", "decision_phase",
                             "legality_mask", "action_features", "opp_revealed")}
        out.append({"role": row["role"], "turn": row["turn"],
                    "fields": {**model_fields(wanted), "label": np.asarray(row["action"])}})
    return out


def compare_game(task) -> dict:
    import torch

    torch.set_num_threads(1)
    out, game = task
    live, replay = load_game(out / f"game_{game:05d}.npz")
    return {"game": game, "ended": replay["ended"],
            **compare_rows(live, replay_rows(replay)),
            "live_history": live_history_diagnostics(live)}


def collect_worker(task) -> None:
    from fog.config import load_config
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.teams import load_pool
    from fog.training.search_distill import build_agent
    import torch

    args, worker = task
    torch.set_num_threads(1)
    agent = build_agent(load_config(args.config), args.checkpoint, args.seed)
    pastes = load_pool(args.pool)
    with ShowdownDirectClient() as client:
        for game in range(worker, args.games, args.workers):
            path = args.out / f"game_{game:05d}.npz"
            if not path.exists():
                rows, replay = collect_game(client, agent, pastes, game, args.seed, args.max_turns)
                save_game(path, rows, replay)
            print(f"collected game {game}", flush=True)


def main() -> None:
    import torch

    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--pool", type=Path, default=Path("data/teams/pool_b"))
    parser.add_argument("--config", type=Path, default=Path("configs/replay_mechanical_cpu.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--max-turns", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ("report", "compare_only", "games", "workers")}
    settings_path = args.out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("collection settings changed")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    if not args.compare_only:
        if args.workers == 1:
            collect_worker((args, 0))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(collect_worker, [(args, i) for i in range(args.workers)]))
    if args.workers == 1:
        reports = [compare_game((args.out, game)) for game in range(args.games)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            reports = list(pool.map(compare_game, [(args.out, game) for game in range(args.games)]))
    fields = defaultdict(lambda: {"rows": 0, "exact": 0, "max_difference": 0.0})
    for report in reports:
        for name, entry in report["fields"].items():
            fields[name]["rows"] += entry["rows"]
            fields[name]["exact"] += entry["exact"]
            fields[name]["max_difference"] = max(fields[name]["max_difference"],
                                                    entry["max_difference"])
    for entry in fields.values():
        entry["exact_share"] = entry["exact"] / entry["rows"]
    report = {"kind": "smoke", "games": args.games,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "aligned": sum(r["aligned"] for r in reports),
              "missing_replay": sum(len(r["missing_replay"]) for r in reports),
              "missing_live": sum(len(r["missing_live"]) for r in reports),
              "fields": dict(fields),
              "live_history": {key: sum(r["live_history"][key] for r in reports)
                               for key in reports[0]["live_history"]},
              "games_detail": reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "games_detail"}, indent=2))


if __name__ == "__main__":
    main()
