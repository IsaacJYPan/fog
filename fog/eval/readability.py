"""Readability evaluation driver — the L-vs-N headline measurements.

Pipeline (one command via `python -m fog.eval.run_eval ... --readability`):
  1. roll `games` battles of the evaluated checkpoint vs a fixed opponent, recording the mirror
     (opponent's-view) observation at every decision + per-game behavioral case studies;
  2. split whole games into fitting, validation, and test sets; train a FRESH TrainableProbe on
     fitting games and select its checkpoint using validation games;
  3. report the readability curve on held-out test games (probe CE over still-hidden attrs — lower = more
     readable), per-attribute-family CE, and the behavioral measurements (tera timing, choice-lock
     concealment delay, move-reveal-order entropy).

`lvn_match` plays an L-checkpoint directly against an N-checkpoint with Wilson intervals, and
`pareto_report` assembles the β-grid readability/win-rate frontier — the strength-controlled framing
(a weaker agent is trivially harder to read; the frontier makes the trade explicit).
"""
from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_BACKBONE_KEYS = (
    "entities", "entity_ids", "field", "history", "active_idx", "history_len",
    "history_ids", "history_state",
)


# ------------------------------------------------------------------------------ per-game case studies
@dataclass
class CaseStudies:
    """Behavioral measurements, one entry per game (or per mon-game where noted)."""

    tera_turns: list[int] = field(default_factory=list)          # 0 = never terastallized
    choice_delays: list[int] = field(default_factory=list)       # turns from first move to lock evidence
    reveal_orders: list[tuple[int, ...]] = field(default_factory=list)  # per mon-game move-slot order

    def extend_from_battle(self, battle) -> None:
        my_role = getattr(battle, "player_role", None) or "p1"
        from fog.encoding.history import observations_by_turn

        obs_map = observations_by_turn(battle)
        events_by_turn = {t: (getattr(o, "events", None) or []) for t, o in obs_map.items()}

        from fog.encoding.events import event_mon_key, event_role, to_id

        tera_turn = 0
        moves_used: dict[str, list[str]] = defaultdict(list)     # mon_key -> distinct moves in order
        first_move_turn: dict[str, int] = {}
        lock_evidence_turn: dict[str, int] = {}
        last_move: dict[str, str] = {}
        active: str | None = None

        for t in sorted(events_by_turn):
            for ev in events_by_turn[t]:
                if len(ev) < 3:
                    continue
                kind, target = ev[1], ev[2]
                if event_role(target) != my_role:
                    continue
                key = event_mon_key(target)
                if kind == "-terastallize" and tera_turn == 0:
                    tera_turn = t
                elif kind in ("switch", "drag"):
                    if active is not None:
                        last_move.pop(active, None)              # lock evidence resets on switch
                    active = key
                elif kind == "move" and len(ev) >= 4:
                    move = to_id(ev[3])
                    if move not in moves_used[key]:
                        moves_used[key].append(move)
                    first_move_turn.setdefault(key, t)
                    if last_move.get(key) == move and key not in lock_evidence_turn:
                        lock_evidence_turn[key] = t              # same move twice without switching
                    last_move[key] = move

        self.tera_turns.append(tera_turn)

        # choice-lock concealment: for my choice-item holders, turns between the first move and the
        # first repeat-without-switching (the earliest public evidence of a lock). Games where the
        # lock never becomes evident are censored (skipped) — report coverage alongside the mean.
        team = getattr(battle, "team", None) or {}
        for key, mon in team.items():
            item = to_id(getattr(mon, "item", "") or "")
            if not item.startswith("choice"):
                continue
            k = str(key)
            if k in first_move_turn and k in lock_evidence_turn:
                self.choice_delays.append(lock_evidence_turn[k] - first_move_turn[k])

        # move-reveal order: per mon that moved, the order of first use as move-slot indices
        for key, mon in team.items():
            used = moves_used.get(str(key))
            if not used:
                continue
            slot_of = {to_id(m): i for i, m in enumerate(list(getattr(mon, "moves", None) or {})[:4])}
            order = tuple(slot_of[m] for m in used if m in slot_of)
            if order:
                self.reveal_orders.append(order)

    def report(self) -> dict:
        from fog.eval.leakage_metrics import (
            choice_lock_reveal_delay,
            move_reveal_entropy,
            tera_timing_distribution,
        )

        return {
            "tera_timing_hist": tera_timing_distribution(self.tera_turns).tolist(),
            "tera_never_frac": float(np.mean([t == 0 for t in self.tera_turns])) if self.tera_turns else float("nan"),
            "choice_lock_delay_mean": choice_lock_reveal_delay(self.choice_delays),
            "choice_lock_n": len(self.choice_delays),
            "move_reveal_entropy": move_reveal_entropy(self.reveal_orders),
        }


# ------------------------------------------------------------------------------------- data collection
def collect_trajectories(ckpt: str, opponent: str, n_games: int, fmt: str, team_pool, cfg: dict,
                         device=None) -> tuple[dict, CaseStudies]:
    """Play `n_games` of `ckpt` vs `opponent`, recording mirror observations + case studies.

    Returns (batch, case_studies): `batch` is a RolloutBuffer batch whose `opp_view`/`turn` fields
    feed probe training. The policy samples at `eval.readability.temperature` (default 1.0) — NOT
    the global `eval.temperature` (typically 0.0 for greedy gauntlets): readability and reveal-order
    entropy of a deterministic policy are different, much narrower quantities, and the global greedy
    setting silently poisoned this measurement once.
    """
    import torch

    from fog.env.player import FogPlayer, unique_account
    from fog.env.teams import make_pool_teambuilder
    from fog.eval.harness import build_player
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

    device = device or torch.device("cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()

    buf = RolloutBuffer()
    cases = CaseStudies()
    rd_temp = float(cfg.get("eval", {}).get("readability", {}).get("temperature", 1.0))
    adapter = FogPlayer(agent, temperature=rd_temp)
    adapter.attach_buffer(buf)
    adapter.on_battle_end = cases.extend_from_battle
    me = adapter.build(
        battle_format=fmt, max_concurrent_battles=1,
        account_configuration=unique_account("fogrd"),
        team=make_pool_teambuilder(team_pool, seed=11) if team_pool else None,
    )
    opp = build_player(opponent, fmt, make_pool_teambuilder(team_pool, seed=12) if team_pool else None,
                       cfg, device)

    async def _run():
        await me.battle_against(opp, n_battles=n_games)

    asyncio.run(_run())
    batch = buf.to_batch(device)
    if "opp_view" not in batch:
        raise RuntimeError(
            "no mirror observations were recorded — readability eval needs built constant tables "
            "(FogPlayer.record_mirror auto-disabled)."
        )
    wins = me.n_won_battles
    finished = me.n_finished_battles
    batch["_win_rate"] = wins / max(1, finished)
    batch["_n_games"] = finished
    return batch, cases


# ---------------------------------------------------------------------------------- probe fit + curves
def split_probe_games(batch: dict, *, train_fraction: float = 0.6,
                      validation_fraction: float = 0.2, seed: int = 0) -> tuple[dict, dict, dict]:
    """Split complete games, never adjacent turns, into fit/validation/final-test sets."""
    import torch

    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("probe fractions must leave nonempty train, validation and test sets")
    ids = batch["game_id"]
    groups = torch.unique(ids).cpu()
    if len(groups) < 3:
        raise ValueError("held-out readability requires at least three complete games")
    groups = groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed))]
    n_train = min(len(groups) - 2, max(1, int(len(groups) * train_fraction)))
    n_validation = min(len(groups) - n_train - 1,
                       max(1, int(len(groups) * validation_fraction)))

    def subset(chosen):
        indices = torch.where(torch.isin(ids, chosen.to(ids.device)))[0]

        def select(value):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(ids):
                return value[indices.to(value.device)]
            if isinstance(value, dict):
                return {key: select(child) for key, child in value.items()}
            return value

        part = {key: select(value) for key, value in batch.items() if key != "preview"}
        part["_n_games"] = len(chosen)
        part["_split_game_ids"] = sorted(chosen.tolist())
        # Do not present full-corpus win rate as a split-specific measurement.
        part.pop("_win_rate", None)
        return part

    return (subset(groups[:n_train]), subset(groups[n_train:n_train + n_validation]),
            subset(groups[n_train + n_validation:]))


def train_fresh_probe(batch: dict, cfg: dict, device=None, epochs: int = 8,
                      minibatch: int = 512, lr: float = 3e-4, seed: int = 0,
                      validation: dict | None = None):
    """Fit a from-scratch TrainableProbe on the collected mirror views for readability evaluation."""
    import torch

    from fog.models.agent import FogAgent
    from fog.models.probe import TrainableProbe

    torch.manual_seed(seed)
    # sizes come from the checkpointed agent's tables — rebuild a throwaway agent for them
    probe = TrainableProbe.from_agent(FogAgent.from_tables(cfg), cfg)
    if device is not None:
        probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    opp_view = batch["opp_view"]
    n = opp_view["entities"].shape[0]

    from fog.training.ppo import _index_nested

    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=opp_view["entities"].device)
        for start in range(0, n, minibatch):
            sub = _index_nested(opp_view, perm[start : start + minibatch])
            opt.zero_grad(set_to_none=True)
            loss = probe.loss(sub)
            loss.backward()
            opt.step()
        if validation is not None:
            probe.eval()
            view = validation["opp_view"]
            count = view["entities"].shape[0]
            total = 0.0
            with torch.no_grad():
                for start in range(0, count, minibatch):
                    idx = torch.arange(start, min(start + minibatch, count),
                                       device=view["entities"].device)
                    total += float(probe.loss(_index_nested(view, idx))) * len(idx)
            score = total / count
            if score < best_loss:
                best_loss = score
                best_state = copy.deepcopy(probe.state_dict())
    if validation is not None and best_state is None:
        raise RuntimeError("readability probe produced no finite validation checkpoint")
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe


def readability_curves(probe, batch: dict, minibatch: int = 512) -> dict:
    """Probe CE over still-hidden attrs, bucketed by turn + split by family. Lower = readable."""
    import torch

    from fog.eval.leakage_metrics import readability_curve
    from fog.training.ppo import _index_nested

    opp_view = batch["opp_view"]
    turns = batch["turn"]
    n = opp_view["entities"].shape[0]
    ce_by_turn: list[tuple[int, float]] = []
    fam_totals: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, n, minibatch):
            idx = torch.arange(start, min(start + minibatch, n), device=turns.device)
            sub = _index_nested(opp_view, idx)
            mirror = {k: sub[k] for k in _BACKBONE_KEYS if k in sub and sub[k] is not None}
            per_fam = probe.leakage_ce(mirror, sub["my_view"], per_family=True)
            total = sum(per_fam.values())
            for i, t in enumerate(turns[idx].tolist()):
                ce_by_turn.append((int(t), float(total[i])))
            for fam, v in per_fam.items():
                fam_totals[fam].extend(v.tolist())
    return {
        "curve": readability_curve(ce_by_turn),
        "per_family_ce": {fam: float(np.mean(v)) for fam, v in fam_totals.items()},
        "mean_ce": float(np.mean([c for _, c in ce_by_turn])),
    }


# ------------------------------------------------------------------------------------------- drivers
def run_readability(ckpt: str, cfg: dict, opponent: str | None = None, games: int | None = None,
                    device=None) -> dict:
    """The readability and behavior report for a checkpoint. Returns (and prints) the full dict."""
    ev = cfg.get("eval", {})
    rd = ev.get("readability", {})
    fmt = cfg.get("env", {}).get("battle_format", "gen9ou")
    team_pool = None
    if cfg.get("env", {}).get("team_pool"):
        from fog.env.teams import load_pool

        team_pool = load_pool(cfg["env"]["team_pool"])
    opponent = opponent or rd.get("opponent", "SimpleHeuristics")
    games = int(games or rd.get("games", 200))

    batch, cases = collect_trajectories(ckpt, opponent, games, fmt, team_pool, cfg, device)
    train, validation, test = split_probe_games(
        batch, train_fraction=float(rd.get("train_fraction", 0.6)),
        validation_fraction=float(rd.get("validation_fraction", 0.2)),
        seed=int(rd.get("split_seed", 0)),
    )
    probe = train_fresh_probe(train, cfg, device, epochs=int(rd.get("probe_epochs", 8)),
                              validation=validation, seed=int(rd.get("probe_seed", 0)))
    report = {
        "ckpt": str(ckpt), "opponent": opponent, "games": int(batch["_n_games"]),
        "win_rate": float(batch["_win_rate"]),
        **readability_curves(probe, test),
        "evaluation_scope": "held_out_games",
        "probe_splits": {
            name: {"games": part["_n_games"], "game_ids": part["_split_game_ids"]}
            for name, part in (("train", train), ("validation", validation), ("test", test))
        },
        "case_studies": cases.report(),
    }
    _print_readability(report)
    return report


def _print_readability(r: dict) -> None:
    print(f"readability: {r['ckpt']} vs {r['opponent']}  ({r['games']} games, "
          f"win_rate={r['win_rate']:.3f})")
    print(f"  mean still-hidden CE (higher = harder to read): {r['mean_ce']:.3f}")
    print("  by turn bucket: " + "  ".join(f"t{t}+:{ce:.2f}" for t, ce in r["curve"].items()))
    print("  by family:      " + "  ".join(f"{f}:{ce:.2f}" for f, ce in r["per_family_ce"].items()))
    cs = r["case_studies"]
    print(f"  tera never-frac {cs['tera_never_frac']:.2f}  "
          f"choice-lock delay {cs['choice_lock_delay_mean']:.2f} (n={cs['choice_lock_n']})  "
          f"reveal-order entropy {cs['move_reveal_entropy']:.3f}")


def lvn_match(ckpt_l: str, ckpt_n: str, n_games: int, fmt: str, team_pool, cfg: dict,
              cfg_n: dict | None = None) -> dict:
    """L-agent vs N-agent head-to-head with a Wilson CI.

    Paired design: each sampled team matchup is played twice with sides swapped, so the team-matchup
    lottery (which dominates single games on a small pool) cancels out of the comparison.

    `cfg_n` builds the N side when the two checkpoints differ architecturally (e.g. the M1.5-vs-M1
    b_vec gate: use_belief_vec changes the head width, so one shared cfg cannot load both).
    Defaults to `cfg`.
    """
    from fog.eval.harness import run_match
    from fog.eval.stats import wilson_interval

    wins, finished = run_match(ckpt_l, ckpt_n, n_games, fmt, team_pool=team_pool, cfg=cfg,
                               paired=True, cfg_b=cfg_n)
    iv = wilson_interval(wins, finished)
    warn = "" if finished >= 1000 else "  [insufficient n: <1000, do not conclude]"
    print(f"L vs N: {wins}/{finished} = {iv.point:.3f} [{iv.low:.3f}, {iv.high:.3f}]{warn}")
    return {"wins": wins, "games": finished, "point": iv.point, "low": iv.low, "high": iv.high}


def pareto_report(entries: list[dict], cfg: dict, device=None) -> list[dict]:
    """β-grid frontier (strength-controlled framing): for each {label, ckpt, beta}, measure win-rate
    + fresh-probe readability vs the SAME fixed opponent, and print the frontier table. Readability
    comparisons at matched win-rate come from reading adjacent rows of this table."""
    rows = []
    for e in entries:
        r = run_readability(e["ckpt"], cfg, device=device)
        rows.append({
            "label": e.get("label", e["ckpt"]), "beta": float(e.get("beta", 0.0)),
            "win_rate": r["win_rate"], "mean_ce": r["mean_ce"],
            "per_family_ce": r["per_family_ce"],
        })
    rows.sort(key=lambda r: r["beta"])
    print("\nβ-grid readability/win-rate frontier (vs fixed opponent):")
    print(f"  {'label':<24} {'beta':>6} {'win_rate':>9} {'still-hidden CE':>16}")
    for r in rows:
        print(f"  {r['label']:<24} {r['beta']:>6.3f} {r['win_rate']:>9.3f} {r['mean_ce']:>16.3f}")
    return rows
