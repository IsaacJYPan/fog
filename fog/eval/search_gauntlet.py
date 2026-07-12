"""Phase-2 search gauntlet (SEARCH_PLAN §5): validate the joint-search expert.

Plays engine games where the SAME checkpoint acts through three arms:

  * policy       — plain batched forward, no search (0 sims/decision);
  * determinized — §8 root determinization over MY uncertainty about the OPPONENT
                   (engine_step.EngineWorldStepper);
  * joint        — §4.3 joint search over the OPPONENT's belief about ME
                   (exact pool enumeration — the Phase-2 setting; no sampling noise).

Two measurements, per SEARCH_PLAN §5:

  * gauntlet rows: each arm vs MaxDamage AND vs the set-oracle READING opponent — the same
    checkpoint played with the learner's true team injected (`model.oracle: set`; engine truth is
    exact, `battle.truth_tokens`). Signaling value is invisible against MaxDamage, so the reading
    opponent is the row the gate reads. NOTE what the oracle reader is: an upper-bound READER
    (truth is handed to it, concealment cannot deny it) — it makes information VALUE visible,
    while the L-vs-N-style head-to-head below is where concealment/bluffing can pay.
  * head-to-head rows: joint vs determinized and joint vs policy directly, PAIRED team schedule
    (each sampled matchup (X, Y) is played twice with the arms' teams swapped, cancelling the
    team-matchup lottery exactly as eval/readability.lvn_match does on the Showdown path).

Gate (§5, evaluated in `gate_report`): joint > determinized AND joint > policy, vs the reading
opponent and in head-to-head, with Wilson CIs. `pass_point` is the point-estimate ordering;
`pass_ci` requires non-overlapping 95% intervals — quote the latter for claims.

Matched simulation budget: both search arms spend ~(worlds × legal actions) one-ply engine sims
per decision. Determinized's world count defaults to the pool size — the joint arm's enumeration
CEILING — but joint's K shrinks as the game reveals my attributes while determinized's k stays
fixed, so the match is verified, not assumed: every row reports measured sims/decision
(`sims_per_decision`), and budget-sensitive conclusions should quote those numbers.

Games run ONE at a time (search batches internally over worlds × actions — same reasoning as
fog/search/posterior_log.py). Failure containment mirrors posterior_log: a game that raises is
dropped and counted, and a >20% failure rate raises rather than reporting on a corrupted sample.
All of this is engine-backend expert VALIDATION (SEARCH_PLAN Phase 2) — headline numbers stay on
Showdown per the binding protocol; these rows never are.

    python -m fog.eval.search_gauntlet --pool data/teams/pool_a \
        --ckpt runs/value_finetune_m15/value_finetune.pt --games 400 --out runs/search_gauntlet

Checkpoints: posterior-blind ones (e.g. raw M1.5) are zero-pad expanded on load
(`value_finetune.load_posterior_agent`) — but the §4.2 fine-tune gate must have PASSED before any
Phase-2 conclusion (untrained posterior columns = weak coupling = a vacuous joint arm).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LEARNER = "p1"
OPPONENT = "p2"
ARMS = ("policy", "determinized", "joint")
GAUNTLET_OPPONENTS = ("MaxDamage", "SetOracle")


# ------------------------------------------------------------------------------------ arm players
class ArmPlayer:
    """Chooses actions for one side of a live EngineBattle under one §5 arm.

    Same `choose(battle, role)` contract as the engine_scripted policies, so game loops treat
    scripted opponents and arms uniformly. Counters (`decisions`, `sims`, `failures`) accumulate
    across games — one ArmPlayer per match row.

    `oracle_truth=True` makes this the READING opponent: the other side's exact truth tokens are
    injected into every forward (`model.oracle: set` must be on in the agent's config — raised
    otherwise, a silent non-oracle reader would fake the gate row).
    """

    records_history = True  # game loop feeds tracker retro-fidelity for agent-driven sides

    def __init__(self, arm: str, agent: Any, *, pool: list[str] | None = None, tables=None,
                 det_k: int | None = None, prior_weight: float = 0.5, iters: int = 16,
                 temperature: float = 0.0, seed: int = 0, oracle_truth: bool = False,
                 own_source: Any | None = None) -> None:
        import random

        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r} (choose from {ARMS})")
        if tables is None:
            from fog.constants.tables import TABLES as tables
        self.arm = arm
        self.agent = agent
        self.tables = tables
        self.det_k = det_k
        self.prior_weight = float(prior_weight)
        self.iters = int(iters)
        self.temperature = float(temperature)
        self.rng = random.Random(seed)
        self.oracle_truth = bool(oracle_truth)
        if oracle_truth and getattr(agent, "oracle", "none") not in ("set", "action"):
            raise ValueError("oracle_truth=True needs an agent built with model.oracle='set' — "
                             "this one would silently ignore the injected truth")
        if oracle_truth and arm != "policy":
            raise ValueError("the reading opponent is a policy-arm ceiling; searching with "
                             "injected truth is not a §5 arm")
        # joint arm's own-state source: exact pool enumeration by default (Phase 2); pass a
        # SampledOwnStates for the Phase-3 sampled arm
        self._own_source = own_source
        if arm == "joint" and self._own_source is None:
            from fog.search.own_states import EnumeratedPoolStates

            if not pool:
                raise ValueError("the joint arm needs the team pool (exact enumeration)")
            self._own_source = EnumeratedPoolStates(pool, tables)
        if arm == "determinized" and det_k is None:
            raise ValueError("the determinized arm needs det_k (worlds per action)")

        self.decisions = 0
        self.sims = 0
        self.failures = 0

    # ---- the three arms ----
    def choose(self, battle: Any, role: str) -> int:
        legal = np.flatnonzero(battle.action_mask(role))
        if len(legal) == 0:
            return 13
        if len(legal) == 1:  # no decision — don't spend sims (all arms agree here by construction)
            return int(legal[0])
        self.decisions += 1
        if self.arm == "policy":
            return self._choose_policy(battle, role)
        if self.arm == "determinized":
            return self._choose_determinized(battle, role)
        return self._choose_joint(battle, role)

    def _batch(self, battle: Any, role: str) -> dict:
        import torch

        from fog.encoding.observation import Observation
        from fog.env.engine_runner import _device, _stack_obs

        obs = Observation.from_battle(battle.view(role), tracker=battle.trackers[role])
        batch = _stack_obs([obs], _device(self.agent))
        if self.oracle_truth:
            other = OPPONENT if role == LEARNER else LEARNER
            batch["opp_truth_tokens"] = torch.as_tensor(
                battle.truth_tokens(other)[None], dtype=torch.long, device=_device(self.agent)
            )
        return batch

    def _choose_policy(self, battle: Any, role: str) -> int:
        import torch

        from fog.env.player import _sample_action

        with torch.no_grad():
            out = self.agent(self._batch(battle, role))
        return _sample_action(out.log_probs[0], temperature=self.temperature)

    def _choose_determinized(self, battle: Any, role: str) -> int:
        import torch

        from fog.env.engine_runner import _device
        from fog.search.determinized import determinized_action
        from fog.search.engine_step import EngineWorldStepper

        stepper = EngineWorldStepper(battle, role, seed=self.rng.randrange(2**31))

        def counted_step(action_idx, world_tokens):
            self.sims += 1
            return stepper(action_idx, world_tokens)

        gen = torch.Generator(device=_device(self.agent))
        gen.manual_seed(self.rng.randrange(2**31))
        action = determinized_action(
            self.agent, self._batch(battle, role), step_fn=counted_step,
            k=int(self.det_k), prior_weight=self.prior_weight, generator=gen,
        )
        self.failures += stepper.failures
        return action

    def _choose_joint(self, battle: Any, role: str) -> int:
        from fog.search.joint import joint_search_action

        res = joint_search_action(
            self.agent, battle, role, own_source=self._own_source, iters=self.iters,
            temperature=self.temperature, seed=self.rng.randrange(2**31),
        )
        # floor-mixing keeps every available (state, action) cell strictly positive and every
        # unavailable cell exactly zero, so the support count IS the simulation count
        self.sims += int(np.count_nonzero(res.policies))
        self.failures += res.failures
        return res.action


# -------------------------------------------------------------------------------------- game loop
def _play_game(battle: Any, sides: dict[str, Any], *, max_turns: int = 300):
    """Step one engine game to completion; `sides` maps role -> ArmPlayer or scripted policy.
    Returns battle.winner (None on a max_turns truncation = draw)."""
    from fog.encoding.events import build_snapshot

    while not battle.finished and battle.turn <= max_turns:
        moves: dict[str, str] = {}
        for role in battle.pending_sides():
            idx = sides[role].choose(battle, role)
            if getattr(sides[role], "records_history", False):
                try:  # history retro-fidelity, same as EngineRunner._agent_wave
                    battle.trackers[role].record_decision(
                        battle.turn, build_snapshot(battle.view(role), battle.tables), idx
                    )
                except Exception:
                    pass
            moves[role] = battle.engine_action(role, idx)
        battle.step(moves)
    return battle.winner


@dataclass
class MatchRow:
    label: str            # e.g. "joint vs SetOracle" / "joint vs determinized [h2h]"
    wins: int = 0
    games: int = 0
    failed_games: int = 0
    decisions: int = 0
    sims: int = 0
    failures: int = 0     # per-simulation failures inside search arms
    turns: int = 0
    truncated: int = 0    # games that hit max_turns (drawn; a rash of them = degenerate stalling)
    meta: dict = field(default_factory=dict)

    @property
    def sims_per_decision(self) -> float:
        return self.sims / self.decisions if self.decisions else 0.0

    def line(self) -> str:
        from fog.eval.stats import wilson_interval

        iv = wilson_interval(self.wins, self.games)
        warn = "" if self.games >= 1000 else "  [insufficient n: <1000, do not conclude]"
        trunc = f" | truncated {self.truncated}" if self.truncated else ""
        return (f"{self.label:<34} {self.wins}/{self.games} = {iv.point:.3f} "
                f"[{iv.low:.3f}, {iv.high:.3f}] | sims/dec {self.sims_per_decision:6.1f} "
                f"| avg turns {self.turns / max(1, self.games):5.1f} "
                f"| sim failures {self.failures}{trunc}{warn}")

    def as_dict(self) -> dict:
        from fog.eval.stats import wilson_interval

        iv = wilson_interval(self.wins, self.games)
        return {"label": self.label, "wins": self.wins, "games": self.games,
                "point": iv.point, "low": iv.low, "high": iv.high,
                "failed_games": self.failed_games, "decisions": self.decisions,
                "sims": self.sims, "sims_per_decision": self.sims_per_decision,
                "sim_failures": self.failures, "turns": self.turns,
                "truncated": self.truncated, **self.meta}


def _team_schedule(pool: list[str], n_games: int, rng, *, paired: bool) -> list[tuple[str, str]]:
    """(p1_paste, p2_paste) per game. `paired`: each sampled matchup (X, Y) is played twice with
    the sides' TEAMS swapped — the lvn_match variance-reduction design."""
    if not paired:
        return [(rng.choice(pool), rng.choice(pool)) for _ in range(int(n_games))]
    pastes: list[tuple[str, str]] = []
    for _ in range((int(n_games) + 1) // 2):
        x, y = rng.choice(pool), rng.choice(pool)
        pastes += [(x, y), (y, x)]
    return pastes[: int(n_games)]


def run_search_match(side_a: Any, side_b: Any, pool: list[str], n_games: int, *, label: str,
                     tables=None, seed: int = 0, max_turns: int = 300,
                     paired: bool = False) -> MatchRow:
    """Play `n_games` engine games of side_a (p1) vs side_b (p2), teams sampled from `pool`
    (`paired` — see `_team_schedule`). Failed games are dropped and counted; >20% failed raises
    (posterior_log's containment rule)."""
    import random

    from fog.env.engine_battle import EngineBattle

    if tables is None:
        from fog.constants.tables import TABLES as tables
    rng = random.Random(seed)
    row = MatchRow(label=label)
    pastes = _team_schedule(pool, n_games, rng, paired=paired)

    for g, (paste_a, paste_b) in enumerate(pastes):
        battle = EngineBattle(paste_a, paste_b, seed=rng.randrange(2**31),
                              tables=tables, battle_tag=f"sgaunt-{g}")
        try:
            winner = _play_game(battle, {LEARNER: side_a, OPPONENT: side_b},
                                max_turns=max_turns)
        except Exception as exc:
            row.failed_games += 1
            print(f"[search_gauntlet] game {g} ({label}) failed and was dropped: {exc!r}",
                  flush=True)
            continue
        row.games += 1
        row.wins += int(winner == LEARNER)
        row.turns += battle.turn
        row.truncated += int(not battle.finished)

    if row.failed_games > max(1, int(n_games) // 5):
        raise RuntimeError(f"{row.failed_games}/{n_games} games failed in {label!r} — that is a "
                           "defect, not noise; fix it before trusting these rows.")
    if isinstance(side_a, ArmPlayer):
        row.decisions, row.sims, row.failures = side_a.decisions, side_a.sims, side_a.failures
    if isinstance(side_b, ArmPlayer) and side_b.decisions:  # h2h: the other arm's budget too
        row.meta["opp_decisions"] = side_b.decisions
        row.meta["opp_sims"] = side_b.sims
        row.meta["opp_sims_per_decision"] = side_b.sims / side_b.decisions
        row.meta["opp_sim_failures"] = side_b.failures
    return row


# --------------------------------------------------------------------------------- the full §5 run
def run_search_gauntlet(
    ckpt: str, pool: list[str], cfg: dict, device, *,
    games_per: int = 200, h2h_games: int = 200, arms=ARMS,
    opponents=GAUNTLET_OPPONENTS, det_k: int | None = None, iters: int = 16,
    prior_weight: float = 0.5, temperature: float = 0.0, seed: int = 0,
    max_turns: int = 300, tables=None, own_source_kind: str = "enumerated",
    sample_k: int = 16,
) -> dict:
    """Gauntlet rows (each arm vs each opponent) + head-to-head rows (joint vs the other arms),
    then the §5 gate report. Returns {"rows": [...], "gate": {...}} (also printed).

    `own_source_kind`: 'enumerated' (exact pool enumeration — Phase 2) or 'sampled' (the learned
    SelfBeliefPredictor sampler at `sample_k` states — the Phase-3 scalability arm; only the
    joint arm consumes it)."""
    import copy

    from fog.training.value_finetune import load_posterior_agent

    if tables is None:
        from fog.constants.tables import TABLES as tables
    if own_source_kind not in ("enumerated", "sampled"):
        raise ValueError(f"own_source_kind must be 'enumerated' or 'sampled', "
                         f"got {own_source_kind!r}")
    det_k = int(det_k) if det_k is not None else len(pool)
    agent = load_posterior_agent(cfg, ckpt, device)
    agent.eval()

    def make_arm(arm: str, arm_seed: int) -> ArmPlayer:
        own_source = None
        if arm == "joint" and own_source_kind == "sampled":
            from fog.search.own_states import SampledOwnStates

            own_source = SampledOwnStates(agent, tables, k=int(sample_k), seed=arm_seed)
        return ArmPlayer(arm, agent, pool=pool, tables=tables, det_k=det_k,
                         prior_weight=prior_weight, iters=iters, temperature=temperature,
                         seed=arm_seed, own_source=own_source)

    def make_opponent(name: str, opp_seed: int):
        if name == "SetOracle":
            oracle_cfg = copy.deepcopy(cfg)
            oracle_cfg.setdefault("model", {})["oracle"] = "set"
            reader = load_posterior_agent(oracle_cfg, ckpt, device)
            reader.eval()
            return ArmPlayer("policy", reader, tables=tables, temperature=temperature,
                             seed=opp_seed, oracle_truth=True)
        import inspect

        from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES

        if name not in SCRIPTED_ENGINE_POLICIES:
            raise ValueError(f"opponent {name!r} is neither 'SetOracle' nor an engine-scripted "
                             f"policy (available: {sorted(SCRIPTED_ENGINE_POLICIES)})")
        cls = SCRIPTED_ENGINE_POLICIES[name]
        kwargs = {"seed": opp_seed}
        if "tables" in inspect.signature(cls.__init__).parameters:
            kwargs["tables"] = tables
        return cls(**kwargs)

    rows: list[MatchRow] = []
    print(f"[search_gauntlet] ckpt={ckpt} pool={len(pool)} teams | det_k={det_k} iters={iters} "
          f"| {games_per} games/row, {h2h_games} games/h2h row", flush=True)
    for a_i, arm in enumerate(arms):
        for o_i, opp_name in enumerate(opponents):
            row = run_search_match(
                make_arm(arm, seed * 1009 + a_i * 101 + o_i),
                make_opponent(opp_name, seed * 1009 + a_i * 101 + o_i + 50),
                pool, games_per, label=f"{arm} vs {opp_name}", tables=tables,
                seed=seed * 7919 + a_i * 13 + o_i, max_turns=max_turns,
            )
            row.meta.update({"arm": arm, "opponent": opp_name, "kind": "gauntlet"})
            rows.append(row)
            print("  " + row.line(), flush=True)

    if "joint" in arms:
        for o_i, other in enumerate(a for a in arms if a != "joint"):
            row = run_search_match(
                make_arm("joint", seed * 2003 + o_i),
                make_arm(other, seed * 2003 + o_i + 500),
                pool, h2h_games, label=f"joint vs {other} [h2h]", tables=tables,
                seed=seed * 6007 + o_i, max_turns=max_turns, paired=True,
            )
            row.meta.update({"arm": "joint", "opponent": other, "kind": "h2h"})
            rows.append(row)
            print("  " + row.line(), flush=True)

    gate = gate_report(rows)
    return {"rows": [r.as_dict() for r in rows], "gate": gate,
            "config": {"ckpt": ckpt, "pool_size": len(pool), "games_per": games_per,
                       "h2h_games": h2h_games, "det_k": det_k, "iters": iters,
                       "prior_weight": prior_weight, "temperature": temperature, "seed": seed,
                       "own_source": own_source_kind, "sample_k": sample_k}}


def gate_report(rows: list[MatchRow]) -> dict:
    """§5 gate: joint > determinized AND joint > policy — vs the reading opponent and in the
    head-to-head rows. Point ordering and CI separation reported separately; if the gate fails,
    that is a finding, not an error (SEARCH_PLAN §5) — so this reports, it never raises."""
    from fog.eval.stats import wilson_interval

    def iv(row):
        return wilson_interval(row.wins, row.games)

    by = {(r.meta.get("kind"), r.meta.get("arm"), r.meta.get("opponent")): r for r in rows}
    checks = []
    joint_reader = by.get(("gauntlet", "joint", "SetOracle"))
    for other in ("determinized", "policy"):
        other_reader = by.get(("gauntlet", other, "SetOracle"))
        if joint_reader and other_reader:
            j, o = iv(joint_reader), iv(other_reader)
            checks.append({"check": f"joint > {other} (vs SetOracle)",
                           "joint": j.point, "other": o.point,
                           "pass_point": j.point > o.point, "pass_ci": j.low > o.high})
        h2h = by.get(("h2h", "joint", other))
        if h2h:
            h = iv(h2h)
            checks.append({"check": f"joint > {other} (head-to-head)",
                           "joint": h.point, "other": 1.0 - h.point,
                           "pass_point": h.point > 0.5, "pass_ci": h.low > 0.5})

    gate = {"checks": checks,
            "pass_point": bool(checks) and all(c["pass_point"] for c in checks),
            "pass_ci": bool(checks) and all(c["pass_ci"] for c in checks)}
    print("\n[search_gauntlet] §5 gate:", flush=True)
    for c in checks:
        print(f"  {c['check']:<40} joint {c['joint']:.3f} vs {c['other']:.3f} "
              f"| point {'PASS' if c['pass_point'] else 'FAIL'} "
              f"| CI {'PASS' if c['pass_ci'] else 'FAIL'}", flush=True)
    verdict = ("PASSED (CI-separated)" if gate["pass_ci"]
               else "point-passed only — more games or a real negative"
               if gate["pass_point"] else "FAILED — debug §4.2's consumer or accept the negative")
    print(f"  gate: {verdict}", flush=True)
    return gate


# ------------------------------------------------------------------------------------------- CLI
def main() -> None:
    import argparse

    import torch

    from fog.config import load_config_with_args
    from fog.env.teams import load_pool

    ap = argparse.ArgumentParser(description="Phase-2 search gauntlet (SEARCH_PLAN §5).")
    ap.add_argument("--pool", type=Path, default=Path("data/teams/pool_a"))
    ap.add_argument("--ckpt", type=str, required=True,
                    help="checkpoint (posterior-blind ones are expanded — but run the §4.2 "
                         "fine-tune first; its gate must pass before Phase-2 conclusions)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs/search_gauntlet"))
    ap.add_argument("--games", type=int, default=200, help="games per gauntlet row")
    ap.add_argument("--h2h-games", type=int, default=200, help="games per head-to-head row")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--opponents", type=str, default=",".join(GAUNTLET_OPPONENTS))
    ap.add_argument("--det-k", type=int, default=None,
                    help="determinized worlds/action (default: pool size = joint's ceiling)")
    ap.add_argument("--own-source", choices=("enumerated", "sampled"), default="enumerated",
                    help="joint arm's own-state source: exact enumeration (Phase 2) or the "
                         "learned sampler (Phase 3)")
    ap.add_argument("--sample-k", type=int, default=16,
                    help="states per decision for --own-source sampled")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--prior-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=300,
                    help="truncate games here (drawn + counted); greedy search play can stall, "
                         "and every stalled turn costs a full worlds×actions sim budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_config_with_args(args.config) if args.config else {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = run_search_gauntlet(
        args.ckpt, load_pool(args.pool), cfg, device,
        games_per=args.games, h2h_games=args.h2h_games,
        arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()),
        opponents=tuple(o.strip() for o in args.opponents.split(",") if o.strip()),
        det_k=args.det_k, iters=args.iters, prior_weight=args.prior_weight,
        temperature=args.temperature, max_turns=args.max_turns, seed=args.seed,
        own_source_kind=args.own_source, sample_k=args.sample_k,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[search_gauntlet] report -> {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
