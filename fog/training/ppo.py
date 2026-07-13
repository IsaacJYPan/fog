"""Training stage 2 — self-play PPO (spec §6).

PPO: clip 0.2, GAE λ=0.95, γ=1.0 (episodic win/loss ±1; optional +0.02 per opposing faint, off by
default), entropy bonus 0.01 decayed, value coef 0.5, batch 16k decision points, 3 epochs/batch,
lr 1e-4. Auxiliary belief/self-belief losses continue with FULL oracle labels (simulator provides both
teams). KL-to-BC-anchor penalty 0.02 for the first 2M decisions, annealed to 0.

Watch the §6 diagnostics dashboard from day one: win-rate vs each frozen league member; entropy per
turn-bucket; belief accuracy vs turn number (MUST be monotone increasing within games); value
calibration; tera-usage fraction; average game length.

`train` runs the full rollout <-> update loop: league opponent sampling -> live battles into a
RolloutBuffer -> GAE -> `ppo_update` -> league snapshots + checkpoints (+ the §7 leakage machinery
when leakage.beta > 0).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fog.training.league import League


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float = 1.0, lam: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation. Real implementation.

    rewards/values/dones are [T] for one trajectory; `values` has length T+1 (bootstrap at the end).
    Returns (advantages[T], returns[T]).
    """
    t = len(rewards)
    adv = np.zeros(t, dtype=np.float32)
    last = 0.0
    for i in reversed(range(t)):
        nonterminal = 1.0 - dones[i]
        delta = rewards[i] + gamma * values[i + 1] * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    returns = adv + values[:t]
    return adv, returns


def _index_nested(x, idx: torch.Tensor):
    """Recursively index tensors inside nested dicts by row (first dim). None passes through."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x[idx]
    if isinstance(x, dict):
        return {k: _index_nested(v, idx) for k, v in x.items()}
    return x


# -inf logits are clamped to this before any arithmetic: exp(-1e4) == 0 exactly in fp32, and unlike
# -inf it cannot produce NaN *gradients* (torch.where guards forward values only — the local backward
# of p*log p at log p = -inf is 0 * -inf = NaN and poisons every parameter).
_LOG_ZERO = -1.0e4


def _masked_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Per-row entropy of a legality-masked policy; -inf (illegal) entries contribute 0. Returns [N]."""
    lp = log_probs.clamp(min=_LOG_ZERO)
    p = lp.exp()
    return -(p * lp).sum(dim=-1)


def _masked_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row where both may contain -inf (same legality mask on both). Where p=0 the
    term contributes 0."""
    lp = log_p.clamp(min=_LOG_ZERO)
    lq = log_q.clamp(min=_LOG_ZERO)
    p = lp.exp()
    return (p * (lp - lq)).sum(dim=-1)


def ppo_update(agent, optimizer, batch: dict, cfg: dict) -> dict:
    """One PPO update over a collected batch (spec §6): 3 epochs of minibatches, clip 0.2.

    Loss = -clip_surrogate + value_coef * value_MSE - entropy_coef * entropy
           + belief_coef * belief_CE + self_belief_coef * self_belief_CE   (full oracle labels)
           + kl_anchor_coef * KL(pi || pi_BC)

    `batch` holds the stacked FogAgent obs tensors (entities, entity_ids, field, history, active_idx,
    legality_mask, optional opp_revealed / opp_view — see models/agent.py) plus, per decision point:
      actions [N] long, old_log_probs [N], advantages [N], returns [N],
      optional bc_log_probs [N, 14] (frozen BC anchor policy, full masked log-probs),
      optional belief_targets/belief_valid + self_belief_targets/self_belief_valid label dicts.

    Advantages are normalized here (over the full batch). Returns a metrics dict (means over
    minibatches): loss terms, approx_kl, clip_frac, entropy.
    """
    ppo_cfg = cfg.get("ppo", {})
    loss_cfg = cfg.get("loss", {})
    clip = ppo_cfg.get("clip", 0.2)
    epochs = ppo_cfg.get("epochs", 3)
    minibatch_size = ppo_cfg.get("minibatch_size", 2048)
    entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
    value_coef = loss_cfg.get("value_coef", 0.5)
    belief_coef = loss_cfg.get("belief_coef", 1.0)
    self_belief_coef = loss_cfg.get("self_belief_coef", 1.0)
    # BELIEF_PLAN §4.2: pass-B chain CE, its own coefficient (start equal to the marginal's).
    belief_chain_coef = loss_cfg.get("belief_chain_coef", belief_coef)
    self_belief_chain_coef = loss_cfg.get("self_belief_chain_coef", self_belief_coef)
    preview_coef = loss_cfg.get("preview_coef", 0.5)
    kl_anchor_coef = ppo_cfg.get("kl_anchor_coef", 0.0)
    max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)

    # preview rows are per-GAME, not per-decision — handled outside the minibatch permutation
    preview = batch.pop("preview", None)

    n = batch["actions"].shape[0]
    device = batch["actions"].device
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    from fog.training.losses import belief_loss, value_loss  # local import: avoid cycles at module load

    metrics: dict[str, list[float]] = {}

    def log(name: str, v: torch.Tensor | float) -> None:
        if isinstance(v, torch.Tensor):
            v = v.detach().item()
        metrics.setdefault(name, []).append(float(v))

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            mb = _index_nested(batch, idx)
            mb_adv = adv[idx]

            out = agent(mb)
            new_log_probs = out.log_probs.gather(1, mb["actions"].unsqueeze(1)).squeeze(1)

            ratio = (new_log_probs - mb["old_log_probs"]).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_loss = value_loss(out.value, mb["returns"])
            entropy = _masked_entropy(out.log_probs).mean()

            loss = policy_loss + value_coef * v_loss - entropy_coef * entropy

            if kl_anchor_coef > 0 and mb.get("bc_log_probs") is not None:
                kl_bc = _masked_kl(out.log_probs, mb["bc_log_probs"]).mean()
                loss = loss + kl_anchor_coef * kl_bc
                log("kl_bc", kl_bc)

            if out.belief is not None and mb.get("belief_targets") is not None:
                b_stats: dict = {}
                # Marginal (pass A) CE: the deployment reads, trained on every example (§3.3). Kept
                # under the historical "belief_loss" metric name (still the pass-A marginal).
                b_loss = belief_loss(
                    out.belief.logits, mb["belief_targets"], mb["belief_valid"], stats=b_stats
                )
                loss = loss + belief_coef * b_loss
                log("belief_loss", b_loss)
                log("belief_ce_marginal", b_loss)
                # dropped-target rate: ~0 post-alignment; systematic > 0 = convention bug (see
                # losses.masked_cross_entropy).
                b_total = b_stats.get("valid", 0) + b_stats.get("dropped_nonfinite", 0)
                log("belief_dropped_rate", b_stats.get("dropped_nonfinite", 0) / max(1, b_total))
                # Chain (pass B) CE: the sampling/scoring reads (§4.2). Separate coefficient + metric.
                if out.belief_chain is not None:
                    bc_loss = belief_loss(
                        out.belief_chain.logits, mb["belief_targets"], mb["belief_valid"]
                    )
                    loss = loss + belief_chain_coef * bc_loss
                    log("belief_ce_chain", bc_loss)
            if out.self_belief is not None and mb.get("self_belief_targets") is not None:
                sb_stats: dict = {}
                sb_loss = belief_loss(
                    out.self_belief.logits, mb["self_belief_targets"], mb["self_belief_valid"],
                    stats=sb_stats,
                )
                loss = loss + self_belief_coef * sb_loss
                log("self_belief_loss", sb_loss)
                log("self_belief_ce_marginal", sb_loss)
                sb_total = sb_stats.get("valid", 0) + sb_stats.get("dropped_nonfinite", 0)
                log("self_belief_dropped_rate",
                    sb_stats.get("dropped_nonfinite", 0) / max(1, sb_total))
                if out.self_belief_chain is not None:
                    sbc_loss = belief_loss(
                        out.self_belief_chain.logits, mb["self_belief_targets"],
                        mb["self_belief_valid"],
                    )
                    loss = loss + self_belief_chain_coef * sbc_loss
                    log("self_belief_ce_chain", sbc_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log("loss", loss)
                log("policy_loss", policy_loss)
                log("value_loss", v_loss)
                log("entropy", entropy)
                log("approx_kl", (mb["old_log_probs"] - new_log_probs).mean())
                log("clip_frac", ((ratio - 1.0).abs() > clip).float().mean())

        # team-preview bandit (§3.1): one clipped-surrogate step per epoch over all preview rows.
        # Reward = game outcome; baseline = batch mean outcome (variance reduction).
        if preview is not None and preview_coef > 0:
            logits = agent.preview_logits(preview)                                # [G, 6]
            lp = torch.log_softmax(logits, dim=-1).gather(
                1, preview["slot"].unsqueeze(1)
            ).squeeze(1)
            pv_adv = preview["outcome"] - preview["outcome"].mean()
            pv_ratio = (lp - preview["old_log_prob"]).exp()
            pv_s1 = pv_ratio * pv_adv
            pv_s2 = torch.clamp(pv_ratio, 1.0 - clip, 1.0 + clip) * pv_adv
            pv_loss = -torch.min(pv_s1, pv_s2).mean() * preview_coef
            optimizer.zero_grad(set_to_none=True)
            pv_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            log("preview_loss", pv_loss)

    return {k: sum(v) / len(v) for k, v in metrics.items()}


def train(cfg: dict) -> dict:
    """Self-play PPO loop (§6): rollouts collect into a RolloutBuffer against league opponents;
    ppo_update consumes it; checkpoints snapshot into the league. Returns the last metrics dict.

    `env.backend` selects the simulator (SIMULATION_PLAN):
      * "showdown" (default) — poke-env vs a live local server; ground truth. Byte-identical
        behavior to the pre-backend-switch trainer.
      * "engine"             — poke-engine batched lockstep self-play (fog.env.engine_runner);
        volume track for M3/M4 grids. No server needed. Final fine-tune + all headline evals
        stay on Showdown (divergence rules in SIMULATION_PLAN §5).

    Config keys used (all under sensible defaults): env.{battle_format, team_pool, backend,
    engine_batch}, ppo.{lr, total_decisions, batch_decisions, games_per_chunk, ...ppo_update keys},
    league.snapshot_every_decisions, leakage.beta (0 = N-agent), log.dir.
    """
    backend = str(cfg.get("env", {}).get("backend", "showdown")).lower()
    if backend == "engine":
        return _train_engine(cfg)
    if backend != "showdown":
        raise ValueError(f"env.backend must be 'showdown' or 'engine', got {backend!r}")
    import asyncio

    return asyncio.run(_train_async(cfg))


class _TrainSetup:
    """Backend-agnostic training state shared by _train_async and _train_engine (one code path
    for the update tail — drift between backends is exactly the bug class the audits punished)."""

    def __init__(self, cfg: dict, *, scripted: tuple[str, ...] | None = None,
                 require_pool: bool = False):
        import time
        from pathlib import Path

        import torch

        from fog.models.agent import FogAgent
        from fog.models.probe import TrainableProbe
        from fog.training.rollout import RolloutBuffer

        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device(
            cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu"
        )

        self.ppo_cfg = cfg.get("ppo", {})
        self.env_cfg = cfg.get("env", {})
        self.fmt = self.env_cfg.get("battle_format", "gen9ou")
        team_pool_path = self.env_cfg.get("team_pool")
        self.pool_teams: list[str] | None = None
        if team_pool_path:
            from fog.env.teams import load_pool

            self.pool_teams = load_pool(team_pool_path)
        elif require_pool or "randombattle" not in self.fmt:
            raise ValueError(
                f"battle_format {self.fmt!r} needs teams: set env.team_pool to a paste directory "
                "(e.g. data/teams/pool_b) — validate it first with fog.env.teams.validate_pool."
                + (" The engine backend has no randombattle generator." if require_pool else "")
            )

        self.total = int(self.ppo_cfg.get("total_decisions", 100_000))
        self.batch_decisions = int(self.ppo_cfg.get("batch_decisions", 16_000))
        self.games_per_chunk = int(self.ppo_cfg.get("games_per_chunk", 8))
        self.lr = float(self.ppo_cfg.get("lr", 1e-4))
        self.leak_cfg = cfg.get("leakage", {})
        self.beta = float(self.leak_cfg.get("beta", 0.0))
        self.snap_every = int(cfg.get("league", {}).get("snapshot_every_decisions", 200_000))
        self.ckpt_dir = Path(cfg.get("log", {}).get("dir", "runs")) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.oracle_mode = str(cfg.get("model", {}).get("oracle", "none"))
        if self.oracle_mode == "action":
            raise NotImplementedError(
                "model.oracle=action cannot be trained in the online loop: the opponent's "
                "committed action is not observable at decision time (choices resolve "
                "simultaneously). The action-oracle ceiling needs an offline/replay pathway "
                "(spec §3.4); failing loudly here beats silently training an N-agent and "
                "reporting a fake ~0 oracle gap."
            )
        if self.oracle_mode == "set" and self.pool_teams is None:
            raise ValueError(
                "model.oracle=set requires env.team_pool — the truth tokens come from the "
                "sampled pool teams (recording teambuilder / attribute match)."
            )

        self.agent = FogAgent.from_tables(cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.agent.parameters(), lr=self.lr)
        self.league = League(
            snapshot_every_decisions=self.snap_every,
            **({"scripted": tuple(scripted)} if scripted else {}),
        )
        # leakage.clip_delta: True = per-step max(0, ΔI) (§7 default); False = signed ΔI, whose
        # per-game sum TELESCOPES to CE_first − CE_final — immune to the clipped estimator's
        # CE-noise bias (E[Σ max(0,ΔI)] grows with probe-CE variance even at zero net reveal).
        # Run at least one β cell with clip_delta=false as the estimator ablation.
        self.buf = RolloutBuffer(
            gamma=self.ppo_cfg.get("gamma", 1.0), lam=self.ppo_cfg.get("lam", 0.95),
            beta=self.beta, clip_deltas=bool(self.leak_cfg.get("clip_delta", True)),
        )
        self.dash = _Dashboard(cfg.get("log", {}).get("dir", "runs"))

        # pool truth (belief supervision fallback / oxygen check §7)
        self.pool_truth = None
        if self.pool_teams is not None:
            from fog.constants.tables import TABLES
            from fog.env.team_truth import PoolTruth

            self.pool_truth = PoolTruth(self.pool_teams, TABLES)
            if self.beta > 0:
                warning = self.pool_truth.hidden_info_warning()
                if warning and not self.leak_cfg.get("allow_unique_species_pool", False):
                    raise ValueError(
                        f"leakage run (beta={self.beta}) on {team_pool_path}: {warning} "
                        "Override with leakage.allow_unique_species_pool=true if this is "
                        "intentional."
                    )

        # BC anchor (KL penalty, §6): freeze a copy of the BC/initial policy for early self-play.
        self.bc_anchor = None
        self.kl_anchor_decisions = int(self.ppo_cfg.get("kl_anchor_decisions", 0))
        if self.ppo_cfg.get("bc_anchor_ckpt"):
            self.bc_anchor = FogAgent.from_tables(cfg).to(self.device)
            self.bc_anchor.load_state_dict(
                torch.load(self.ppo_cfg["bc_anchor_ckpt"], map_location=self.device)
            )
            self.bc_anchor.eval()

        # entropy decay + leakage Φ̂ refreeze schedule (§7)
        self.entropy0 = float(self.ppo_cfg.get("entropy_coef", 0.01))
        self.entropy_decay_to = float(self.ppo_cfg.get("entropy_coef_final", self.entropy0))
        self.refreeze_every = int(self.leak_cfg.get("refreeze_every_decisions", 100_000))
        self.watchdog = None
        if self.beta > 0:
            from fog.training.leakage import DegenerateSolutionWatchdog

            baseline_run = self.leak_cfg.get("baseline_run")
            if baseline_run:
                self.watchdog = DegenerateSolutionWatchdog.from_control_run(baseline_run)
                print(f"[ppo] watchdog baselines from control run: {baseline_run}", flush=True)
            else:
                self.watchdog = DegenerateSolutionWatchdog()
                print(
                    "[ppo] WARNING: leakage.baseline_run not set - the diversity and stalling "
                    "detectors are OFF (they need beta=0 control baselines; fabricated defaults "
                    "false-halted fresh runs at their first update). Only the stale-probe "
                    "detector runs. Run the beta=0 control first and point leakage.baseline_run "
                    "at its log dir.",
                    flush=True,
                )
            # stale-Φ̂ exploitation needs time-against-a-frozen-Φ̂ to exist at all; don't let
            # init-noise CE gaps halt the run (observed live: frozen−fresh ≈ 13.6 nats at update
            # one, no staleness).
            self.watchdog.warmup_decisions = int(
                self.leak_cfg.get("watchdog_warmup_decisions", 20_000)
            )

        # fresh probe (§7 staleness watchdog + honest readability tracking): an independent
        # reader trained on every rollout batch. On for any leakage run; optional otherwise.
        self.fresh_probe = None
        self.fresh_opt = None
        if self.beta > 0 or self.leak_cfg.get("fresh_probe", False):
            self.fresh_probe = TrainableProbe.from_agent(self.agent, cfg).to(self.device)
            self.fresh_opt = torch.optim.AdamW(
                self.fresh_probe.parameters(), lr=float(self.leak_cfg.get("fresh_probe_lr", 3e-4))
            )

        # update-loop counters (mutated by _maybe_chunk_update)
        self.decisions_seen = 0
        self.next_snapshot = self.snap_every
        self.next_refreeze = self.refreeze_every
        self.metrics: dict = {}
        self.chunk_t0 = time.monotonic()
        self.chunk_battles = 0


def _maybe_chunk_update(s: "_TrainSetup", phi_owner, recent_results: dict) -> str | None:
    """The shared update tail: batch -> schedules -> probes -> ppo_update -> dashboard ->
    watchdog -> refreeze -> snapshot. `phi_owner` is whatever holds `.phi_hat` for this backend
    (the FogPlayer adapter / the engine runner's helper). Returns "halt" on watchdog fire."""
    import time

    import torch

    from fog.models.probe import FrozenProbe

    if s.buf.n_decisions < min(s.batch_decisions, s.total - s.decisions_seen):
        return None

    frac = min(1.0, s.decisions_seen / max(1, s.total))
    step_cfg = dict(s.cfg)
    step_ppo = dict(s.ppo_cfg)
    step_ppo["entropy_coef"] = s.entropy0 + (s.entropy_decay_to - s.entropy0) * frac
    step_ppo["kl_anchor_coef"] = (
        s.ppo_cfg.get("kl_anchor_coef", 0.02)
        if (s.bc_anchor is not None and s.decisions_seen < s.kl_anchor_decisions)
        else 0.0
    )
    step_cfg["ppo"] = step_ppo

    n_games = s.buf.n_games
    batch = s.buf.to_batch(s.device)
    if s.bc_anchor is not None and step_ppo["kl_anchor_coef"] > 0:
        with torch.no_grad():
            batch["bc_log_probs"] = s.bc_anchor(batch).log_probs
    n = int(batch["actions"].shape[0])

    # fresh probe (§7): fit the independent reader on this batch's mirror views, then measure
    # fresh vs frozen readability CE for the staleness watchdog.
    probe_metrics = _fit_and_measure_probes(
        s.fresh_probe, s.fresh_opt, phi_owner.phi_hat, batch,
        steps=int(s.leak_cfg.get("fresh_probe_steps", 4)),
    )

    metrics = ppo_update(s.agent, s.opt, batch, step_cfg)
    metrics.update(probe_metrics)
    metrics.update(_belief_accuracy_vs_turn(s.agent, batch))
    s.buf.clear()
    s.decisions_seen += n

    all_recent = [r for d in recent_results.values() for r in d]
    win_rate = float(np.mean(all_recent)) if all_recent else 0.0
    dt = max(1e-6, time.monotonic() - s.chunk_t0)
    metrics.update(
        decisions=s.decisions_seen,
        win_rate=win_rate,
        avg_game_length=n / max(1, n_games),
        battles_per_sec=s.chunk_battles / dt,
        decisions_per_sec=n / dt,
        entropy_coef=step_ppo["entropy_coef"],
        move_entropy=metrics.get("entropy", 0.0),
    )
    for name, results in recent_results.items():
        if results:
            metrics[f"win_rate_vs_{name}"] = float(np.mean(results))
    s.chunk_t0, s.chunk_battles = time.monotonic(), 0
    s.dash.log(s.decisions_seen, metrics)
    print(
        f"[ppo] decisions={s.decisions_seen}  win_rate={win_rate:.3f}  "
        f"loss={metrics['loss']:.4f}  entropy={metrics['entropy']:.3f}  "
        f"kl={metrics['approx_kl']:.4f}  clip={metrics['clip_frac']:.3f}"
        + (f"  belief={metrics.get('belief_loss', 0):.3f}" if 'belief_loss' in metrics else "")
        + (f"  self_belief={metrics.get('self_belief_loss', 0):.3f}"
           if 'self_belief_loss' in metrics else ""),
        flush=True,
    )
    s.metrics = metrics

    if s.watchdog is not None:
        # baselines from the control run at MATCHED progress: comparing a fresh run's early
        # win-rate/entropy against the control's FINAL values false-fires both diversity
        # conditions on the first update of every L-run.
        s.watchdog.update_to(s.decisions_seen)
        flags = s.watchdog.check_all(metrics)
        if any(flags.values()):
            print(f"[ppo] WATCHDOG {flags} — HALTING for inspection (§7/§10).", flush=True)
            torch.save(s.agent.state_dict(), s.ckpt_dir / "ppo_watchdog_halt.pt")
            metrics["watchdog_halt"] = 1.0
            s.dash.log(s.decisions_seen, metrics)
            return "halt"

    if s.beta > 0 and s.decisions_seen >= s.next_refreeze:
        phi_owner.phi_hat = FrozenProbe.from_agent(s.agent).to(s.device)
        s.next_refreeze += s.refreeze_every

    if s.decisions_seen >= s.next_snapshot:
        path = s.ckpt_dir / f"ppo_{s.decisions_seen}.pt"
        torch.save(s.agent.state_dict(), path)
        s.league.past_ckpts.append(path)
        s.league.latest_ckpt = path
        s.next_snapshot += s.snap_every
    return "updated"


async def _train_async(cfg: dict) -> dict:
    from collections import deque
    from pathlib import Path

    import torch

    from fog.env.player import FogPlayer, unique_account
    from fog.models.agent import FogAgent
    from fog.models.probe import FrozenProbe

    try:
        from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("poke-env is required for self-play training.") from exc

    s = _TrainSetup(cfg)
    device = s.device
    agent, league, buf = s.agent, s.league, s.buf
    env_cfg, leak_cfg = s.env_cfg, s.leak_cfg
    fmt, pool_teams, beta, seed = s.fmt, s.pool_teams, s.beta, s.seed
    games_per_chunk, total = s.games_per_chunk, s.total
    concurrent = int(env_cfg.get("concurrent_battles", 1))

    def team_for(seed_offset: int):
        """Per-player pool sampler (teams re-sampled every battle, spec §6)."""
        if pool_teams is None:
            return None
        from fog.env.teams import make_pool_teambuilder

        return make_pool_teambuilder(pool_teams, seed=seed + seed_offset)

    adapter = FogPlayer(agent)
    adapter.attach_buffer(buf)
    adapter.leak_family_weights = leak_cfg.get("family_weights")
    if beta > 0:
        # Φ̂ = frozen (backbone + self-belief) pair — freezing only the head leaves the reward a
        # function of the live trunk (staleness/OOD hole; see models/probe.py).
        adapter.phi_hat = FrozenProbe.from_agent(agent).to(device)
    # Ground truth for belief supervision (§3.2) + the set oracle (§3.4). The EXACT source is the
    # recording teambuilder — the trainer sampled the opponent's team, so with sequential battles it
    # simply knows the paste. Species-multiset matching is only the concurrent-battles fallback: it
    # is permanently ambiguous on shared-species pools (POOL-A), which once silently disabled belief
    # supervision on exactly the pool where sets are hidden (audit 2026-07-05).
    current_opp_tb: dict[str, object] = {"tb": None}   # set per chunk in the rollout loop
    opp_teambuilders: dict[str, object] = {}
    if pool_teams is not None:
        from fog.constants.tables import TABLES
        from fog.env.team_truth import truth_tokens_from_paste

        pool_truth = s.pool_truth
        _paste_tokens: dict[str, "np.ndarray"] = {}

        def _truth(battle):
            tb = current_opp_tb.get("tb")
            paste = getattr(tb, "last_paste", None) if tb is not None else None
            if concurrent == 1 and paste is not None:
                tok = _paste_tokens.get(paste)
                if tok is None:
                    tok = _paste_tokens[paste] = truth_tokens_from_paste(paste, TABLES)
                return tok.copy()
            return pool_truth.match_battle(battle)   # attribute match; None until unambiguous

        adapter.truth_provider = _truth
        if s.oracle_mode == "set":
            adapter.oracle_truth_provider = _truth
    me = adapter.build(
        battle_format=fmt,
        max_concurrent_battles=concurrent,
        account_configuration=unique_account("fogppo"),
        team=team_for(1),
    )

    scripted_classes = {
        "Random": RandomPlayer,
        "MaxDamage": MaxBasePowerPlayer,
        "SimpleHeuristics": SimpleHeuristicsPlayer,
    }
    scripted_instances: dict[str, object] = {}
    ckpt_opponents: dict[str, object] = {}

    def get_opponent(spec) -> object:
        name = str(spec)
        if name in scripted_classes:
            if name not in scripted_instances:
                tb = team_for(2)
                opp_teambuilders[name] = tb
                scripted_instances[name] = scripted_classes[name](
                    battle_format=fmt,
                    max_concurrent_battles=concurrent,
                    account_configuration=unique_account("fogscr"),
                    team=tb,
                )
            return scripted_instances[name]
        if name not in ckpt_opponents:  # a checkpoint Path: frozen past self
            frozen = FogAgent.from_tables(cfg).to(device)
            frozen.load_state_dict(torch.load(spec, map_location=device))
            frozen.eval()
            tb = team_for(3)
            opp_teambuilders[name] = tb
            ckpt_opponents[name] = FogPlayer(frozen).build(
                battle_format=fmt,
                max_concurrent_battles=concurrent,
                account_configuration=unique_account("fogpast"),
                team=tb,
            )
        return ckpt_opponents[name]

    # per-opponent rolling win records (dashboard §6: win-rate vs each league member, NOT cumulative)
    recent_results: dict[str, deque] = {}
    while s.decisions_seen < total:
        opp_spec = league.sample_opponent()
        opp_name = str(opp_spec) if str(opp_spec) in scripted_classes else Path(str(opp_spec)).stem
        opponent = get_opponent(opp_spec)
        current_opp_tb["tb"] = opp_teambuilders.get(str(opp_spec))
        won0, fin0 = me.n_won_battles, me.n_finished_battles
        await me.battle_against(opponent, n_battles=games_per_chunk)
        dw, df = me.n_won_battles - won0, me.n_finished_battles - fin0
        rec = recent_results.setdefault(opp_name, deque(maxlen=200))
        rec.extend([1.0] * dw + [0.0] * (df - dw))
        s.chunk_battles += df
        # poke-env retains every finished battle in Player._battles (full per-turn event
        # observations, Pokemon objects) and n_won/n_finished are COMPUTED from it — nothing
        # evicts. Measured live 2026-07-10: ~2 GB commit growth per ~90k decisions, enough to
        # exhaust a 16 GB box mid-grid. Everything per-battle is already consumed by the time
        # battle_against returns (buffer end_game + adapter state eviction run in the finished
        # callback), and the win-rate accounting above re-baselines every chunk, so dropping the
        # objects here is safe. Failure is downgraded to a warning: a stuck reset must not kill
        # an hours-long run — it just resumes leaking.
        try:
            me.reset_battles()
            opponent.reset_battles()
        except Exception as exc:  # pragma: no cover - only on unfinished-battle races
            print(f"[ppo] WARNING: reset_battles failed ({exc}); battle objects retained "
                  "this chunk (memory will grow).", flush=True)

        if _maybe_chunk_update(s, adapter, recent_results) == "halt":
            return s.metrics

    final = s.ckpt_dir / "ppo_final.pt"
    torch.save(agent.state_dict(), final)
    print(f"[ppo] done: {s.decisions_seen} decisions -> {final}", flush=True)
    return s.metrics


def _train_engine(cfg: dict) -> dict:
    """Engine-backend training loop (SIMULATION_PLAN §4/§5): synchronous batched lockstep
    self-play on poke-engine, sharing the exact update tail with the Showdown path.

    Differences from _train_async, all deliberate:
      * no server / poke-env players — EngineRunner drives EngineBattles directly;
      * scripted league opponents are the engine ports (Random, MaxDamage). SimpleHeuristics is
        Showdown-only and raises here (eval it on Showdown; SIMULATION_PLAN §4);
      * belief/oracle truth is exact by construction (the runner sampled both pastes);
      * the cross-sim divergence gauntlet (§5) runs every `cross_sim.every_decisions` when
        enabled: engine-vs-Showdown win-rate gap > `cross_sim.halt_gap` (default 0.03) HALTS.
    """
    from collections import deque
    from pathlib import Path

    import torch

    from fog.env.engine_battle import HAVE_ENGINE
    from fog.env.engine_runner import EngineRunner, FrozenAgentPolicy
    from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES
    from fog.models.agent import FogAgent
    from fog.models.probe import FrozenProbe

    if not HAVE_ENGINE:
        raise RuntimeError("env.backend=engine requires poke-engine: pip install poke-engine")

    engine_scripted = tuple(
        cfg.get("league", {}).get("scripted", ("MaxDamage", "Random"))
    )
    unknown = [n for n in engine_scripted if n not in SCRIPTED_ENGINE_POLICIES]
    if unknown:
        raise ValueError(
            f"league.scripted {unknown} cannot run on the engine backend (engine ports exist "
            f"for {sorted(SCRIPTED_ENGINE_POLICIES)}). SimpleHeuristics and other poke-env "
            "players are Showdown-only: evaluate against them with env.backend=showdown."
        )

    s = _TrainSetup(cfg, scripted=engine_scripted, require_pool=True)
    total, games_per_chunk = s.total, s.games_per_chunk

    phi_hat = FrozenProbe.from_agent(s.agent).to(s.device) if s.beta > 0 else None
    runner = EngineRunner(
        s.agent,
        s.buf,
        pool=s.pool_teams,
        batch_size=int(s.env_cfg.get("engine_batch", 64)),
        seed=s.seed,
        phi_hat=phi_hat,
        leak_family_weights=s.leak_cfg.get("family_weights"),
        oracle_truth=(s.oracle_mode == "set"),
    )

    cross_cfg = cfg.get("cross_sim", {})
    cross_every = int(cross_cfg.get("every_decisions", 0))
    next_cross = cross_every if cross_every else None

    scripted_instances = {
        name: SCRIPTED_ENGINE_POLICIES[name](seed=s.seed + 100 + i)
        for i, name in enumerate(engine_scripted)
    }
    ckpt_opponents: dict[str, FrozenAgentPolicy] = {}

    def get_opponent(spec):
        name = str(spec)
        if name in scripted_instances:
            return scripted_instances[name]
        if name not in ckpt_opponents:  # a checkpoint Path: frozen past self
            frozen = FogAgent.from_tables(cfg).to(s.device)
            frozen.load_state_dict(torch.load(spec, map_location=s.device))
            ckpt_opponents[name] = FrozenAgentPolicy(frozen, name=Path(name).stem)
        return ckpt_opponents[name]

    recent_results: dict[str, deque] = {}
    while s.decisions_seen < total:
        opp_spec = s.league.sample_opponent()
        opp_name = str(opp_spec) if str(opp_spec) in scripted_instances else Path(str(opp_spec)).stem
        opponent = get_opponent(opp_spec)
        # collection runs with the agent in its live training mode — identical to the Showdown
        # path (an eval() toggle here would silently change the PPO ratio semantics per backend)
        stats = runner.play_games(opponent, games_per_chunk)
        rec = recent_results.setdefault(opp_name, deque(maxlen=200))
        rec.extend([1.0] * stats["wins"] + [0.0] * (stats["games"] - stats["wins"]))
        s.chunk_battles += stats["games"]

        if _maybe_chunk_update(s, runner._helper, recent_results) == "halt":
            return s.metrics

        if next_cross is not None and s.decisions_seen >= next_cross:
            from fog.eval.cross_sim import cross_sim_gauntlet

            report = cross_sim_gauntlet(s.agent, cfg)
            s.metrics.update({f"cross_sim_{k}": v for k, v in report.items()
                              if isinstance(v, (int, float))})
            s.dash.log(s.decisions_seen, s.metrics)
            gap = report.get("gap")
            halt_gap = float(cross_cfg.get("halt_gap", 0.03))
            if gap is not None and abs(gap) > halt_gap:
                print(
                    f"[ppo] CROSS-SIM HALT: engine-vs-Showdown win-rate gap {gap:+.3f} exceeds "
                    f"{halt_gap:.3f} — the policy is exploiting simulator divergence "
                    f"(SIMULATION_PLAN §5). Checkpoint saved for inspection.",
                    flush=True,
                )
                torch.save(s.agent.state_dict(), s.ckpt_dir / "ppo_cross_sim_halt.pt")
                s.metrics["cross_sim_halt"] = 1.0
                s.dash.log(s.decisions_seen, s.metrics)
                return s.metrics
            next_cross += cross_every

    final = s.ckpt_dir / "ppo_final.pt"
    torch.save(s.agent.state_dict(), final)
    print(f"[ppo] done (engine backend): {s.decisions_seen} decisions -> {final}", flush=True)
    return s.metrics


def _fit_and_measure_probes(fresh_probe, fresh_opt, phi_hat, batch: dict, steps: int = 4,
                            max_rows: int = 1024) -> dict:
    """Fit the fresh probe on this batch's mirror views + measure fresh vs frozen readability CE.

    Returns {} when there is no mirror data or no probe. `fresh_ce` < `frozen_ce` by a wide margin
    is the §7 staleness-exploitation signature (agent opaque to Φ̂ but readable to an up-to-date
    adversary) — fed to DegenerateSolutionWatchdog.check_stale_phi_exploitation.
    """
    import torch

    opp_view = batch.get("opp_view")
    if fresh_probe is None or not isinstance(opp_view, dict) or "my_view" not in opp_view:
        return {}
    n = opp_view["entities"].shape[0]
    idx = torch.randperm(n, device=opp_view["entities"].device)[: min(n, max_rows)]
    sub = _index_nested(opp_view, idx)

    out: dict[str, float] = {}
    fresh_probe.train()
    for _ in range(max(1, steps)):
        fresh_opt.zero_grad(set_to_none=True)
        loss = fresh_probe.loss(sub)
        loss.backward()
        fresh_opt.step()
    out["fresh_probe_loss"] = float(loss.detach())
    fresh_probe.eval()

    backbone_batch = {k: sub[k] for k in ("entities", "entity_ids", "field", "history", "active_idx")}
    with torch.no_grad():
        out["fresh_ce"] = float(fresh_probe.leakage_ce(backbone_batch, sub["my_view"]).mean())
        if phi_hat is not None:
            out["frozen_ce"] = float(phi_hat.leakage_ce(backbone_batch, sub["my_view"]).mean())
    return out


_TURN_BUCKETS = ((0, 5), (5, 10), (10, 15), (15, 1_000))


def _belief_accuracy_vs_turn(agent, batch: dict, max_rows: int = 512) -> dict:
    """Belief top-1 accuracy per turn bucket (§6 dashboard: MUST rise within games — flat means the
    history encoder is broken). Aggregated over all attribute families with valid labels."""
    import torch

    if "belief_targets" not in batch or "turn" not in batch or "opp_revealed" not in batch:
        return {}
    n = batch["turn"].shape[0]
    idx = torch.randperm(n, device=batch["turn"].device)[: min(n, max_rows)]
    sub = _index_nested(batch, idx)
    with torch.no_grad():
        out = agent(sub)
    if out.belief is None:
        return {}
    turns = sub["turn"]
    correct = torch.zeros_like(turns, dtype=torch.float32)
    counts = torch.zeros_like(turns, dtype=torch.float32)
    for fam, logits in out.belief.logits.items():
        tgt = sub["belief_targets"][fam]
        valid = sub["belief_valid"][fam] & (tgt >= 0)
        pred = logits.argmax(dim=-1)
        hit = ((pred == tgt) & valid).float()
        # collapse attribute dims -> per-row sums
        correct += hit.reshape(hit.shape[0], -1).sum(dim=1)
        counts += valid.float().reshape(valid.shape[0], -1).sum(dim=1)
    metrics: dict[str, float] = {}
    prev = None
    monotone = True
    for lo, hi in _TURN_BUCKETS:
        m = (turns >= lo) & (turns < hi) & (counts > 0)
        if m.any():
            acc = float((correct[m] / counts[m]).mean())
            metrics[f"belief_top1_t{lo}"] = acc
            if prev is not None and acc < prev - 0.02:
                monotone = False
            prev = acc
    if len(metrics) >= 2:
        metrics["belief_top1_monotone"] = 1.0 if monotone else 0.0
    return metrics


class _Dashboard:
    """CSV + (optional) tensorboard metric logger (§6 dashboard).

    The CSV is rewritten in full whenever a new metric key appears (e.g. the first game vs a new
    league opponent) — rows are cheap at update cadence and a fixed-from-first-row header silently
    drops late columns."""

    def __init__(self, log_dir: str) -> None:
        from pathlib import Path

        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "metrics.csv"
        self.keys: list[str] = []
        self.rows: list[tuple[int, dict]] = []
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.dir / "tb"))
        except Exception:
            self.tb = None

    def log(self, step: int, metrics: dict) -> None:
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.rows.append((step, scalars))
        new_keys = [k for k in scalars if k not in self.keys]
        if new_keys:
            self.keys.extend(new_keys)
            with open(self.csv, "w", encoding="utf-8") as f:
                f.write("step," + ",".join(self.keys) + "\n")
                for s, row in self.rows:
                    f.write(f"{s}," + ",".join(f"{row.get(k, 0.0):.5f}" for k in self.keys) + "\n")
        else:
            with open(self.csv, "a", encoding="utf-8") as f:
                f.write(f"{step}," + ",".join(f"{scalars.get(k, 0.0):.5f}" for k in self.keys) + "\n")
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)


def main() -> None:
    from fog.config import load_config_with_args

    ap = argparse.ArgumentParser(description="FOG self-play PPO (§6).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", help="dotted overrides, e.g. leakage.beta=0.03")
    args = ap.parse_args()
    train(load_config_with_args(args.config, args.overrides))


if __name__ == "__main__":
    main()
