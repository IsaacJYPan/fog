"""Training stage 2 — self-play PPO (spec §6).

PPO: clip 0.2, GAE λ=0.95, γ=1.0 (episodic win/loss ±1; optional +0.02 per opposing faint, off by
default), entropy bonus 0.01 decayed, value coef 0.5, batch 16k decision points, 3 epochs/batch,
lr 1e-4. Auxiliary belief/self-belief losses continue with FULL oracle labels (simulator provides both
teams). KL-to-BC-anchor penalty 0.02 for the first 2M decisions, annealed to 0.

Watch the §6 diagnostics dashboard from day one: win-rate vs each frozen league member; entropy per
turn-bucket; belief accuracy vs turn number (MUST be monotone increasing within games); value
calibration; tera-usage fraction; average game length.

GAE and `ppo_update` are real; `train` (the rollout <-> update loop) needs the env runner and is
specified in IMPLEMENTATION_PLAN.md.
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
                b_loss = belief_loss(out.belief.logits, mb["belief_targets"], mb["belief_valid"])
                loss = loss + belief_coef * b_loss
                log("belief_loss", b_loss)
            if out.self_belief is not None and mb.get("self_belief_targets") is not None:
                sb_loss = belief_loss(
                    out.self_belief.logits, mb["self_belief_targets"], mb["self_belief_valid"]
                )
                loss = loss + self_belief_coef * sb_loss
                log("self_belief_loss", sb_loss)

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
    """Self-play PPO loop (§6): FogPlayer collects into a RolloutBuffer against league opponents;
    ppo_update consumes it; checkpoints snapshot into the league. Returns the last metrics dict.

    Requires: built constant tables, a running local Showdown server, poke-env installed.
    Config keys used (all under sensible defaults): env.battle_format, ppo.{lr, total_decisions,
    batch_decisions, games_per_chunk, ...ppo_update keys}, league.snapshot_every_decisions,
    leakage.beta (0 = N-agent), log.dir.
    """
    import asyncio

    return asyncio.run(_train_async(cfg))


async def _train_async(cfg: dict) -> dict:
    import time
    from collections import deque
    from pathlib import Path

    import torch

    from fog.env.player import FogPlayer, unique_account
    from fog.models.agent import FogAgent
    from fog.models.probe import FrozenProbe, TrainableProbe
    from fog.training.rollout import RolloutBuffer

    try:
        from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("poke-env is required for self-play training.") from exc

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(
        cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu"
    )

    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})
    fmt = env_cfg.get("battle_format", "gen9ou")
    team_pool_path = env_cfg.get("team_pool")
    pool_teams: list[str] | None = None
    if team_pool_path:
        from fog.env.teams import load_pool

        pool_teams = load_pool(team_pool_path)
    elif "randombattle" not in fmt:
        raise ValueError(
            f"battle_format {fmt!r} needs teams: set env.team_pool to a paste directory "
            "(e.g. data/teams/pool_b) — validate it first with fog.env.teams.validate_pool."
        )

    def team_for(seed_offset: int):
        """Per-player pool sampler (teams re-sampled every battle, spec §6)."""
        if pool_teams is None:
            return None
        from fog.env.teams import make_pool_teambuilder

        return make_pool_teambuilder(pool_teams, seed=seed + seed_offset)
    total = int(ppo_cfg.get("total_decisions", 100_000))
    batch_decisions = int(ppo_cfg.get("batch_decisions", 16_000))
    games_per_chunk = int(ppo_cfg.get("games_per_chunk", 8))
    lr = float(ppo_cfg.get("lr", 1e-4))
    beta = float(cfg.get("leakage", {}).get("beta", 0.0))
    snap_every = int(cfg.get("league", {}).get("snapshot_every_decisions", 200_000))
    ckpt_dir = Path(cfg.get("log", {}).get("dir", "runs")) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent = FogAgent.from_tables(cfg).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=lr)
    league = League(snapshot_every_decisions=snap_every)
    buf = RolloutBuffer(gamma=ppo_cfg.get("gamma", 1.0), lam=ppo_cfg.get("lam", 0.95), beta=beta)
    dash = _Dashboard(cfg.get("log", {}).get("dir", "runs"))

    # BC anchor (KL penalty, §6): freeze a copy of the BC/initial policy to anchor early self-play.
    bc_anchor = None
    kl_anchor_decisions = int(ppo_cfg.get("kl_anchor_decisions", 0))
    if cfg.get("ppo", {}).get("bc_anchor_ckpt"):
        bc_anchor = FogAgent.from_tables(cfg).to(device)
        bc_anchor.load_state_dict(torch.load(cfg["ppo"]["bc_anchor_ckpt"], map_location=device))
        bc_anchor.eval()

    # entropy decay + leakage Φ̂ refreeze schedule (§7)
    entropy0 = float(ppo_cfg.get("entropy_coef", 0.01))
    entropy_decay_to = float(ppo_cfg.get("entropy_coef_final", entropy0))
    leak_cfg = cfg.get("leakage", {})
    refreeze_every = int(leak_cfg.get("refreeze_every_decisions", 100_000))
    watchdog = None
    if beta > 0:
        from fog.training.leakage import DegenerateSolutionWatchdog

        baseline_run = leak_cfg.get("baseline_run")
        if baseline_run:
            watchdog = DegenerateSolutionWatchdog.from_control_run(baseline_run)
            print(f"[ppo] watchdog baselines from control run: {baseline_run}", flush=True)
        else:
            watchdog = DegenerateSolutionWatchdog()
            print(
                "[ppo] WARNING: leakage.baseline_run not set — watchdog runs on default baselines "
                "(stalling detector disabled, diversity detector weak). Run the beta=0 control "
                "first and point leakage.baseline_run at its log dir.",
                flush=True,
            )

    # fresh probe (§7 staleness watchdog + honest readability tracking): an independent reader
    # trained on every rollout batch. On for any leakage run; optional otherwise.
    fresh_probe = None
    fresh_opt = None
    if beta > 0 or leak_cfg.get("fresh_probe", False):
        fresh_probe = TrainableProbe.from_agent(agent, cfg).to(device)
        fresh_opt = torch.optim.AdamW(fresh_probe.parameters(), lr=float(leak_cfg.get("fresh_probe_lr", 3e-4)))

    adapter = FogPlayer(agent)
    adapter.attach_buffer(buf)
    adapter.leak_family_weights = leak_cfg.get("family_weights")
    if beta > 0:
        # Φ̂ = frozen (backbone + self-belief) pair — freezing only the head leaves the reward a
        # function of the live trunk (staleness/OOD hole; see models/probe.py).
        adapter.phi_hat = FrozenProbe.from_agent(agent).to(device)
    # self-play belief supervision: match the finished opponent to its pool team for full-truth labels
    if pool_teams is not None and cfg.get("loss", {}).get("belief_coef", 1.0) > 0:
        from fog.constants.tables import TABLES
        from fog.env.team_truth import PoolTruth, revealed_opponent_species

        pool_truth = PoolTruth(pool_teams, TABLES)

        def _truth(battle):
            return pool_truth.match(revealed_opponent_species(battle, TABLES))

        adapter.truth_provider = _truth
    me = adapter.build(
        battle_format=fmt,
        max_concurrent_battles=1,
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
                scripted_instances[name] = scripted_classes[name](
                    battle_format=fmt,
                    account_configuration=unique_account("fogscr"),
                    team=team_for(2),
                )
            return scripted_instances[name]
        if name not in ckpt_opponents:  # a checkpoint Path: frozen past self
            frozen = FogAgent.from_tables(cfg).to(device)
            frozen.load_state_dict(torch.load(spec, map_location=device))
            frozen.eval()
            ckpt_opponents[name] = FogPlayer(frozen).build(
                battle_format=fmt,
                max_concurrent_battles=1,
                account_configuration=unique_account("fogpast"),
                team=team_for(3),
            )
        return ckpt_opponents[name]

    decisions_seen = 0
    next_snapshot = snap_every
    next_refreeze = refreeze_every
    metrics: dict = {}
    # per-opponent rolling win records (dashboard §6: win-rate vs each league member, NOT cumulative)
    recent_results: dict[str, deque] = {}
    chunk_t0 = time.monotonic()
    chunk_battles = 0
    while decisions_seen < total:
        opp_spec = league.sample_opponent()
        opp_name = str(opp_spec) if str(opp_spec) in scripted_classes else Path(str(opp_spec)).stem
        opponent = get_opponent(opp_spec)
        won0, fin0 = me.n_won_battles, me.n_finished_battles
        await me.battle_against(opponent, n_battles=games_per_chunk)
        dw, df = me.n_won_battles - won0, me.n_finished_battles - fin0
        rec = recent_results.setdefault(opp_name, deque(maxlen=200))
        rec.extend([1.0] * dw + [0.0] * (df - dw))
        chunk_battles += df

        if buf.n_decisions >= min(batch_decisions, total - decisions_seen):
            # per-update schedules
            frac = min(1.0, decisions_seen / max(1, total))
            step_cfg = dict(cfg)
            step_ppo = dict(ppo_cfg)
            step_ppo["entropy_coef"] = entropy0 + (entropy_decay_to - entropy0) * frac
            step_ppo["kl_anchor_coef"] = (
                ppo_cfg.get("kl_anchor_coef", 0.02)
                if (bc_anchor is not None and decisions_seen < kl_anchor_decisions)
                else 0.0
            )
            step_cfg["ppo"] = step_ppo

            n_games = buf.n_games
            batch = buf.to_batch(device)
            if bc_anchor is not None and step_ppo["kl_anchor_coef"] > 0:
                with torch.no_grad():
                    batch["bc_log_probs"] = bc_anchor(batch).log_probs
            n = int(batch["actions"].shape[0])

            # fresh probe (§7): fit the independent reader on this batch's mirror views, then
            # measure fresh vs frozen readability CE for the staleness watchdog.
            probe_metrics = _fit_and_measure_probes(
                fresh_probe, fresh_opt, adapter.phi_hat, batch,
                steps=int(leak_cfg.get("fresh_probe_steps", 4)),
            )

            metrics = ppo_update(agent, opt, batch, step_cfg)
            metrics.update(probe_metrics)
            metrics.update(_belief_accuracy_vs_turn(agent, batch))
            buf.clear()
            decisions_seen += n

            all_recent = [r for d in recent_results.values() for r in d]
            win_rate = float(np.mean(all_recent)) if all_recent else 0.0
            dt = max(1e-6, time.monotonic() - chunk_t0)
            metrics.update(
                decisions=decisions_seen,
                win_rate=win_rate,
                avg_game_length=n / max(1, n_games),
                battles_per_sec=chunk_battles / dt,
                decisions_per_sec=n / dt,
                entropy_coef=step_ppo["entropy_coef"],
                move_entropy=metrics.get("entropy", 0.0),
            )
            for name, results in recent_results.items():
                if results:
                    metrics[f"win_rate_vs_{name}"] = float(np.mean(results))
            chunk_t0, chunk_battles = time.monotonic(), 0
            dash.log(decisions_seen, metrics)
            print(
                f"[ppo] decisions={decisions_seen}  win_rate={win_rate:.3f}  "
                f"loss={metrics['loss']:.4f}  entropy={metrics['entropy']:.3f}  "
                f"kl={metrics['approx_kl']:.4f}  clip={metrics['clip_frac']:.3f}"
                + (f"  belief={metrics.get('belief_loss', 0):.3f}" if 'belief_loss' in metrics else "")
                + (f"  self_belief={metrics.get('self_belief_loss', 0):.3f}"
                   if 'self_belief_loss' in metrics else ""),
                flush=True,
            )
            if watchdog is not None:
                flags = watchdog.check_all(metrics)
                if any(flags.values()):
                    print(f"[ppo] WATCHDOG {flags} — HALTING for inspection (§7/§10).", flush=True)
                    torch.save(agent.state_dict(), ckpt_dir / "ppo_watchdog_halt.pt")
                    metrics["watchdog_halt"] = 1.0
                    dash.log(decisions_seen, metrics)
                    return metrics

            if beta > 0 and decisions_seen >= next_refreeze:
                adapter.phi_hat = FrozenProbe.from_agent(agent).to(device)
                next_refreeze += refreeze_every

            if decisions_seen >= next_snapshot:
                path = ckpt_dir / f"ppo_{decisions_seen}.pt"
                torch.save(agent.state_dict(), path)
                league.past_ckpts.append(path)
                league.latest_ckpt = path
                next_snapshot += snap_every

    final = ckpt_dir / "ppo_final.pt"
    torch.save(agent.state_dict(), final)
    print(f"[ppo] done: {decisions_seen} decisions -> {final}", flush=True)
    return metrics


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
