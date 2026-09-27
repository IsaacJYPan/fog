"""Self-play PPO training with auxiliary belief losses.

`train` samples league opponents, collects live battles into a RolloutBuffer,
computes GAE, updates the model, and saves league snapshots and checkpoints.
Belief supervision uses simulator team labels. Optional leakage shaping runs
when leakage.beta > 0; configuration controls PPO and auxiliary-loss weights."""
from __future__ import annotations

import argparse
import math
from functools import wraps

import numpy as np
import torch

from fog.training.dashboard import TrainingDashboard as _Dashboard
from fog.training.gae import compute_gae
from fog.training.league import League

__all__ = ["compute_gae", "ppo_update", "train"]


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


def _chain_probability_metrics(output, attr_valid: dict) -> dict[str, float]:
    """Calibration of the randomly addressed replay query, including the no-uncertainty cut.

    ``last`` means every other recoverable hidden fact on that row was supplied as a generated
    assumption. With only one target left, its true-value probability should approach one.
    """
    if output.loss_valid is None:
        return {}
    targets = output.loss_targets
    if targets is None:
        return {}
    all_prob: list[torch.Tensor] = []
    last_prob: list[torch.Tensor] = []
    is_last = None
    if output.context_count is not None and output.route_length is not None:
        is_last = output.context_count == (output.route_length - 1)
    for family, logits in output.logits.items():
        target = targets[family]
        valid = attr_valid[family] & output.loss_valid[family]
        safe = target.clamp(min=0, max=logits.shape[-1] - 1)
        probability = torch.softmax(logits, dim=-1).gather(
            -1, safe.unsqueeze(-1)
        ).squeeze(-1)
        finite = torch.isfinite(probability)
        use = valid & finite
        if bool(use.any()):
            all_prob.append(probability[use])
        if is_last is not None:
            row_mask = is_last
            while row_mask.ndim < use.ndim:
                row_mask = row_mask.unsqueeze(-1)
            last_use = use & row_mask
            if bool(last_use.any()):
                last_prob.append(probability[last_use])

    result: dict[str, float] = {}
    if all_prob:
        values = torch.cat(all_prob).detach()
        result["belief_chain_true_prob"] = float(values.mean())
        result["belief_chain_confident98_rate"] = float((values >= 0.98).float().mean())
    if last_prob:
        values = torch.cat(last_prob).detach()
        result["belief_chain_last_true_prob"] = float(values.mean())
        result["belief_chain_last_confident98_rate"] = float(
            (values >= 0.98).float().mean()
        )
        result["belief_chain_last_count"] = float(values.numel())
    return result


def _deterministic_policy_update(fn):
    """Disable dropout, not gradients; restore the caller's per-module modes on exit."""
    @wraps(fn)
    def wrapped(agent, *args, **kwargs):
        modes = [(module, module.training) for module in agent.modules()]
        agent.eval()
        try:
            return fn(agent, *args, **kwargs)
        finally:
            for module, training in modes:
                module.training = training
    return wrapped


@_deterministic_policy_update
def ppo_update(agent, optimizer, batch: dict, cfg: dict, replay_regularizer=None) -> dict:
    """One PPO update over a collected batch (spec §6): 3 epochs of minibatches, clip 0.2.

    Loss = -clip_surrogate + value_coef * value_MSE - entropy_coef * entropy
           + belief_coef * belief_CE + self_belief_coef * self_belief_CE   (full oracle labels)
           + kl_anchor_coef * KL(pi || pi_BC)

    `batch` holds the stacked FogAgent obs tensors (entities, entity_ids, field, history, active_idx,
    legality_mask, optional opp_revealed / opp_view — see models/agent.py) plus, per decision point:
      actions [N] long, old_log_probs [N], advantages [N], returns [N], step_rewards [N],
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
    # The belief-chain CE has its own coefficient, initially equal to the marginal's.
    belief_chain_coef = loss_cfg.get("belief_chain_coef", belief_coef)
    self_belief_chain_coef = loss_cfg.get("self_belief_chain_coef", self_belief_coef)
    preview_coef = loss_cfg.get("preview_coef", 0.5)
    kl_anchor_coef = ppo_cfg.get("kl_anchor_coef", 0.0)
    max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
    target_kl = ppo_cfg.get("target_kl")
    if target_kl is not None and (not math.isfinite(float(target_kl)) or target_kl <= 0):
        raise ValueError("ppo.target_kl must be finite and positive")
    rollout_temperature = float(ppo_cfg.get("rollout_temperature", 1.0))
    if not np.isfinite(rollout_temperature) or rollout_temperature <= 0:
        raise ValueError("ppo.rollout_temperature must be finite and positive")
    rollout_tera_logit_bonus = float(ppo_cfg.get("rollout_tera_logit_bonus", 0.0))
    if not np.isfinite(rollout_tera_logit_bonus) or rollout_tera_logit_bonus < 0:
        raise ValueError("ppo.rollout_tera_logit_bonus must be finite and non-negative")

    from fog.env.player import _behavior_log_probs, _temperature_log_probs

    # preview rows are per-GAME, not per-decision — handled outside the minibatch permutation
    preview = batch.get("preview")
    batch = {key: value for key, value in batch.items() if key != "preview"}

    n = batch["actions"].shape[0]
    device = batch["actions"].device
    adv = batch["advantages"]
    if n == 0:
        raise ValueError("cannot update PPO from an empty batch")
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    from fog.training.losses import (  # local import: avoid cycles at module load
        belief_loss,
        belief_output_loss,
        value_loss,
    )

    metrics: dict[str, list[float]] = {}
    logged_drop_examples: set[str] = set()
    if replay_regularizer is not None:
        replay_regularizer.begin_update()

    def log(name: str, v: torch.Tensor | float) -> None:
        if isinstance(v, torch.Tensor):
            v = v.detach().item()
        metrics.setdefault(name, []).append(float(v))

    step_rewards = batch.get("step_rewards")
    if step_rewards is not None:
        if not torch.isfinite(step_rewards).all():
            raise RuntimeError("non-finite per-decision rewards")
        log("reward_mean", step_rewards.mean())
        log("reward_std", step_rewards.std(unbiased=False))
        log("reward_min", step_rewards.min())
        log("reward_max", step_rewards.max())
        log("reward_positive_rate", (step_rewards > 0).float().mean())
        log("reward_negative_rate", (step_rewards < 0).float().mean())
        log("reward_nonzero_rate", (step_rewards != 0).float().mean())

    actions = batch["actions"]
    raw_advantages = batch["advantages"]
    game_outcomes = batch.get("game_outcomes")
    log("rollout_temperature", rollout_temperature)
    log("rollout_tera_logit_bonus", rollout_tera_logit_bonus)
    for family, selected in {
        "ordinary_move": (actions >= 0) & (actions < 4),
        "tera_move": (actions >= 4) & (actions < 8),
        "switch": (actions >= 8) & (actions < 13),
        "forced": actions == 13,
    }.items():
        log(f"selected_{family}_rate", selected.float().mean())
        if bool(selected.any()):
            log(f"selected_{family}_advantage_mean", raw_advantages[selected].mean())
            if isinstance(game_outcomes, torch.Tensor):
                log(
                    f"selected_{family}_game_win_rate",
                    (game_outcomes[selected] > 0).float().mean(),
                )
    legality = batch.get("legality_mask")
    if isinstance(legality, torch.Tensor) and legality.ndim == 2 and legality.shape[1] >= 14:
        tera_legal = legality[:, 4:8].any(dim=-1)
        log("tera_legal_decision_rate", tera_legal.float().mean())
        if bool(tera_legal.any()):
            selected_tera = (actions >= 4) & (actions < 8)
            log("tera_selected_when_legal_rate", selected_tera[tera_legal].float().mean())

    def log_drop_examples(prefix: str, stats: dict, view: dict | None) -> None:
        if not isinstance(view, dict):
            return
        tokens = view.get("tokens")
        masks = view.get("masks") or {}
        for family, family_stats in stats.get("by_family", {}).items():
            examples = family_stats.get("dropped_examples") or []
            key = f"{prefix}:{family}"
            if not examples or key in logged_drop_examples:
                continue
            logged_drop_examples.add(key)
            fields_per_slot = 4 if family == "move" else 6 if family == "ev" else 1
            decoded = []
            for example in examples:
                flat_index = int(example["flat_index"])
                row, rem = divmod(flat_index, 6 * fields_per_slot)
                slot, field = divmod(rem, fields_per_slot)
                target = int(example["target"])
                static_mask = masks.get(family)
                mask_allows = None
                if static_mask is not None and target < static_mask.shape[-1]:
                    mask_allows = bool(static_mask[row, slot, target])
                slot_tokens = (
                    tokens[row, slot].detach().cpu().tolist()
                    if isinstance(tokens, torch.Tensor)
                    else None
                )
                decoded.append(
                    {
                        "row": row,
                        "slot": slot,
                        "field": field,
                        "target": target,
                        "static_mask_allows": mask_allows,
                        "tokens": slot_tokens,
                    }
                )
            print(f"[ppo-drop-debug] {key} {decoded}", flush=True)

    stopped_for_kl = False
    optimizer_steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            mb = _index_nested(batch, idx)
            # Evaluation mode is the behavior policy. Chain supervision remains explicit.
            mb["compute_belief_chain"] = True
            mb_adv = adv[idx]

            out = agent(mb)
            behavior_log_probs = _behavior_log_probs(
                out.log_probs,
                temperature=rollout_temperature,
                tera_logit_bonus=rollout_tera_logit_bonus,
            )
            new_log_probs = behavior_log_probs.gather(
                1, mb["actions"].unsqueeze(1)
            ).squeeze(1)

            ratio = (new_log_probs - mb["old_log_probs"]).exp()
            log_ratio = new_log_probs - mb["old_log_probs"]
            sampled_kl = (torch.expm1(log_ratio) - log_ratio).mean()
            old_distribution = mb.get("behavior_log_probs")
            policy_kl = (
                _masked_kl(old_distribution, behavior_log_probs).mean()
                if old_distribution is not None else sampled_kl
            )
            if not torch.isfinite(policy_kl):
                raise FloatingPointError("nonfinite PPO behavior-policy KL")
            log("policy_kl", policy_kl)
            log("sampled_kl", sampled_kl)
            if target_kl is not None and float(policy_kl.detach()) > float(target_kl):
                stopped_for_kl = True
                break
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_loss = value_loss(out.value, mb["returns"])
            entropy = _masked_entropy(behavior_log_probs).mean()

            loss = policy_loss + value_coef * v_loss - entropy_coef * entropy

            if kl_anchor_coef > 0 and mb.get("bc_log_probs") is not None:
                kl_bc = _masked_kl(out.log_probs, mb["bc_log_probs"]).mean()
                loss = loss + kl_anchor_coef * kl_bc
                log("kl_bc", kl_bc)

            if out.belief is not None and mb.get("belief_targets") is not None:
                b_stats: dict = {}
                # Marginal CE: the policy no longer reads this
                # (b_vec is gone) — it keeps the on-demand canonical marginal forward calibrated for
                # the consumers that still read it: pool_id's fast path, the belief dashboard, and the
                # leakage probe (which reads a marginal, not the chain). Set belief_coef=0 for a
                # pure-chain ablation. Like the chain term it conditions on belief_context, so with
                # aux_into_trunk it also shapes the shared trunk.
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
                log_drop_examples("belief", b_stats, mb.get("opp_revealed"))
                for family, family_stats in b_stats.get("by_family", {}).items():
                    family_total = (
                        family_stats.get("valid", 0)
                        + family_stats.get("dropped_nonfinite", 0)
                    )
                    log(
                        f"belief_dropped_rate_{family}",
                        family_stats.get("dropped_nonfinite", 0) / max(1, family_total),
                    )
                # Chain CE (§3.6): THE primary belief aux loss — the input pattern sampling/scoring
                # query, and (with aux_into_trunk) the gradient that makes the contexts informative
                # for the policy. Separate coefficient + metric.
                if out.belief_chain is not None:
                    bc_loss = belief_output_loss(
                        out.belief_chain, mb["belief_targets"], mb["belief_valid"]
                    )
                    loss = loss + belief_chain_coef * bc_loss
                    log("belief_ce_chain", bc_loss)
                    for metric_name, metric_value in _chain_probability_metrics(
                        out.belief_chain, mb["belief_valid"]
                    ).items():
                        log(metric_name, metric_value)
            if out.self_belief is not None and mb.get("self_belief_targets") is not None:
                sb_stats: dict = {}
                # Marginal self-belief CE — keeps the leakage probe's marginal forward calibrated
                # (the β arm reads it, §8.2). self_belief_coef=0 for a pure-chain ablation.
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
                log_drop_examples(
                    "self_belief", sb_stats, (mb.get("opp_view") or {}).get("my_view")
                )
                for family, family_stats in sb_stats.get("by_family", {}).items():
                    family_total = (
                        family_stats.get("valid", 0)
                        + family_stats.get("dropped_nonfinite", 0)
                    )
                    log(
                        f"self_belief_dropped_rate_{family}",
                        family_stats.get("dropped_nonfinite", 0) / max(1, family_total),
                    )
                if out.self_belief_chain is not None:
                    sbc_loss = belief_output_loss(
                        out.self_belief_chain, mb["self_belief_targets"],
                        mb["self_belief_valid"],
                    )
                    loss = loss + self_belief_chain_coef * sbc_loss
                    log("self_belief_ce_chain", sbc_loss)

            if replay_regularizer is not None:
                replay_term = replay_regularizer.loss(agent)
                if replay_term is not None:
                    replay_loss, replay_metrics = replay_term
                    loss = loss + replay_loss
                    for replay_name, replay_value in replay_metrics.items():
                        log(replay_name, replay_value)
                    if ppo_cfg.get("gradient_diagnostics", False):
                        from fog.training.diagnostics import gradient_comparison

                        replay_nll = replay_regularizer.last_policy_loss
                        if replay_nll is not None:
                            for key, value in gradient_comparison(
                                policy_loss, replay_nll, agent.parameters(), replay_regularizer.coef
                            ).items():
                                log(key, value)
                    replay_regularizer.last_policy_loss = None

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss; optimizer step refused")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                agent.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            log("grad_norm", grad_norm)
            optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                log("loss", loss)
                log("policy_loss", policy_loss)
                log("value_loss", v_loss)
                log("entropy", entropy)
                log("approx_kl", (mb["old_log_probs"] - new_log_probs).mean())
                log("clip_frac", ((ratio - 1.0).abs() > clip).float().mean())

        if stopped_for_kl:
            break
        # team-preview bandit (§3.1): one clipped-surrogate step per epoch over all preview rows.
        # Reward = game outcome; baseline = batch mean outcome (variance reduction).
        if preview is not None and preview_coef > 0:
            logits = agent.preview_logits(preview)                                # [G, 6]
            preview_log_probs = _temperature_log_probs(
                torch.log_softmax(logits, dim=-1), temperature=rollout_temperature
            )
            lp = preview_log_probs.gather(
                1, preview["slot"].unsqueeze(1)
            ).squeeze(1)
            pv_adv = preview["outcome"] - preview["outcome"].mean()
            pv_ratio = (lp - preview["old_log_prob"]).exp()
            pv_s1 = pv_ratio * pv_adv
            pv_s2 = torch.clamp(pv_ratio, 1.0 - clip, 1.0 + clip) * pv_adv
            pv_loss = -torch.min(pv_s1, pv_s2).mean() * preview_coef
            if not torch.isfinite(pv_loss):
                raise FloatingPointError("nonfinite PPO preview loss")
            optimizer.zero_grad(set_to_none=True)
            pv_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            log("preview_loss", pv_loss)

    log("stopped_for_kl", float(stopped_for_kl))
    log("optimizer_steps", optimizer_steps)
    return {k: sum(v) / len(v) for k, v in metrics.items()}


def _restricted_optimizer_groups(
    agent,
    trainable_prefixes: tuple[str, ...],
    lr_scales: dict[str, float],
    base_lr: float,
) -> tuple[list[dict], list[torch.nn.Parameter], dict[float, list[str]]]:
    """Freeze unselected tensors and assign deterministic per-prefix learning-rate scales."""

    if not trainable_prefixes:
        if lr_scales:
            raise ValueError("ppo.trainable_lr_scales requires trainable_prefixes")
        trainable = list(agent.parameters())
        return [{"params": trainable, "lr": base_lr}], trainable, {1.0: ["*"]}
    for prefix, scale in lr_scales.items():
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("ppo.trainable_lr_scales keys must be non-empty prefixes")
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError(
                f"ppo.trainable_lr_scales[{prefix!r}] must be finite and positive"
            )
    grouped: dict[float, list[torch.nn.Parameter]] = {}
    group_names: dict[float, list[str]] = {}
    matched_scale_prefixes: set[str] = set()
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in agent.named_parameters():
        selected = name.startswith(trainable_prefixes)
        parameter.requires_grad_(selected)
        if not selected:
            continue
        trainable.append(parameter)
        matching = [prefix for prefix in lr_scales if name.startswith(prefix)]
        scale_prefix = max(matching, key=len) if matching else None
        scale = float(lr_scales[scale_prefix]) if scale_prefix is not None else 1.0
        if scale_prefix is not None:
            matched_scale_prefixes.add(scale_prefix)
        grouped.setdefault(scale, []).append(parameter)
        group_names.setdefault(scale, []).append(name)
    if not trainable:
        raise ValueError(
            f"ppo.trainable_prefixes matched no parameters: {trainable_prefixes}"
        )
    unmatched = sorted(set(lr_scales) - matched_scale_prefixes)
    if unmatched:
        raise ValueError(
            f"ppo.trainable_lr_scales prefixes matched no trainable parameters: {unmatched}"
        )
    groups = [
        {"params": grouped[scale], "lr": base_lr * scale}
        for scale in sorted(grouped, reverse=True)
    ]
    return groups, trainable, group_names


def train(cfg: dict) -> dict:
    """Self-play PPO loop (§6): rollouts collect into a RolloutBuffer against league opponents;
    ppo_update consumes it; checkpoints snapshot into the league. Returns the last metrics dict.

    `env.backend` selects the simulator (ARCHITECTURE.md §3.2):
      * "showdown_direct" (project-config default) — Showdown rules and the poke-env parser through
        a persistent local Node worker. No server, accounts, HTTP, or WebSockets.
      * "showdown" — poke-env vs a live local server, retained for external/live compatibility.
      * "engine"             — poke-engine batched lockstep self-play (fog.env.engine_runner);
        volume track for M3/M4 grids. No server needed. Final fine-tune + all headline evals
        stay on Showdown (divergence rules in ARCHITECTURE.md §3.2).

    Config keys used (all under sensible defaults): env.{battle_format, team_pool, backend,
    engine_batch}, ppo.{lr, total_decisions, batch_decisions, games_per_chunk, ...ppo_update keys},
    league.snapshot_every_decisions, leakage.beta (0 = N-agent), log.dir.
    """
    from fog.training.preflight import require_preflight

    require_preflight(cfg)
    backend = str(cfg.get("env", {}).get("backend", "showdown")).lower()
    if backend == "engine":
        return _train_engine(cfg)
    if backend == "showdown_direct":
        return _train_showdown_direct(cfg)
    if backend != "showdown":
        raise ValueError(
            "env.backend must be 'showdown', 'showdown_direct', or 'engine', "
            f"got {backend!r}"
        )
    import asyncio

    return asyncio.run(_train_async(cfg))


class _TrainSetup:
    """Backend-agnostic training state shared by _train_async and _train_engine (one code path
    for the update tail — drift between backends is exactly the bug class the audits punished)."""

    def __init__(self, cfg: dict, *, scripted: tuple[str, ...] | None = None,
                 require_pool: bool = False):
        import random
        import time
        from pathlib import Path

        import torch

        from fog.models.agent import FogAgent
        from fog.models.probe import TrainableProbe
        from fog.training.rollout import RolloutBuffer

        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0))
        random.seed(self.seed)
        self.rng = random.Random(self.seed)
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
            minimum_pool_size = int(self.env_cfg.get("min_team_pool_size", 1))
            if len(self.pool_teams) < minimum_pool_size:
                raise ValueError(
                    f"team pool {team_pool_path} has {len(self.pool_teams)} teams; "
                    f"this run requires at least {minimum_pool_size}"
                )
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
        self.rollout_temperature = float(self.ppo_cfg.get("rollout_temperature", 1.0))
        if not np.isfinite(self.rollout_temperature) or self.rollout_temperature <= 0:
            raise ValueError("ppo.rollout_temperature must be finite and positive")
        self.rollout_tera_logit_bonus = float(
            self.ppo_cfg.get("rollout_tera_logit_bonus", 0.0)
        )
        if (
            not np.isfinite(self.rollout_tera_logit_bonus)
            or self.rollout_tera_logit_bonus < 0
        ):
            raise ValueError("ppo.rollout_tera_logit_bonus must be finite and non-negative")
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

        # Single-worker collection uses this model directly. Autograd is independent of mode.
        self.agent = FogAgent.from_tables(cfg).to(self.device).eval()
        if self.ppo_cfg.get("init_ckpt") and self.ppo_cfg.get("resume_ckpt"):
            raise ValueError("ppo.init_ckpt and ppo.resume_ckpt are mutually exclusive")
        # ppo.init_ckpt: warm-start the agent's weights from a saved state dict (resume after an
        # interrupted run, or fine-tune from a prior experiment). Optimizer moments are NOT
        # restored — set total_decisions to the REMAINING budget and entropy_coef to the schedule
        # value at the interruption point (the decay runs over this config's own total).
        if self.ppo_cfg.get("init_ckpt"):
            initial = torch.load(self.ppo_cfg["init_ckpt"], map_location=self.device)
            if self.ppo_cfg.get("init_compatible", False):
                manifest = self.agent.load_compatible_state_dict(initial)
                loaded = len(manifest["loaded"])
                initialized = len(manifest["initialized"])
                print(
                    f"[init] compatible warm start: {loaded} tensors loaded, "
                    f"{initialized} initialized for the new architecture",
                    flush=True,
                )
                import json

                init_manifest = self.ckpt_dir.parent / "init_compatibility.json"
                init_manifest.write_text(json.dumps(manifest, indent=2))
            else:
                self.agent.load_state_dict(initial)
        self.rule_prior0 = float(self.agent.policy_value.simple_rule_prior_strength)
        self.rule_prior_final = float(
            cfg.get("model", {}).get(
                "simple_rule_prior_strength_final", self.rule_prior0
            )
        )
        if self.rule_prior_final < 0:
            raise ValueError("model.simple_rule_prior_strength_final must be non-negative")
        trainable_prefixes = tuple(self.ppo_cfg.get("trainable_prefixes", ()) or ())
        trainable_lr_scales = dict(
            self.ppo_cfg.get("trainable_lr_scales", {}) or {}
        )
        optimizer_groups, trainable, group_names = _restricted_optimizer_groups(
            self.agent, trainable_prefixes, trainable_lr_scales, self.lr
        )
        if trainable_prefixes:
            print(
                f"[ppo] restricted update to {sum(p.numel() for p in trainable):,} parameters "
                f"under {trainable_prefixes}",
                flush=True,
            )
        for scale, names in sorted(group_names.items(), reverse=True):
            print(
                f"[ppo] lr scale {scale:g}: {len(names)} tensors at {self.lr * scale:g}",
                flush=True,
            )
        self.opt = torch.optim.AdamW(optimizer_groups)
        from fog.training.replay_regularizer import ReplayPolicyRegularizer

        self.replay_regularizer = ReplayPolicyRegularizer(cfg, self.device)
        if self.replay_regularizer.enabled:
            print(
                "[ppo] expert replay term enabled: "
                f"coef={self.replay_regularizer.coef:g}, "
                f"belief={self.replay_regularizer.belief_coef:g}, "
                f"chain={self.replay_regularizer.belief_chain_coef:g}, "
                f"value={self.replay_regularizer.value_coef:g}, "
                f"steps/update={self.replay_regularizer.steps_per_update}",
                flush=True,
            )
        league_cfg = cfg.get("league", {}) or {}
        self.league = League(
            latest_frac=float(league_cfg.get("latest_frac", 0.50)),
            past_frac=float(league_cfg.get("past_frac", 0.35)),
            scripted_frac=float(league_cfg.get("scripted_frac", 0.15)),
            snapshot_every_decisions=self.snap_every,
            warmup_opponent=league_cfg.get("warmup_opponent"),
            warmup_decisions=int(league_cfg.get("warmup_decisions", 0)),
            **({"scripted": tuple(scripted)} if scripted else {}),
        )
        # league.past_ckpts: seed past-self opponents (a resumed run keeps facing the snapshots
        # its interrupted predecessor banked; latest = the last entry).
        seed_past = [Path(p) for p in cfg.get("league", {}).get("past_ckpts", [])]
        if seed_past:
            missing = [p for p in seed_past if not p.exists()]
            if missing:
                raise FileNotFoundError(f"league.past_ckpts not found: {missing}")
            self.league.past_ckpts.extend(seed_past)
            self.league.latest_ckpt = seed_past[-1]
        # leakage.clip_delta: True = per-step max(0, ΔI) (§7 default); False = signed ΔI, whose
        # per-game sum TELESCOPES to CE_first − CE_final — immune to the clipped estimator's
        # CE-noise bias (E[Σ max(0,ΔI)] grows with probe-CE variance even at zero net reveal).
        # Run at least one β cell with clip_delta=false as the estimator ablation.
        self.buf = RolloutBuffer(
            gamma=self.ppo_cfg.get("gamma", 1.0), lam=self.ppo_cfg.get("lam", 0.95),
            beta=self.beta, clip_deltas=bool(self.leak_cfg.get("clip_delta", True)),
            progress_coef=float(cfg.get("reward", {}).get("progress_coef", 0.0)),
        )
        self.dash = _Dashboard(cfg.get("log", {}).get("dir", "runs"))

        # pool truth (belief supervision fallback / oxygen check §7)
        self.pool_truth = None
        self.pool_sample_counts = (
            np.zeros(len(self.pool_teams), dtype=np.int64)
            if self.pool_teams is not None else np.zeros(0, dtype=np.int64)
        )
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
            self.bc_anchor.policy_value.simple_rule_prior_strength = float(
                self.ppo_cfg.get(
                    "bc_anchor_rule_prior_strength", self.rule_prior0
                )
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
        self.backend_rng = None
        self.resume_backend_rng_state = None
        if self.ppo_cfg.get("resume_ckpt"):
            from fog.training.training_state import restore_training_state

            restore_training_state(self, self.ppo_cfg["resume_ckpt"])
            if self.decisions_seen >= self.total:
                raise ValueError(
                    f"resume checkpoint already consumed {self.decisions_seen} decisions, "
                    f"not less than configured total {self.total}"
                )
            print(
                f"[resume] exact training state restored at {self.decisions_seen} decisions",
                flush=True,
            )
        from fog.training.diagnostics import tera_bonus_at

        self.rollout_tera_logit_bonus = tera_bonus_at(self.ppo_cfg, self.decisions_seen, self.total)


def _maybe_chunk_update(s: _TrainSetup, phi_owner, recent_results: dict) -> str | None:
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
    # This exact value was held fixed for the entire collected batch.
    step_ppo["rollout_tera_logit_bonus"] = s.rollout_tera_logit_bonus
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

    metrics = ppo_update(
        s.agent,
        s.opt,
        batch,
        step_cfg,
        replay_regularizer=s.replay_regularizer,
    )
    metrics.update(probe_metrics)
    metrics.update(_belief_accuracy_vs_turn(s.agent, batch))
    s.buf.clear()
    s.decisions_seen += n
    from fog.training.diagnostics import tera_bonus_at

    s.rollout_tera_logit_bonus = tera_bonus_at(s.ppo_cfg, s.decisions_seen, s.total)
    if hasattr(phi_owner, "tera_logit_bonus"):
        phi_owner.tera_logit_bonus = s.rollout_tera_logit_bonus

    # The rule player's choice is a safety prior, not a permanent hard override.  Annealing it
    # between batches lets the learned scorer make evidence-backed deviations while the frozen
    # anchor above still penalizes wholesale drift.  Changing it only after the update preserves
    # the acting-vs-update probability contract for the batch just consumed.
    rule_frac = min(1.0, s.decisions_seen / max(1, s.total))
    rule_strength = s.rule_prior0 + (s.rule_prior_final - s.rule_prior0) * rule_frac
    s.agent.policy_value.simple_rule_prior_strength = float(rule_strength)

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
        rule_prior_strength=float(rule_strength),
    )
    for name, results in recent_results.items():
        if results:
            metrics[f"win_rate_vs_{name}"] = float(np.mean(results))
    if len(s.pool_sample_counts):
        metrics["pool_teams_seen"] = float(np.count_nonzero(s.pool_sample_counts))
        metrics["pool_min_games"] = float(s.pool_sample_counts.min())
        metrics["pool_max_games"] = float(s.pool_sample_counts.max())
        for index, count in enumerate(s.pool_sample_counts):
            metrics[f"pool_team_{index}_games"] = float(count)
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
            from fog.training.training_state import save_training_state

            save_training_state(s, s.ckpt_dir / "ppo_watchdog_halt.pt")
            metrics["watchdog_halt"] = 1.0
            s.dash.log(s.decisions_seen, metrics)
            return "halt"

    if s.beta > 0 and s.decisions_seen >= s.next_refreeze:
        phi_owner.phi_hat = FrozenProbe.from_agent(s.agent).to(s.device)
        s.next_refreeze += s.refreeze_every

    if s.decisions_seen >= s.next_snapshot:
        from fog.training.training_state import save_training_state

        path = s.ckpt_dir / f"ppo_{s.decisions_seen}.pt"
        if not s.cfg.get("league", {}).get("freeze_history", False):
            s.league.past_ckpts.append(path)
            s.league.latest_ckpt = path
        s.next_snapshot += s.snap_every
        save_training_state(s, path)
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

    adapter = FogPlayer(
        agent,
        temperature=s.rollout_temperature,
        tera_logit_bonus=s.rollout_tera_logit_bonus,
    )
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
        _paste_tokens: dict[str, np.ndarray] = {}

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
        opp_spec = league.sample_opponent(s.rng, decisions_seen=s.decisions_seen)
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
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - warning below is explicit
            print(f"[ppo] WARNING: reset_battles failed ({exc}); battle objects retained "
                  "this chunk (memory will grow).", flush=True)

        if _maybe_chunk_update(s, adapter, recent_results) == "halt":
            return s.metrics

    final = s.ckpt_dir / "ppo_final.pt"
    from fog.training.training_state import save_training_state

    save_training_state(s, final)
    print(f"[ppo] done: {s.decisions_seen} decisions -> {final}", flush=True)
    return s.metrics


def _train_engine(cfg: dict) -> dict:
    """Engine-backend training loop (ARCHITECTURE.md §3.2): synchronous batched lockstep
    self-play on poke-engine, sharing the exact update tail with the Showdown path.

    Differences from _train_async, all deliberate:
      * no server / poke-env players — EngineRunner drives EngineBattles directly;
      * scripted league opponents are the engine ports (Random, MaxDamage). SimpleHeuristics is
        Showdown-only and raises here (eval it on Showdown; ARCHITECTURE.md §3.2);
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
        temperature=s.rollout_temperature,
        tera_logit_bonus=s.rollout_tera_logit_bonus,
        phi_hat=phi_hat,
        leak_family_weights=s.leak_cfg.get("family_weights"),
        oracle_truth=(s.oracle_mode == "set"),
    )
    s.backend_rng = runner.rng
    if s.resume_backend_rng_state is not None:
        runner.rng.setstate(s.resume_backend_rng_state)

    cross_cfg = cfg.get("cross_sim", {})
    cross_every = int(cross_cfg.get("every_decisions", 0))
    next_cross = (
        ((s.decisions_seen // cross_every) + 1) * cross_every
        if cross_every else None
    )

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
        opp_spec = s.league.sample_opponent(s.rng, decisions_seen=s.decisions_seen)
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
                    f"(ARCHITECTURE.md §3.2). Checkpoint saved for inspection.",
                    flush=True,
                )
                from fog.training.training_state import save_training_state

                save_training_state(s, s.ckpt_dir / "ppo_cross_sim_halt.pt")
                s.metrics["cross_sim_halt"] = 1.0
                s.dash.log(s.decisions_seen, s.metrics)
                return s.metrics
            next_cross += cross_every

    final = s.ckpt_dir / "ppo_final.pt"
    from fog.training.training_state import save_training_state

    save_training_state(s, final)
    print(f"[ppo] done (engine backend): {s.decisions_seen} decisions -> {final}", flush=True)
    return s.metrics


def _copy_module_to_shared_cpu(target, source) -> None:
    """Refresh shared CPU tensors in place so spawned actors see one atomic-between-chunks policy."""
    with torch.no_grad():
        target_state = target.state_dict()
        source_state = source.state_dict()
        if target_state.keys() != source_state.keys():
            raise RuntimeError("shared rollout actor does not match learner state")
        for key, target_value in target_state.items():
            target_value.copy_(source_state[key].detach().to(device="cpu"))


class _ShowdownRolloutWorker:
    """Long-lived CPU actor plus one long-lived authoritative Showdown subprocess."""

    _SCRIPTED = frozenset({"Random", "MaxDamage", "SimpleHeuristics"})

    def __init__(
        self,
        worker_id: int,
        cfg: dict,
        shared_actor,
        shared_phi,
        pool_teams: list[str],
        rollout_settings: tuple[float, float, float, bool, float],
        supervision: str,
        client_kwargs: dict,
        actor_threads: int,
        opponent_cache_size: int,
    ) -> None:
        from collections import OrderedDict

        import torch

        from fog.constants.tables import TABLES
        from fog.env.player import FogPlayer
        from fog.env.showdown_direct import ShowdownDirectClient
        from fog.env.showdown_local_runner import bind_fog_player
        from fog.env.team_truth import truth_tokens_from_paste
        from fog.env.teams import to_packed

        self.worker_id = int(worker_id)
        self.cfg = cfg
        self.pool_teams = pool_teams
        self.rollout_settings = rollout_settings
        self.supervision = supervision
        self.oracle_mode = str(cfg.get("model", {}).get("oracle", "none"))
        self.fmt = cfg.get("env", {}).get("battle_format", "gen9ou")
        self.packed = {paste: to_packed(paste) for paste in pool_teams}
        self.truth = {
            paste: truth_tokens_from_paste(paste, TABLES) for paste in pool_teams
        }
        self.current_private_state: str | None = None
        self.current_opponent: str | None = None
        self.opponent_cache_size = max(1, int(opponent_cache_size))
        self.opponents = OrderedDict()

        torch.set_num_threads(max(1, int(actor_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.set_grad_enabled(False)
        torch.manual_seed(int(cfg.get("seed", 0)) + self.worker_id)

        self.actor = shared_actor
        self.actor.eval()
        self.adapter = bind_fog_player(
            FogPlayer(
                self.actor,
                temperature=float(cfg.get("ppo", {}).get("rollout_temperature", 1.0)),
                tera_logit_bonus=float(
                    cfg.get("ppo", {}).get("rollout_tera_logit_bonus", 0.0)
                ),
            )
        )
        self.adapter.leak_family_weights = cfg.get("leakage", {}).get("family_weights")
        self.adapter.phi_hat = shared_phi

        def local_truth(battle):
            paste = self.current_opponent
            if paste is None:
                return None
            complete = self.truth[paste]
            if self.supervision == "replay_censored":
                from fog.training.replay_supervision import oracle_censored_truth

                return oracle_censored_truth(battle, complete, TABLES)
            return complete.copy()

        self.adapter.truth_provider = local_truth
        self.adapter.private_state_truth_provider = lambda _battle: (
            None
            if self.current_private_state is None
            else self.truth[self.current_private_state].copy()
        )
        if self.oracle_mode == "set":
            self.adapter.oracle_truth_provider = lambda _battle: (
                None
                if self.current_opponent is None
                else self.truth[self.current_opponent].copy()
            )
        self.client = ShowdownDirectClient(**client_kwargs).start()

    def _opponent_controller(self, spec: str):
        if spec in self._SCRIPTED:
            return spec.lower()
        cached = self.opponents.pop(spec, None)
        if cached is not None:
            self.opponents[spec] = cached
            return cached

        import torch

        from fog.env.player import FogPlayer
        from fog.env.showdown_local_runner import bind_fog_player
        from fog.models.agent import FogAgent

        frozen = FogAgent.from_tables(self.cfg).cpu()
        frozen.load_compatible_state_dict(torch.load(spec, map_location="cpu"))
        frozen.eval()
        controller = bind_fog_player(FogPlayer(frozen))
        self.opponents[spec] = controller
        while len(self.opponents) > self.opponent_cache_size:
            self.opponents.popitem(last=False)
        return controller

    def run(self, payload):
        import numpy as np
        import torch

        from fog.env.showdown_local_runner import play_local_battle
        from fog.training.rollout import RolloutBuffer

        assigned, opponent_spec, tera_bonus = payload
        self.adapter.tera_logit_bonus = float(tera_bonus)
        gamma, lam, beta, clip_deltas, progress_coef = self.rollout_settings
        local_buf = RolloutBuffer(
            gamma=gamma,
            lam=lam,
            beta=beta,
            clip_deltas=clip_deltas,
            progress_coef=progress_coef,
        )
        self.adapter.attach_buffer(local_buf)
        controller = self._opponent_controller(str(opponent_spec))
        wins = 0
        for game in assigned:
            p1_paste = self.pool_teams[game.p1_index]
            p2_paste = self.pool_teams[game.p2_index]
            self.current_private_state = p1_paste
            self.current_opponent = p2_paste
            # Policy sampling is per-game deterministic, independent of process scheduling.
            torch.manual_seed(game.seed % (2**63 - 1))
            np.random.seed(game.seed % (2**32))
            with torch.inference_mode():
                result = play_local_battle(
                    self.client,
                    {
                        "formatid": self.fmt,
                        "seed": f"sodium,{game.seed:064x}",
                        "strictChoices": True,
                        "p1": {"name": "FOG", "team": self.packed[p1_paste]},
                        "p2": {"name": "Opponent", "team": self.packed[p2_paste]},
                    },
                    self.adapter,
                    controller,
                    seed=game.seed & (2**31 - 1),
                    battle_tag=game.battle_tag,
                )
            wins += int(result.p1_won is True)
        return local_buf, wins

    def close(self) -> None:
        self.client.close()


def _train_showdown_direct(cfg: dict) -> dict:
    """Train on authoritative Showdown rules without the network/server layer.

    Showdown produces player-censored protocol messages and private requests in memory. They pass
    through poke-env's ordinary battle parser before observation construction, so this backend
    changes transport and scheduling, not the information available to the policy.
    """
    import os
    from collections import deque
    from contextlib import ExitStack
    from pathlib import Path

    import torch

    from fog.constants.tables import TABLES
    from fog.env.player import FogPlayer
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player, play_local_battle
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import to_packed
    from fog.models.agent import FogAgent
    from fog.models.probe import FrozenProbe
    from fog.training.direct_parallel import (
        PersistentProcessPool,
        WorkerGame,
        partition_games,
        run_workers,
    )
    from fog.training.rollout import RolloutBuffer

    scripted = tuple(
        cfg.get("league", {}).get("scripted", ("MaxDamage", "SimpleHeuristics"))
    )
    supported = {"Random", "MaxDamage", "SimpleHeuristics"}
    unknown = [name for name in scripted if name not in supported]
    if unknown:
        raise ValueError(
            f"league.scripted {unknown} cannot run on showdown_direct; "
            f"supported={sorted(supported)}"
        )

    s = _TrainSetup(cfg, scripted=scripted, require_pool=True)
    assert s.pool_teams is not None
    from fog.training.diagnostics import direct_rng_streams

    rng, opponent_rng = direct_rng_streams(s)
    worker_count = int(s.env_cfg.get("rollout_workers", 1))
    if worker_count < 1:
        raise ValueError("env.rollout_workers must be >= 1")
    worker_mode = str(s.env_cfg.get("rollout_worker_mode", "process")).lower()
    if worker_mode not in {"process", "thread"}:
        raise ValueError("env.rollout_worker_mode must be 'process' or 'thread'")
    actor_threads = int(
        s.env_cfg.get(
            "actor_threads",
            1 if worker_mode == "process" else max(1, (os.cpu_count() or 1) // worker_count),
        )
    )
    if actor_threads < 1:
        raise ValueError("env.actor_threads must be >= 1")
    if worker_count > 1:
        if worker_count > s.games_per_chunk:
            raise ValueError(
                "env.rollout_workers cannot exceed ppo.games_per_chunk "
                "(idle workers add overhead without collecting a game)"
            )
        if worker_mode == "thread":
            torch.set_num_threads(actor_threads)
        print(
            f"[ppo] direct rollout workers={worker_count}, mode={worker_mode}, "
            f"actor_threads={actor_threads}, persistent CPU actors + one centralized learner",
            flush=True,
        )
    packed = {paste: to_packed(paste) for paste in s.pool_teams}
    truth = {paste: truth_tokens_from_paste(paste, TABLES) for paste in s.pool_teams}

    adapter = bind_fog_player(
        FogPlayer(
            s.agent,
            temperature=s.rollout_temperature,
            tera_logit_bonus=s.rollout_tera_logit_bonus,
        )
    )
    adapter.attach_buffer(s.buf)
    adapter.leak_family_weights = s.leak_cfg.get("family_weights")
    if s.beta > 0:
        adapter.phi_hat = FrozenProbe.from_agent(s.agent).to(s.device)

    current_private_state: dict[str, str | None] = {"paste": None}
    current_opponent: dict[str, str | None] = {"paste": None}
    balanced_opponent_order: list[int] = []

    def sample_opponent_team_index() -> int:
        if not s.env_cfg.get("balanced_team_sampling", False):
            return rng.randrange(len(s.pool_teams))
        if not balanced_opponent_order:
            balanced_opponent_order.extend(range(len(s.pool_teams)))
            rng.shuffle(balanced_opponent_order)
        return balanced_opponent_order.pop()

    supervision = str(s.env_cfg.get("belief_supervision", "full_truth"))
    if supervision not in ("full_truth", "replay_censored"):
        raise ValueError(
            "env.belief_supervision must be 'full_truth' or 'replay_censored', "
            f"got {supervision!r}"
        )

    def opponent_truth(_battle):
        paste = current_opponent["paste"]
        if paste is None:
            return None
        complete = truth[paste]
        if supervision == "replay_censored":
            from fog.training.replay_supervision import oracle_censored_truth

            return oracle_censored_truth(_battle, complete, TABLES)
        return complete.copy()

    adapter.truth_provider = opponent_truth
    adapter.private_state_truth_provider = lambda _battle: (
        None
        if current_private_state["paste"] is None
        else truth[current_private_state["paste"]].copy()
    )
    if s.oracle_mode == "set":
        adapter.oracle_truth_provider = lambda _battle: (
            None
            if current_opponent["paste"] is None
            else truth[current_opponent["paste"]].copy()
        )

    frozen_opponents: dict[str, FogPlayer] = {}

    def opponent_controller(spec):
        name = str(spec)
        if name in supported:
            return name.lower()
        if name not in frozen_opponents:
            frozen = FogAgent.from_tables(cfg).to(s.device)
            frozen.load_state_dict(torch.load(spec, map_location=s.device))
            frozen.eval()
            frozen_opponents[name] = bind_fog_player(FogPlayer(frozen))
        return frozen_opponents[name]

    recent_results: dict[str, deque] = {}
    battle_number = 0
    client_kwargs = {}
    if s.env_cfg.get("showdown_root"):
        client_kwargs["showdown_root"] = s.env_cfg["showdown_root"]
    if s.env_cfg.get("node_binary"):
        client_kwargs["node_binary"] = s.env_cfg["node_binary"]
    shared_actor = None
    shared_phi = None
    with ExitStack() as stack:
        process_pool = None
        if worker_count > 1 and worker_mode == "process":
            shared_actor = FogAgent.from_tables(cfg).cpu().eval()
            _copy_module_to_shared_cpu(shared_actor, s.agent)
            shared_actor.share_memory()
            if s.beta > 0:
                shared_phi = FrozenProbe.from_agent(s.agent).cpu().eval()
                assert adapter.phi_hat is not None
                _copy_module_to_shared_cpu(shared_phi, adapter.phi_hat)
                shared_phi.share_memory()
            rollout_settings = (
                float(s.buf.gamma),
                float(s.buf.lam),
                float(s.buf.beta),
                bool(s.buf.clip_deltas),
                float(s.buf.progress_coef),
            )
            process_pool = stack.enter_context(
                PersistentProcessPool(
                    worker_count,
                    _ShowdownRolloutWorker,
                    cfg,
                    shared_actor,
                    shared_phi,
                    s.pool_teams,
                    rollout_settings,
                    supervision,
                    client_kwargs,
                    actor_threads,
                    int(s.env_cfg.get("opponent_cache_size", 2)),
                    timeout=float(s.env_cfg.get("rollout_worker_timeout", 900.0)),
                )
            )
            print(f"[ppo] rollout process pids={process_pool.pids}", flush=True)
            clients = []
        else:
            clients = [
                stack.enter_context(ShowdownDirectClient(**client_kwargs))
                for _ in range(worker_count)
            ]
        client = clients[0] if clients else None
        while s.decisions_seen < s.total:
            opp_spec = s.league.sample_opponent(opponent_rng, decisions_seen=s.decisions_seen)
            opp_name = str(opp_spec) if str(opp_spec) in supported else Path(str(opp_spec)).stem
            if worker_count == 1:
                assert client is not None
                controller = opponent_controller(opp_spec)
                wins = 0
                for _ in range(s.games_per_chunk):
                    p1_index = rng.randrange(len(s.pool_teams))
                    p2_index = sample_opponent_team_index()
                    p1_paste = s.pool_teams[p1_index]
                    p2_paste = s.pool_teams[p2_index]
                    current_private_state["paste"] = p1_paste
                    s.pool_sample_counts[p2_index] += 1
                    current_opponent["paste"] = p2_paste
                    battle_number += 1
                    result = play_local_battle(
                        client,
                        {
                            "formatid": s.fmt,
                            "seed": f"sodium,{rng.getrandbits(256):064x}",
                            "strictChoices": True,
                            "p1": {"name": "FOG", "team": packed[p1_paste]},
                            "p2": {"name": "Opponent", "team": packed[p2_paste]},
                        },
                        adapter,
                        controller,
                        seed=rng.randrange(2**31),
                        battle_tag=f"battle-direct-{battle_number}",
                    )
                    wins += int(result.p1_won is True)
            else:
                games = []
                for _ in range(s.games_per_chunk):
                    battle_number += 1
                    p1_index = rng.randrange(len(s.pool_teams))
                    p2_index = sample_opponent_team_index()
                    s.pool_sample_counts[p2_index] += 1
                    games.append(
                        WorkerGame(
                            worker_id=-1,
                            game_number=battle_number,
                            seed=rng.getrandbits(256),
                            p1_index=p1_index,
                            p2_index=p2_index,
                        )
                    )
                assignments = partition_games(games, worker_count)
                # Preserve the actual exogenous schedule and behavior intervention for attribution.
                import json
                from dataclasses import asdict

                with (s.ckpt_dir.parent / "rollout_schedule.jsonl").open("a", encoding="utf-8") as stream:
                    for game in games:
                        stream.write(json.dumps({**asdict(game), "opponent": str(opp_spec),
                            "learner_decisions": s.decisions_seen,
                            "tera_bonus": s.rollout_tera_logit_bonus}) + "\n")
                if worker_mode == "process":
                    assert process_pool is not None
                    worker_results = process_pool.map(
                        [
                            (assigned, str(opp_spec), s.rollout_tera_logit_bonus)
                            for assigned in assignments
                        ]
                    )
                else:
                    actor_state = {
                        key: value.detach().cpu()
                        for key, value in s.agent.state_dict().items()
                    }
                    opponent_state = None
                    if str(opp_spec) not in supported:
                        opponent_state = torch.load(str(opp_spec), map_location="cpu")

                    def collect(
                        worker_id,
                        assigned,
                        actor_state=actor_state,
                        opponent_state=opponent_state,
                        opponent_spec=str(opp_spec),
                    ):
                        actor = FogAgent.from_tables(cfg).cpu()
                        actor.load_state_dict(actor_state)
                        actor.eval()
                        local_buf = RolloutBuffer(
                            gamma=s.buf.gamma,
                            lam=s.buf.lam,
                            beta=s.buf.beta,
                            clip_deltas=s.buf.clip_deltas,
                            progress_coef=s.buf.progress_coef,
                        )
                        local_adapter = bind_fog_player(
                            FogPlayer(
                                actor,
                                temperature=s.rollout_temperature,
                                tera_logit_bonus=s.rollout_tera_logit_bonus,
                            )
                        )
                        local_adapter.attach_buffer(local_buf)
                        local_adapter.leak_family_weights = s.leak_cfg.get("family_weights")
                        if s.beta > 0:
                            local_adapter.phi_hat = FrozenProbe.from_agent(actor).cpu()
                        local_private_state = {"paste": None}
                        local_current = {"paste": None}

                        def local_truth(battle):
                            paste = local_current["paste"]
                            if paste is None:
                                return None
                            complete = truth[paste]
                            if supervision == "replay_censored":
                                from fog.training.replay_supervision import oracle_censored_truth

                                return oracle_censored_truth(battle, complete, TABLES)
                            return complete.copy()

                        local_adapter.truth_provider = local_truth
                        local_adapter.private_state_truth_provider = lambda _battle: (
                            None
                            if local_private_state["paste"] is None
                            else truth[local_private_state["paste"]].copy()
                        )
                        if s.oracle_mode == "set":
                            local_adapter.oracle_truth_provider = lambda _battle: (
                                None
                                if local_current["paste"] is None
                                else truth[local_current["paste"]].copy()
                            )
                        if opponent_spec in supported:
                            local_controller = opponent_spec.lower()
                        else:
                            frozen = FogAgent.from_tables(cfg).cpu()
                            assert opponent_state is not None
                            frozen.load_compatible_state_dict(opponent_state)
                            frozen.eval()
                            local_controller = bind_fog_player(FogPlayer(frozen))
                        local_wins = 0
                        for game in assigned:
                            p1_paste = s.pool_teams[game.p1_index]
                            p2_paste = s.pool_teams[game.p2_index]
                            local_private_state["paste"] = p1_paste
                            local_current["paste"] = p2_paste
                            result = play_local_battle(
                                clients[worker_id],
                                {
                                    "formatid": s.fmt,
                                    "seed": f"sodium,{game.seed:064x}",
                                    "strictChoices": True,
                                    "p1": {"name": "FOG", "team": packed[p1_paste]},
                                    "p2": {"name": "Opponent", "team": packed[p2_paste]},
                                },
                                local_adapter,
                                local_controller,
                                seed=game.seed & (2**31 - 1),
                                battle_tag=game.battle_tag,
                            )
                            local_wins += int(result.p1_won is True)
                        return local_buf, local_wins

                    worker_results = run_workers(assignments, collect)
                wins = 0
                for local_buf, local_wins in worker_results:
                    s.buf.extend(local_buf)
                    wins += local_wins
            rec = recent_results.setdefault(opp_name, deque(maxlen=200))
            rec.extend([1.0] * wins + [0.0] * (s.games_per_chunk - wins))
            s.chunk_battles += s.games_per_chunk

            update_status = _maybe_chunk_update(s, adapter, recent_results)
            if update_status == "halt":
                return s.metrics
            if update_status == "updated" and process_pool is not None:
                assert shared_actor is not None
                _copy_module_to_shared_cpu(shared_actor, s.agent)
                if shared_phi is not None:
                    assert adapter.phi_hat is not None
                    _copy_module_to_shared_cpu(shared_phi, adapter.phi_hat)

    final = s.ckpt_dir / "ppo_final.pt"
    from fog.training.training_state import save_training_state

    save_training_state(s, final)
    print(
        f"[ppo] done (direct Showdown backend): {s.decisions_seen} decisions -> {final}",
        flush=True,
    )
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

    backbone_batch = {
        k: sub[k]
        for k in (
            "entities", "entity_ids", "field", "history", "active_idx", "history_len",
            "history_ids", "history_state",
        )
        if k in sub and sub[k] is not None
    }
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
    # Preview tensors have one row per game, not one row per decision.
    sub = _index_nested({key: value for key, value in batch.items() if key != "preview"}, idx)
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


def main() -> None:
    from fog.config import load_config_with_args

    ap = argparse.ArgumentParser(description="FOG self-play PPO (§6).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", help="dotted overrides, e.g. leakage.beta=0.03")
    args = ap.parse_args()
    train(load_config_with_args(args.config, args.overrides))


if __name__ == "__main__":
    main()
