"""Read-only probability, outcome, and gradient diagnostics for frozen-policy rollouts."""
from __future__ import annotations

import math

import numpy as np
import torch


def probability_contract(old: torch.Tensor, new: torch.Tensor, actions: torch.Tensor,
                         recorded: torch.Tensor, clip: float = .05) -> dict:
    """Compare the complete legal distributions and the separately recorded chosen log-probability.

    Renormalizing in float64 makes KL insensitive to float32 softmax normalization error.
    Raw log-probability errors and selected ratios remain unrenormalized, as in PPO.
    """
    if (old.shape != new.shape or old.ndim != 2 or not len(old)
            or actions.shape != (len(old),) or recorded.shape != (len(old),)):
        raise ValueError("Mismatched policy batch shapes")
    if actions.dtype != torch.long or (actions < 0).any() or (actions >= old.shape[1]).any():
        raise ValueError("Actions must be valid integer indices")
    if any(x.device != old.device for x in (new, actions, recorded)):
        raise ValueError("Policy tensors must share a device")
    if not 0 < clip < 1:
        raise ValueError("clip must be between zero and one")
    if any(torch.isnan(x).any() or torch.isposinf(x).any() for x in (old, new)):
        raise ValueError("Policy contains NaN or positive infinity")
    support = torch.isfinite(old)
    if not support.any(-1).all() or not torch.equal(support, torch.isfinite(new)):
        raise ValueError("Policy support changed or an action row is empty")
    selected_old = old.gather(1, actions[:, None]).squeeze(1)
    selected_new = new.gather(1, actions[:, None]).squeeze(1)
    if not torch.isfinite(selected_old).all() or not torch.isfinite(recorded).all():
        raise ValueError("Recorded action is outside the behavior support")
    delta = selected_new.double() - recorded.double()
    old64 = old.double() - old.double().logsumexp(-1, keepdim=True)
    new64 = new.double() - new.double().logsumexp(-1, keepdim=True)
    difference = torch.where(support, old64 - new64, 0.)
    kl = (old64.exp() * difference).sum(-1)
    return {
        "rows": len(old), "legal_entries": int(support.sum()),
        "chosen_log_prob_abs_error_quantiles": torch.quantile(
            delta.abs(), torch.tensor([.5, .95, .99, 1.], dtype=torch.float64,
                                      device=old.device)).tolist(),
        "legal_log_prob_max_abs_error": float((old - new)[support].abs().max()),
        "recorded_vs_distribution_max_abs_error": float((selected_old - recorded).abs().max()),
        "ratio_min": float(delta.exp().min()), "ratio_max": float(delta.exp().max()),
        "clip_fraction": float(((delta.exp() - 1).abs() > clip).double().mean()),
        "categorical_kl_mean": float(kl.mean()), "categorical_kl_max": float(kl.max()),
    }


def game_mean_interval(values, *, seed: int = 0, samples: int = 5000) -> dict:
    """Bootstrap independent game summaries, giving long and short games equal weight."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("Expected finite, nonempty per-game scalar values")
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(len(data), size=(samples, len(data)))].mean(-1)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(games=len(data), mean=float(data.mean()), low=float(lo), high=float(hi))


def outcome_metrics(games: list[dict], baseline: float) -> dict:
    """Score values on [-1,1] outcomes, weighting each game equally at every reduction.

    Each record contains ``values`` and a terminal ``outcome``. The constant baseline must
    be estimated on separate games. Explained variance is undefined for constant outcomes.
    """
    if not games or not math.isfinite(baseline):
        raise ValueError("Need games and a finite separately fitted baseline")
    errors, constant_errors, residual_means, squared_residuals, targets = [], [], [], [], []
    for game in games:
        values = np.asarray(game['values'], dtype=np.float64)
        target = float(game['outcome'])
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or target not in (-1, 0, 1):
            raise ValueError("Invalid value trajectory or terminal outcome")
        residual = target - values
        errors.append(float(np.mean(residual ** 2)))
        constant_errors.append((target - baseline) ** 2)
        residual_means.append(float(residual.mean()))
        squared_residuals.append(float(np.mean(residual ** 2)))
        targets.append(target)
    target_variance = float(np.var(targets))
    residual_variance = float(np.mean(squared_residuals) - np.mean(residual_means) ** 2)
    return dict(value_mse=game_mean_interval(errors), baseline_mse=game_mean_interval(constant_errors),
                mse_difference=game_mean_interval(np.array(errors) - constant_errors),
                baseline=baseline, mean_outcome=float(np.mean(targets)),
                explained_variance=(1 - residual_variance / target_variance
                                    if target_variance > 0 else None))


def gradient_geometry(losses: dict[str, torch.Tensor], named_parameters: list[tuple],
                      groups: dict[str, list[int]]) -> dict:
    """Measure weighted objective gradients without accumulating .grad or stepping an optimizer.

    Report their Gram matrix, alignment with the policy term, and the combined gradient.
    Parameter groups must index this exact deduplicated named-parameter list.
    """
    names = list(losses)
    parameters = [p for _, p in named_parameters]
    gradients = {}
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite objective: {name}")
        gradients[name] = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    report = {}
    for group, indices in groups.items():
        gram = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j in range(i + 1):
                b = names[j]
                dot = sum(float((gradients[a][k].detach().double() *
                                 gradients[b][k].detach().double()).sum()) for k in indices
                          if gradients[a][k] is not None and gradients[b][k] is not None)
                gram[i, j] = gram[j, i] = dot
        norm = np.sqrt(np.maximum(0, gram.diagonal()))
        combined = math.sqrt(max(0, float(gram.sum())))
        policy = names.index('policy')
        report[group] = dict(
            names=names, gram=gram.tolist(), norms=dict(zip(names, norm.tolist(), strict=True)),
            combined_norm=combined,
            cosine_with_policy={name: float(gram[policy, i] / (norm[policy] * norm[i]))
                                if norm[policy] * norm[i] > 0 else None
                                for i, name in enumerate(names)},
            combined_policy_cosine=(float(gram[policy].sum() / (norm[policy] * combined))
                                    if norm[policy] * combined > 0 else None),
        )
    return report
