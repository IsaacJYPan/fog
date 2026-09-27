"""Game-weighted outcome fitting for a readout of already frozen model features."""
from __future__ import annotations

import copy

import torch


def game_weighted_mse(prediction: torch.Tensor, outcome: torch.Tensor,
                      game_id: torch.Tensor) -> torch.Tensor:
    """Give each game equal weight even when trajectories have different lengths."""
    if prediction.ndim != 1 or prediction.shape != outcome.shape or game_id.shape != outcome.shape:
        raise ValueError("Expected matching one-dimensional predictions, outcomes and game ids")
    if not prediction.numel() or not torch.isfinite(prediction).all() or not torch.isfinite(outcome).all():
        raise ValueError("Expected finite, nonempty trajectories")
    _, inverse, count = torch.unique(game_id, return_inverse=True, return_counts=True)
    return (((prediction-outcome)**2) / count[inverse]).sum() / len(count)


def fit_readout(head, fit: dict, development: dict, *, seed: int, lr: float = 1e-4,
                weight_decay: float = 1e-4, max_epochs: int = 60, patience: int = 8,
                min_delta: float = .001, games_per_batch: int = 12) -> dict:
    """Fit a private head copy; selection sees development games only, including epoch zero.

    Features must already be detached. No agent, policy, representation, or report split is
    accepted, which keeps the experiment's training boundary explicit.
    """
    if any(data['features'].requires_grad for data in (fit, development)):
        raise ValueError("Readout features must be detached")
    if set(fit['game_id'].tolist()) & set(development['game_id'].tolist()):
        raise ValueError("Fit and development game identities overlap")
    model = copy.deepcopy(head).eval()
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    ids = fit['game_id'].unique()

    @torch.no_grad()
    def score(data):
        prediction = model(data['features']).squeeze(-1).tanh()
        return float(game_weighted_mse(prediction, data['outcome'], data['game_id']))

    best = score(development)
    state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    history = [dict(epoch=0, fit_mse=score(fit), development_mse=best)]
    for epoch in range(1, max_epochs+1):
        order = ids[torch.randperm(len(ids), generator=generator)]
        for start in range(0, len(order), games_per_batch):
            chosen = torch.isin(fit['game_id'], order[start:start+games_per_batch])
            prediction = model(fit['features'][chosen]).squeeze(-1).tanh()
            loss = game_weighted_mse(prediction, fit['outcome'][chosen], fit['game_id'][chosen])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        value = score(development)
        history.append(dict(epoch=epoch, fit_mse=score(fit), development_mse=value))
        if value < best-min_delta:
            best, state, best_epoch, stale = value, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    return dict(model=model, history=history, best_epoch=best_epoch, best_development_mse=best)
