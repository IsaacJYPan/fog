"""Small replay policy/belief/value retention terms mixed into a PPO optimizer step."""
from __future__ import annotations

import math
from pathlib import Path

import torch


class ReplayPolicyRegularizer:
    """Stream held-out-safe replay batches without loading the corpus into RAM."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        from fog.training.dataset import StreamingReplayDataset

        settings = cfg.get("replay_policy", {}) or {}
        self.coef = float(settings.get("coef", 0.0))
        self.belief_coef = float(settings.get("belief_coef", 0.0))
        self.belief_chain_coef = float(settings.get("belief_chain_coef", 0.0))
        self.value_coef = float(settings.get("value_coef", 0.0))
        self.batch_size = int(settings.get("batch_size", 1536))
        self.steps_per_update = int(settings.get("steps_per_update", 2))
        self.seed = int(cfg.get("seed", 0)) + 610_000
        manifest = Path(str(settings.get("manifest", "")))
        for name, coefficient in (
            ("coef", self.coef),
            ("belief_coef", self.belief_coef),
            ("belief_chain_coef", self.belief_chain_coef),
            ("value_coef", self.value_coef),
        ):
            if coefficient < 0 or not math.isfinite(coefficient):
                raise ValueError(
                    f"replay_policy.{name} must be finite and non-negative"
                )
        any_coefficient = any(
            coefficient > 0
            for coefficient in (
                self.coef,
                self.belief_coef,
                self.belief_chain_coef,
                self.value_coef,
            )
        )
        if any_coefficient and not manifest.is_file():
            raise FileNotFoundError(f"replay policy manifest not found: {manifest}")
        self.dataset = (
            StreamingReplayDataset(
                manifest, shard_buffer=int(settings.get("shard_buffer", 16))
            )
            if any_coefficient
            else None
        )
        self.device = device
        self.epoch = 0
        self.iterator = None
        self.remaining = 0
        self.batch_in_epoch = 0
        self._resume_skip = 0

    @property
    def enabled(self) -> bool:
        return self.dataset is not None and self.steps_per_update > 0

    def begin_update(self) -> None:
        self.remaining = self.steps_per_update if self.enabled else 0

    def _next_batch(self) -> dict:
        assert self.dataset is not None
        while True:
            if self.iterator is None:
                epoch = self.epoch
                self.iterator = iter(
                    self.dataset.batches(
                        self.batch_size,
                        shuffle=True,
                        seed=self.seed + epoch,
                        device=self.device,
                    )
                )
                self.epoch += 1
                self.batch_in_epoch = 0
                for _ in range(self._resume_skip):
                    try:
                        next(self.iterator)
                    except StopIteration as exc:
                        raise ValueError("saved replay-regularizer offset exceeds its epoch") from exc
                    self.batch_in_epoch += 1
                self._resume_skip = 0
            try:
                batch = next(self.iterator)
                self.batch_in_epoch += 1
                if self.belief_coef <= 0 and self.belief_chain_coef <= 0:
                    # Avoid a large unused belief forward for policy/value-only retention.
                    for key in ("opp_revealed", "belief_targets", "belief_valid"):
                        batch.pop(key, None)
                for key in ("self_belief_targets", "self_belief_valid"):
                    batch.pop(key, None)
                return batch
            except StopIteration:
                self.iterator = None

    def state_dict(self) -> dict[str, int]:
        active_epoch = self.epoch - 1 if self.iterator is not None else self.epoch
        return {
            "active_epoch": int(active_epoch),
            "batch_in_epoch": int(self.batch_in_epoch if self.iterator is not None else 0),
            "remaining": int(self.remaining),
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.iterator = None
        self.epoch = int(state["active_epoch"])
        self.batch_in_epoch = 0
        self._resume_skip = int(state["batch_in_epoch"])
        self.remaining = int(state.get("remaining", 0))

    def loss(self, agent) -> tuple[torch.Tensor, dict[str, float]] | None:
        if self.remaining <= 0:
            return None
        self.remaining -= 1
        batch = self._next_batch()
        if self.belief_chain_coef > 0:
            batch["compute_belief_chain"] = True
        out = agent(batch)
        total = out.value.new_zeros(())
        metrics: dict[str, float] = {
            "replay_policy_coef": self.coef,
            "replay_belief_coef": self.belief_coef,
            "replay_belief_chain_coef": self.belief_chain_coef,
            "replay_value_coef": self.value_coef,
        }
        if self.coef > 0:
            chosen = -out.log_probs.gather(
                -1, batch["action"].unsqueeze(-1)
            ).squeeze(-1)
            valid = batch.get(
                "policy_valid", torch.ones_like(chosen, dtype=torch.bool)
            ).bool()
            valid &= torch.isfinite(chosen)
            if not bool(valid.any()):
                raise RuntimeError("replay policy batch contains no finite valid decisions")
            nll = chosen[valid].mean()
            top1 = (
                out.log_probs.argmax(-1)[valid] == batch["action"][valid]
            ).float().mean()
            total = total + self.coef * nll
            metrics.update(
                {
                    "replay_policy_nll": float(nll.detach()),
                    "replay_policy_top1": float(top1.detach()),
                    "replay_policy_rows": float(valid.sum()),
                }
            )
        if self.belief_coef > 0:
            from fog.training.losses import belief_loss

            if out.belief is None:
                raise RuntimeError("replay belief retention requires a belief output")
            stats: dict = {}
            marginal = belief_loss(
                out.belief.logits,
                batch["belief_targets"],
                batch["belief_valid"],
                stats=stats,
            )
            total = total + self.belief_coef * marginal
            metrics["replay_belief_loss"] = float(marginal.detach())
            metrics["replay_belief_dropped_count"] = float(
                stats.get("dropped_nonfinite", 0)
            )
        if self.belief_chain_coef > 0:
            from fog.training.losses import belief_output_loss

            if out.belief_chain is None:
                raise RuntimeError("replay chain retention requires a belief-chain output")
            chain = belief_output_loss(
                out.belief_chain, batch["belief_targets"], batch["belief_valid"]
            )
            total = total + self.belief_chain_coef * chain
            metrics["replay_belief_chain_loss"] = float(chain.detach())
        if self.value_coef > 0:
            from fog.training.losses import value_loss

            value = value_loss(out.value, batch["outcome"])
            total = total + self.value_coef * value
            metrics["replay_value_loss"] = float(value.detach())
        return total, metrics


__all__ = ["ReplayPolicyRegularizer"]
