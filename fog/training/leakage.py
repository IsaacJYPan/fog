"""The research objective — information-leakage penalty (spec §7).

Two-player population: L-agents (leakage-penalized) and N-agents (normal), same architecture/budget.

Mechanism:
  * Every 100k decisions, freeze `Φ̂` — a FrozenProbe (backbone + self-belief pair, models/probe.py).
    Freezing the full measurement path matters: a frozen head fed by the live trunk drifts OOD and
    the CE stops meaning "readability".
  * L-agent per-turn reward:  r_t = r_game_terminal − β · max(0, ΔI_t)
    where ΔI_t = CE_t − CE_{t+1} under Φ̂, each CE summed over the attributes still hidden AT THAT
    measurement (the "moving-mask" estimator — canonical, matching the RolloutBuffer derivation):
    a hard reveal between t and t+1 drops out of the t+1 sum, so it contributes its full surprisal
    −log p_Φ̂(truth) at time t — exactly the information the reveal handed over. The battle-finished
    hook contributes a post-game measurement so the LAST decision's reveals are penalized too.
    Intuition: pay β per bit revealed, but only for revealing (max(0,·) — can't earn by becoming less
    known).
  * β schedule / grid: {0 (control), 0.01, 0.03, 0.1}. β too high should visibly degrade win-rate.
  * `family_weights` (config leakage.family_weights) rescales per-family CE in ΔI — the weighted-ΔI
    ablation (a tera type is strategically worth more than a fourth filler move).

Degenerate-solution watchlist (each with a scripted detector, wired into training/ppo.py's loop —
the trainer HALTS on any True):
  * refusing to attack with revealing moves at the cost of tempo (win-rate crash + move-diversity spike)
  * exploiting `Φ̂` staleness (CE of the fresh probe trained alongside diverging from Φ̂'s)
  * stalling (game length vs the β=0 control baseline — set `leakage.baseline_run`)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LeakageConfig:
    enabled: bool = False
    beta: float = 0.0
    refreeze_every_decisions: int = 100_000


def delta_information(phi_hat, obs_before: dict, obs_after: dict) -> torch.Tensor:
    """ΔI_t = CE_before − CE_after under the frozen probe `Φ̂` (moving-mask estimator, canonical).

    Each term sums over the attributes still hidden AT THAT measurement (each obs's own view
    tokens), so an attribute hard-revealed during the turn leaves the second sum and contributes its
    full surprisal to ΔI. Positive ΔI = the opponent got a better read on me across the turn.

    Both obs dicts carry the mirror backbone batch keys plus "my_view" (see encoding/mirror.py);
    `phi_hat` is a models.probe.FrozenProbe. Returns [B]. The live training path derives the same
    quantity by differencing per-decision recorded CEs (RolloutBuffer.to_batch) — this function is
    the reference implementation and the eval-time entry point.
    """
    ce_before = phi_hat.leakage_ce(obs_before, obs_before["my_view"])
    ce_after = phi_hat.leakage_ce(obs_after, obs_after["my_view"])
    return ce_before - ce_after


def shaped_reward(terminal_reward: float, delta_I: torch.Tensor, beta: float) -> torch.Tensor:
    """r_t = r_game_terminal − β · max(0, ΔI_t). Real."""
    return terminal_reward - beta * torch.clamp(delta_I, min=0.0)


@dataclass
class DegenerateSolutionWatchdog:
    """Scripted detectors for the §7 failure modes. Wire into the §6 dashboard.

    Baselines come from the matched N-agent (β=0 control) or the pre-penalty checkpoint — pass them in
    at construction and refresh them when the control run's metrics update. Every check returns True
    when the degenerate pattern is detected; the trainer should halt-for-inspection on any True (same
    policy as the >2σ regression rule, spec §10).
    """

    # baselines (from the β=0 control at matched training progress)
    baseline_win_rate: float = 0.5
    baseline_move_entropy: float = 0.0
    baseline_game_length: float = 0.0

    # thresholds
    win_rate_crash: float = 0.10          # absolute win-rate drop that counts as a crash
    move_entropy_spike: float = 0.30      # absolute per-turn move-entropy (nats) rise that counts
    stale_phi_gap: float = 0.5            # nats: frozen-Φ̂ CE exceeding fresh-predictor CE by this
    stall_length_ratio: float = 1.5       # avg game length vs baseline

    @classmethod
    def from_control_run(cls, log_dir, tail_rows: int = 10) -> "DegenerateSolutionWatchdog":
        """Baselines from a finished β=0 control run's metrics.csv (mean of the last `tail_rows`).

        Reads the win_rate / move_entropy (falls back to entropy) / avg_game_length columns written
        by training/ppo.py's dashboard.
        """
        import csv
        from pathlib import Path

        path = Path(log_dir)
        csv_path = path if path.suffix == ".csv" else path / "metrics.csv"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        tail = rows[-tail_rows:]
        if not tail:
            raise ValueError(f"no metric rows in {csv_path}")

        def col(name: str, fallback: str | None = None) -> float:
            key = name if name in tail[0] else (fallback if fallback and fallback in tail[0] else None)
            if key is None:
                return 0.0
            return float(sum(float(r[key]) for r in tail) / len(tail))

        return cls(
            baseline_win_rate=col("win_rate"),
            baseline_move_entropy=col("move_entropy", "entropy"),
            baseline_game_length=col("avg_game_length"),
        )

    def check_move_diversity_spike(self, metrics: dict) -> bool:
        """Refusing to attack with revealing moves at the cost of tempo shows up as a win-rate crash
        paired with a move-diversity (entropy) spike. `metrics` needs win_rate + move_entropy."""
        crashed = metrics["win_rate"] < self.baseline_win_rate - self.win_rate_crash
        spiked = metrics["move_entropy"] > self.baseline_move_entropy + self.move_entropy_spike
        return bool(crashed and spiked)

    def check_stale_phi_exploitation(self, fresh_ce: float, frozen_ce: float) -> bool:
        """Exploiting Φ̂ staleness: the agent looks opaque to the FROZEN predictor (high frozen CE)
        while a freshly-trained predictor still reads it fine (low fresh CE)."""
        return bool(frozen_ce - fresh_ce > self.stale_phi_gap)

    def check_stalling(self, avg_game_length: float) -> bool:
        return bool(
            self.baseline_game_length > 0
            and avg_game_length > self.stall_length_ratio * self.baseline_game_length
        )

    def check_all(self, metrics: dict) -> dict[str, bool]:
        """Run every detector off one dashboard metrics dict; keys absent -> detector skipped."""
        out: dict[str, bool] = {}
        if "win_rate" in metrics and "move_entropy" in metrics:
            out["move_diversity_spike"] = self.check_move_diversity_spike(metrics)
        if "fresh_ce" in metrics and "frozen_ce" in metrics:
            out["stale_phi_exploitation"] = self.check_stale_phi_exploitation(
                metrics["fresh_ce"], metrics["frozen_ce"]
            )
        if "avg_game_length" in metrics:
            out["stalling"] = self.check_stalling(metrics["avg_game_length"])
        return out
