"""The research objective — information-leakage penalty.

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


def shaped_reward(
    terminal_reward: float, delta_I: torch.Tensor, beta: float, clip: bool = True
) -> torch.Tensor:
    """r_t = r_game_terminal − β · max(0, ΔI_t)  (clip=True, §7 default).

    clip=False uses the signed ΔI: the per-game sum then TELESCOPES to CE_first − CE_final, so the
    total penalty is exactly the net readability handed over, credit-assigned to the step where it
    happened. The clipped estimator instead accumulates every upward CE fluctuation — its
    expectation grows with probe-CE variance even at zero net reveal, and it rewards *smooth*
    (predictable-early) CE trajectories over spiky ones. Run at least one β cell unclipped
    (leakage.clip_delta=false) as the estimator ablation; the clip's rationale ("can't earn by
    becoming less known") bounds the unclipped credit at total CE anyway.
    """
    penalty = torch.clamp(delta_I, min=0.0) if clip else delta_I
    return terminal_reward - beta * penalty


@dataclass
class DegenerateSolutionWatchdog:
    """Scripted detectors for the §7 failure modes. Wire into the §6 dashboard.

    Baselines come from the matched N-agent (β=0 control) or the pre-penalty checkpoint — pass them in
    at construction and refresh them when the control run's metrics update. Every check returns True
    when the degenerate pattern is detected; the trainer should halt-for-inspection on any True (same
    policy as the >2σ regression rule, spec §10).
    """

    # baselines (from the β=0 control at matched training progress — see update_to). None = the
    # detector needing them is OFF: fabricated defaults are worse than no detector — a 0.5 win-rate
    # / 0.0 entropy default made the diversity check GUARANTEED-true on any fresh run's first
    # update (observed live 2026-07-05 on a pool_gen smoke run).
    baseline_win_rate: float | None = None
    baseline_move_entropy: float | None = None
    baseline_game_length: float = 0.0

    # thresholds
    win_rate_crash: float = 0.10          # absolute win-rate drop that counts as a crash
    move_entropy_spike: float = 0.30      # absolute per-turn move-entropy (nats) rise that counts
    stale_phi_gap: float = 0.5            # nats: absolute floor for the frozen-vs-fresh CE excess
    stale_phi_rel: float = 0.25           # AND relative excess: frozen > fresh * (1 + this).
    #   Both are required. leakage_ce is summed over every still-hidden attribute, so its scale is
    #   ~tens of nats early in training — an absolute 0.5-nat gap alone fires on init noise (Φ̂ is a
    #   copy of the untrained head at step one; measured frozen−fresh ≈ 13.6 nats on CEs of ~100,
    #   i.e. a 15% excess, at the FIRST update of a smoke run — with no staleness to exploit).
    stall_length_ratio: float = 1.5       # avg game length vs baseline

    # arm-up: exploitation of a stale Φ̂ cannot exist before the policy has trained against a frozen
    # Φ̂ for a while; skip the stale check below this many decisions (set from config in ppo.py,
    # tracked via update_to).
    warmup_decisions: int = 0
    decisions_seen: int = 0

    @classmethod
    def from_control_run(cls, log_dir, tail_rows: int = 10) -> DegenerateSolutionWatchdog:
        """Load a finished β=0 control run's full metrics.csv and start with final-tail baselines.

        Call `update_to(decisions)` before each check so the baselines come from the control at
        MATCHED training progress: a fresh L-run's first update naturally sits well below the
        control's FINAL win-rate and above its FINAL (decayed) entropy — comparing against the tail
        false-fires both diversity-spike conditions on update one.
        """
        import csv
        from pathlib import Path

        path = Path(log_dir)
        csv_path = path if path.suffix == ".csv" else path / "metrics.csv"
        with open(csv_path, encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
        if not raw:
            raise ValueError(f"no metric rows in {csv_path}")

        rows: list[tuple[int, dict]] = []
        for r in raw:
            try:
                rows.append((int(float(r.get("step", 0))), r))
            except (TypeError, ValueError):
                continue
        rows.sort(key=lambda x: x[0])

        wd = cls()
        wd._control_rows = rows
        wd._tail_rows = tail_rows
        wd.update_to(None)
        return wd

    def update_to(self, decisions: int | None, window: int = 5) -> DegenerateSolutionWatchdog:
        """Set baselines from the control rows at matched progress (`decisions` seen so far).

        None -> final tail (legacy behavior). Earlier than the control's first row -> its first
        `window` rows (the closest available match). No-op when constructed without a control run.
        """
        if decisions is not None:
            self.decisions_seen = int(decisions)
        rows = getattr(self, "_control_rows", None)
        if not rows:
            return self
        if decisions is None:
            sel = rows[-getattr(self, "_tail_rows", 10):]
        else:
            upto = [r for r in rows if r[0] <= decisions]
            sel = upto[-window:] if upto else rows[:window]

        def col(name: str, fallback: str | None = None) -> float:
            key = name if name in sel[0][1] else (fallback if fallback and fallback in sel[0][1] else None)
            if key is None:
                return 0.0
            vals = []
            for _, r in sel:
                try:
                    vals.append(float(r[key]))
                except (TypeError, ValueError, KeyError):
                    pass
            return float(sum(vals) / len(vals)) if vals else 0.0

        self.baseline_win_rate = col("win_rate")
        self.baseline_move_entropy = col("move_entropy", "entropy")
        self.baseline_game_length = col("avg_game_length")
        return self

    def check_move_diversity_spike(self, metrics: dict) -> bool:
        """Refusing to attack with revealing moves at the cost of tempo shows up as a win-rate crash
        paired with a move-diversity (entropy) spike. `metrics` needs win_rate + move_entropy.
        OFF (False) until win-rate/entropy baselines exist — i.e. without a β=0 control run."""
        if self.baseline_win_rate is None or self.baseline_move_entropy is None:
            return False
        crashed = metrics["win_rate"] < self.baseline_win_rate - self.win_rate_crash
        spiked = metrics["move_entropy"] > self.baseline_move_entropy + self.move_entropy_spike
        return bool(crashed and spiked)

    def check_stale_phi_exploitation(self, fresh_ce: float, frozen_ce: float) -> bool:
        """Exploiting Φ̂ staleness: the agent looks opaque to the FROZEN predictor (high frozen CE)
        while a freshly-trained predictor still reads it fine (low fresh CE). Fires only past
        `warmup_decisions` AND when the excess clears both the absolute floor and the relative
        threshold (see the field comments — either alone false-fires at init-scale CE)."""
        if self.decisions_seen < self.warmup_decisions:
            return False
        return bool(
            frozen_ce - fresh_ce > self.stale_phi_gap
            and frozen_ce > fresh_ce * (1.0 + self.stale_phi_rel)
        )

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
