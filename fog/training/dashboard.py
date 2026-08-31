"""Stable experiment-metric reporting independent of PPO rollout orchestration."""

from __future__ import annotations

from pathlib import Path


class TrainingDashboard:
    """CSV plus optional TensorBoard scalar logger with a complete evolving header."""

    def __init__(self, log_dir: str) -> None:
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "metrics.csv"
        self.keys: list[str] = []
        self.rows: list[tuple[int, dict]] = []
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.dir / "tb"))
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"[training] tensorboard disabled: {type(exc).__name__}: {exc}", flush=True)

    def log(self, step: int, metrics: dict) -> None:
        scalars = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        self.rows.append((step, scalars))
        new_keys = [key for key in scalars if key not in self.keys]
        if new_keys:
            self.keys.extend(new_keys)
            with self.csv.open("w", encoding="utf-8") as stream:
                stream.write("step," + ",".join(self.keys) + "\n")
                stream.writelines(
                    f"{logged_step},"
                    + ",".join(f"{row.get(key, 0.0):.5f}" for key in self.keys)
                    + "\n"
                    for logged_step, row in self.rows
                )
        else:
            with self.csv.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{step},"
                    + ",".join(f"{scalars.get(key, 0.0):.5f}" for key in self.keys)
                    + "\n"
                )
        if self.tb is not None:
            for key, value in scalars.items():
                self.tb.add_scalar(key, value, step)
