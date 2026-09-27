"""Run PPO or behavior cloning from the CLI or a Python training session."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from fog.config import load_config_with_args


class TrainerKind(StrEnum):
    PPO = "ppo"
    BEHAVIOR_CLONING = "bc"


class Trainer(Protocol):
    """Training backend interface."""

    config: dict[str, Any]

    def run(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class PPOTrainer:
    config: dict[str, Any]

    def run(self) -> dict[str, Any]:
        from fog.training.ppo import train

        return train(self.config)


@dataclass(slots=True)
class BehaviorCloningTrainer:
    config: dict[str, Any]

    def run(self) -> dict[str, Any]:
        from fog.training.bc import train

        return train(self.config)


_TRAINERS: dict[TrainerKind, type[PPOTrainer | BehaviorCloningTrainer]] = {
    TrainerKind.PPO: PPOTrainer,
    TrainerKind.BEHAVIOR_CLONING: BehaviorCloningTrainer,
}


@dataclass(slots=True)
class TrainingSession:
    """A resolved experiment. Callers may edit ``config`` before calling :meth:`run`."""

    config: dict[str, Any]
    kind: TrainerKind | str | None = None
    source: Path | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        kind: TrainerKind | str | None = None,
        overrides: list[str] | tuple[str, ...] | None = None,
    ) -> TrainingSession:
        source = Path(path)
        return cls(
            load_config_with_args(source, list(overrides or ())),
            kind=kind,
            source=source,
        )

    def resolved_kind(self) -> TrainerKind:
        if self.kind is not None:
            return TrainerKind(self.kind)

        configured = self.config.get("trainer")
        if configured is not None:
            return TrainerKind(str(configured))

        candidates = [
            kind
            for kind, section in (
                (TrainerKind.PPO, "ppo"),
                (TrainerKind.BEHAVIOR_CLONING, "bc"),
            )
            if section in self.config
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "cannot infer trainer: set top-level trainer to 'ppo' or 'bc', or pass kind="
            )
        raise ValueError(
            "config contains both ppo and bc sections; set top-level trainer or pass kind="
        )

    def build_trainer(self) -> Trainer:
        return _TRAINERS[self.resolved_kind()](self.config)

    def run(self) -> dict[str, Any]:
        return self.build_trainer().run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one PPO or behavior-cloning experiment.")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--trainer", choices=[kind.value for kind in TrainerKind])
    parser.add_argument("--set", nargs="*", dest="overrides", help="dotted key=value overrides")
    args = parser.parse_args()
    TrainingSession.from_file(
        args.config,
        kind=args.trainer,
        overrides=args.overrides,
    ).run()


if __name__ == "__main__":
    main()
