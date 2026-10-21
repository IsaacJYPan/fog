"""Run PPO or behavior cloning from the CLI or a Python training session."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from fog.config import load_config_with_args


class TrainerKind(StrEnum):
    PPO = "ppo"
    BEHAVIOR_CLONING = "bc"


@dataclass(slots=True)
class TrainingSession:
    """An experiment config that can be edited before calling ``run()``."""

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

        candidates = [kind for kind in TrainerKind if kind.value in self.config]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "cannot infer trainer: set top-level trainer to 'ppo' or 'bc', or pass kind="
            )
        raise ValueError(
            "config contains both ppo and bc sections; set top-level trainer or pass kind="
        )

    def run(self) -> dict[str, Any]:
        if self.resolved_kind() is TrainerKind.PPO:
            from fog.training.ppo import train
        else:
            from fog.training.bc import train

        return train(self.config)


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
