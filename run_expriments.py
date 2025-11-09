"""Utility script for running multiple training configurations."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import re


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

BASE_ARGS: dict[str, Any] = {
    "eval": True,
}


class _FlagOnly:
    """Sentinel for CLI switches that do not take a value."""

    __slots__ = ()


FLAG = _FlagOnly()


@dataclass(frozen=True)
class Experiment:
    """Definition of a single training configuration."""

    name: str
    args: dict[str, Any]

    def resolved_args(self, base: dict[str, Any]) -> dict[str, Any]:
        overrides = {**base, **self.args}
        overrides.setdefault(
            "model_path",
            str(PROJECT_ROOT / "output" / "experiments" / slugify(self.name)),
        )

        overrides.setdefault(
            "source_path",
            str(PROJECT_PARENT / "dataset" / "assets" / slugify(self.name)),
        )
        return overrides


def slugify(value: str) -> str:
    """Normalize experiment names so they can be used as folder names."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower() or "run"


def args_to_cli(arg_mapping: dict[str, Any]) -> list[str]:
    """Convert a mapping of CLI parameters into a flat list of strings."""

    cli_args: list[str] = []
    for key, value in arg_mapping.items():
        if value is None:
            continue

        flag = f"--{key}"
        if value is FLAG:
            cli_args.append(flag)
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue

        cli_args.append(flag)

        if isinstance(value, (list, tuple)):
            cli_args.extend(str(v) for v in value)
        else:
            cli_args.append(str(value))

    return cli_args


EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="bicycle",
        args={
            "resolution": 4,
            "max_shapes": 6400000,
            "outdoor": FLAG,
        },
    ),
    Experiment(
        name="garden",
        args={
            "resolution": 4,
            "max_shapes": 5200000,
            "outdoor": FLAG,
        },
    ),
    Experiment(
        name="stump",
        args={
            "resolution": 4,
            "max_shapes": 4750000,
            "outdoor": FLAG,
        },
    ),
    Experiment(
        name="bonsai",
        args={
            "resolution": 2,
            "max_shapes": 3000000,
        },
    ),
    Experiment(
        name="counter",
        args={
            "resolution": 2,
            "max_shapes": 2500000,
        },
    ),
    Experiment(
        name="room",
        args={
            "resolution": 2,
            "max_shapes": 2100000,
        },
    ),
    Experiment(
        name="kitchen",
        args={
            "resolution": 2,
            "max_shapes": 2400000,
        },
    ),
    Experiment(
        name="train",
        args={
            "resolution": 1,
            "max_shapes": 2500000,
            "outdoor": FLAG,
        },
    ),
    Experiment(
        name="truck",
        args={
            "resolution": 1,
            "max_shapes": 2000000,
            "outdoor": FLAG,
        },
    ),
]


def run_experiment(exp: Experiment) -> None:
    merged_args = exp.resolved_args(BASE_ARGS)
    command = [sys.executable, str(TRAIN_SCRIPT), *args_to_cli(merged_args)]

    print(f"\n=== Running experiment: {exp.name} ===")
    print("CLI:", " ".join(command))

    subprocess.run(command, check=True)


def run_all(experiments: Iterable[Experiment]) -> None:
    for exp in experiments:
        run_experiment(exp)


if __name__ == "__main__":
    run_all(EXPERIMENTS)
