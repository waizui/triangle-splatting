"""Utility script for running multiple training configurations.
"""

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
            "souce_path",
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
        args={},
    ),
    Experiment(
        name="garden",
        args={},
    ),
    Experiment(
        name="stump",
        args={},
    ),
    Experiment(
        name="bonsai",
        args={},
    ),
    Experiment(
        name="counter",
        args={},
    ),
    Experiment(
        name="room",
        args={},
    ),
    Experiment(
        name="kitchen",
        args={},
    ),
    Experiment(
        name="train",
        args={},
    ),
    Experiment(
        name="truck",
        args={},
    ),
    Experiment(
        name="drjohnson",
        args={},
    ),
    Experiment(
        name="playroom",
        args={},
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
