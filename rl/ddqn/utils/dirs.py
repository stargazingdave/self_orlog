from __future__ import annotations

from typing import Literal
from pathlib import Path

from rl.utils.dirs import ensure_dir


def build_outputs_dir(run_name: str) -> Path:
    return ensure_dir(Path("outputs") / "rl" / "ddqn" / run_name)


def build_checkpoint_dir(
    run_name: str,
    checkpoint: Literal["best", "last"],
) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / checkpoint)


def build_best_for_stage_dir(run_name: str, stage_name: str | None) -> Path:
    return ensure_dir(
        build_checkpoint_dir(run_name, "best") / stage_name
        if stage_name
        else "unnamed_stage"
    )


def build_graphs_dir(run_name: str) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / "graphs")


def build_histories_dir(run_name: str) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / "histories")


def build_pool_dir(run_name: str) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / "pool")
