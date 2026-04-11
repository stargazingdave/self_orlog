from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl.utils.dirs import ensure_dir


def build_outputs_dir(run_name: str) -> Path:
    return ensure_dir(Path("outputs") / "rl" / "pg" / "ppo" / run_name)


def build_checkpoint_dir(
    run_name: str,
    benchmark_name: str,
    checkpoint: Literal["best", "last"],
) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / benchmark_name / checkpoint)


def build_graphs_dir(run_name: str) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / "graphs")


def build_histories_dir(run_name: str) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / "histories")


def build_pool_dir(run_name: str) -> Path:
    return ensure_dir(build_outputs_dir(run_name) / "pool")
