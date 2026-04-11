from __future__ import annotations

from typing import Literal

from sb3_contrib import MaskablePPO

from rl.pg.ppo.utils.dirs import build_checkpoint_dir


def save_model(
    model: MaskablePPO,
    run_name: str,
    benchmark_name: str,
    checkpoint: Literal["best", "last"],
) -> None:
    path = build_checkpoint_dir(run_name, benchmark_name, checkpoint) / "model.zip"
    model.save(str(path))
    print(f"[saved] model checkpoint: {checkpoint} for '{benchmark_name}': {path}")
