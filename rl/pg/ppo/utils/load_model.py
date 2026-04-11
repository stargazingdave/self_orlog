from __future__ import annotations

from typing import Literal

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.pg.ppo.utils.dirs import build_checkpoint_dir


def load_model(
    run_name: str,
    benchmark_name: str,
    checkpoint: Literal["best", "last"],
    env: DummyVecEnv,
) -> MaskablePPO:
    path = build_checkpoint_dir(run_name, benchmark_name, checkpoint) / "model.zip"
    return MaskablePPO.load(
        str(path),
        env=env,
        device="auto",
        verbose=0,
    )
