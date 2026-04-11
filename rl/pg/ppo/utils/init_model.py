from __future__ import annotations

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.pg.ppo.utils.hyperparams import get_best_hyperparams


def init_model(
    env: DummyVecEnv,
    n_steps: int | None = None,
    learning_rate: float | None = None,
) -> MaskablePPO:
    if n_steps is None or learning_rate is None:
        best_hyperparams = get_best_hyperparams()
        best_n_steps = best_hyperparams.get("n_steps")
        best_learning_rate = best_hyperparams.get("learning_rate")

        if best_n_steps is None or best_learning_rate is None:
            raise ValueError(
                "Best hyperparameters not found in the JSON file: "
                f"{best_hyperparams}"
            )

        if not isinstance(best_n_steps, int) or not isinstance(
            best_learning_rate, (float, int)
        ):
            raise ValueError(
                "Invalid hyperparameter types in the JSON file: "
                f"n_steps={best_n_steps} (type={type(best_n_steps).__name__}), "
                f"learning_rate={best_learning_rate} (type={type(best_learning_rate).__name__})"
            )

        n_steps = n_steps or best_n_steps
        learning_rate = learning_rate or best_learning_rate

    if n_steps is None or learning_rate is None:
        raise ValueError(
            "Missing hyperparameters: n_steps and learning_rate are required"
        )

    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
        batch_size=256,
        learning_rate=learning_rate,
        gamma=0.99,
        ent_coef=0.02,
    )
