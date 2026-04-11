from __future__ import annotations

import json
import json
from pathlib import Path

from rl.ddqn.config import DDQNConfig
from rl.ddqn.model import DDQN
from rl.env.env import OrlogEnv


def get_best_hyperparams() -> dict[str, float | int]:
    path = (
        Path("outputs")
        / "rl"
        / "ddqn"
        / "tune"
        / "best_hyperparameters_20260403_204742.json"
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_model(
    env: OrlogEnv,
    hyperparams: DDQNConfig | None = None,
) -> DDQN:
    chosen_hyperparams = None

    if hyperparams is not None:
        chosen_hyperparams = hyperparams
    else:
        best_hyperparams = get_best_hyperparams()
        chosen_hyperparams = DDQNConfig(
            hidden_dim=int(best_hyperparams.get("hidden_dim", 64)),
            n_layers=int(best_hyperparams.get("n_layers", 2)),
            lr=float(best_hyperparams.get("lr", 2.9380279387035334e-05)),
            gamma=float(best_hyperparams.get("gamma", 0.965459808211767)),
            batch_size=int(best_hyperparams.get("batch_size", 256)),
            buffer_size=int(best_hyperparams.get("buffer_size", 500000)),
            eps_start=float(best_hyperparams.get("eps_start", 0.9832442640800422)),
            eps_end=float(best_hyperparams.get("eps_end", 0.03698712885426209)),
            eps_decay_steps=int(best_hyperparams.get("eps_decay_steps", 50000)),
            target_update_freq=int(best_hyperparams.get("target_update_freq", 10000)),
            learning_starts=int(best_hyperparams.get("learning_starts", 6000)),
            train_freq=int(best_hyperparams.get("train_freq", 2)),
        )

    return DDQN(env=env, config=chosen_hyperparams)
