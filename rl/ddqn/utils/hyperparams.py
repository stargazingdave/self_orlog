from __future__ import annotations

import json
from typing import Any
import json
from pathlib import Path

from rl.utils.dirs import ensure_dir


def load_best_hyperparameters() -> dict[str, float | int]:
    path = Path("outputs") / "rl" / "ddqn" / "tune" / "best_hyperparams.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_best_hyperparameters(
    best_params: dict[str, Any], file_name: str = "best_hyperparameters.json"
) -> None:
    path = Path("outputs") / "rl" / "ddqn" / "tune"
    ensure_dir(path)
    with open(path / file_name, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved best hyperparameters to {path / file_name}:")
    print(json.dumps(best_params, indent=4))
