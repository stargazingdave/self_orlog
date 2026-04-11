from __future__ import annotations

import json
from pathlib import Path


def get_best_hyperparams() -> dict[str, float | int]:
    path = Path("outputs") / "rl" / "pg" / "ppo" / "tune" / "best_hyperparams.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
