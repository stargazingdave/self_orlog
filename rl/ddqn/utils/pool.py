from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.ddqn.model import DDQN
    from rl.env.env import OrlogEnv


@dataclass(frozen=True)
class PoolEntry:
    path: Path
    score: float
    step: int


class SelfPlayPool:
    def __init__(self, pool_dir: Path | str, max_size: int = 8):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        self.max_size = int(max_size)
        self.entries: list[PoolEntry] = []
        self._cache: dict[Path, DDQN] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def add_checkpoint(
        self,
        path: Path | str,
        score: float,
        step: int,
        model: DDQN | None = None,
    ) -> None:
        p = Path(path)

        if any(entry.path == p for entry in self.entries):
            return

        self.entries.append(PoolEntry(path=p, score=float(score), step=int(step)))
        self.entries.sort(key=lambda x: (x.score, x.step), reverse=True)

        if model is not None:
            self._cache[p] = model

        if len(self.entries) > self.max_size:
            removed = self.entries[self.max_size :]
            self.entries = self.entries[: self.max_size]
            for entry in removed:
                self._cache.pop(entry.path, None)

    def make_snapshot_path(self, step: int, score: float) -> Path:
        safe_score = f"{score:.4f}".replace(".", "_")
        return self.pool_dir / f"step_{step:09d}_score_{safe_score}.zip"

    def sample_model(self, env: OrlogEnv) -> DDQN:
        if not self.entries:
            raise RuntimeError("SelfPlayPool is empty")

        entry = random.choice(self.entries)

        model = self._cache.get(entry.path)
        if model is None:
            from rl.ddqn.utils.load_model import load_model_from_path

            model = load_model_from_path(entry.path, env)
            self._cache[entry.path] = model

        return model
