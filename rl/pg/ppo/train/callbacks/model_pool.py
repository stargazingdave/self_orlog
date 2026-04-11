from __future__ import annotations

from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from rl.pg.ppo.utils.pool import SelfPlayPool


class PoolBootstrapCallback(BaseCallback):
    """
    Callback to bootstrap the model pool with an initial snapshot.
    """

    def __init__(self, pool: SelfPlayPool):
        super().__init__()
        self.pool = pool
        self._did_bootstrap = False

    def _on_training_start(self) -> None:
        if not isinstance(self.model, MaskablePPO):
            raise TypeError(
                f"PoolBootstrapCallback requires MaskablePPO, got {type(self.model).__name__}"
            )
        if self._did_bootstrap:
            return
        path = self.pool.add_recent_snapshot_from_model(self.model, step=0)
        self._did_bootstrap = True
        print(f"[pool] bootstrap recent snapshot: {path}")

    def _on_step(self) -> bool:
        return True


class PoolSnapshotCallback(BaseCallback):
    """
    Callback to save snapshots of the model to the pool at regular intervals.
    """

    def __init__(self, pool: SelfPlayPool, save_freq: int = 200_000):
        super().__init__()
        self.pool = pool
        self.save_freq = int(save_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        if not isinstance(self.model, MaskablePPO):
            raise TypeError(
                f"PoolSnapshotCallback requires MaskablePPO, got {type(self.model).__name__}"
            )

        path = self.pool.add_recent_snapshot_from_model(
            self.model, step=self.num_timesteps
        )
        print(f"[pool] recent snapshot saved: {path}")
        return True


class BestModelPoolSyncCallback(BaseCallback):
    """
    Callback to synchronize the best model snapshot with the pool.

    Args:
        pool: The self-play pool to synchronize with.
        best_model_path: The path to the best model snapshot to synchronize.
        check_freq: The frequency (in timesteps) to check for updates to the best model snapshot.
    """

    def __init__(
        self,
        pool: SelfPlayPool,
        best_model_path: Path,
        check_freq: int,
    ):
        super().__init__()
        self.pool = pool
        self.best_model_path = Path(best_model_path)
        self.check_freq = int(check_freq)
        self._last_seen_mtime_ns: int | None = None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        if not self.best_model_path.exists():
            return True

        mtime = self.best_model_path.stat().st_mtime_ns
        if self._last_seen_mtime_ns == mtime:
            return True

        target = self.pool.sync_best_snapshot(
            self.best_model_path, step=self.num_timesteps
        )
        self._last_seen_mtime_ns = mtime
        print(f"[pool] synced best snapshot: {target}")
        return True
