from __future__ import annotations

import time
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """
    Prints a compact one-line training summary every `log_every` steps.

    Expected info fields from env (same as your diagnostics):
      - ep_steps
      - ep_shaping
      - ep_terminal
      - ep_truncated
    """

    def __init__(self, log_every: int = 50_000):
        super().__init__()
        self.log_every = int(log_every)

        # rolling stats since last print
        self._n_eps = 0
        self._n_trunc = 0
        self._sum_len = 0
        self._sum_shaping = 0.0
        self._sum_terminal = 0.0

        # timing
        self._last_time = time.time()
        self._last_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        # accumulate episode stats
        if infos is not None and dones is not None:
            for info, done in zip(infos, dones):
                if not done or info is None:
                    continue

                self._n_eps += 1
                self._n_trunc += 1 if info.get("ep_truncated") else 0
                self._sum_len += int(info.get("ep_steps", 0))
                self._sum_shaping += float(info.get("ep_shaping", 0.0))
                self._sum_terminal += float(info.get("ep_terminal", 0.0))

        if self.n_calls % self.log_every != 0:
            return True

        # compute averages
        if self._n_eps > 0:
            avg_len = self._sum_len / self._n_eps
            trunc_rate = self._n_trunc / self._n_eps
            avg_shaping = self._sum_shaping / self._n_eps
            avg_terminal = self._sum_terminal / self._n_eps
        else:
            avg_len = 0.0
            trunc_rate = 0.0
            avg_shaping = 0.0
            avg_terminal = 0.0

        # compute fps since last print
        now = time.time()
        dt = now - self._last_time
        steps = self.num_timesteps - self._last_step
        fps = int(steps / dt) if dt > 0 else 0

        # print compact line
        print(
            f"[step {self.num_timesteps}] "
            f"fps={fps} "
            f"eps={self._n_eps} "
            f"len={avg_len:.1f} "
            f"trunc={trunc_rate:.2f} "
            f"shp={avg_shaping:.3f} "
            f"term={avg_terminal:.3f}"
        )

        # reset accumulators
        self._n_eps = 0
        self._n_trunc = 0
        self._sum_len = 0
        self._sum_shaping = 0.0
        self._sum_terminal = 0.0

        self._last_time = now
        self._last_step = self.num_timesteps

        return True
