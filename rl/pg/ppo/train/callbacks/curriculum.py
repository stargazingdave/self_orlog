from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from rl.utils.curriculum import MixPoint, lerp_mix, normalize_mix, to_opponent_dict
from rl.env.config import OpponentName
from rl.env.env import OrlogEnv


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        train_env: DummyVecEnv,
        schedule: list[MixPoint],
        ramp: int = 100_000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.train_env = train_env
        self.schedule = sorted(schedule, key=lambda x: x.t)
        self.ramp = int(ramp)
        self._last_applied: dict[OpponentName, float] | None = None

    def _get_base(self) -> OrlogEnv:
        e = self.train_env.envs[0]
        cur = e
        while isinstance(cur, gym.Wrapper):
            cur = cur.env
        assert isinstance(cur, OrlogEnv)
        return cur

    def _mix_at(self, t: int) -> dict[OpponentName, float]:
        if len(self.schedule) == 1:
            return normalize_mix(self.schedule[0].mix)

        pts = self.schedule
        if t <= pts[0].t:
            return normalize_mix(pts[0].mix)
        if t >= pts[-1].t + self.ramp:
            return normalize_mix(pts[-1].mix)

        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            if t < b.t:
                return normalize_mix(a.mix)
            if b.t <= t <= b.t + self.ramp:
                alpha = (t - b.t) / float(self.ramp)
                return lerp_mix(normalize_mix(a.mix), normalize_mix(b.mix), alpha)

        return normalize_mix(pts[-1].mix)

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        mix = self._mix_at(t)

        if self._last_applied == mix:
            return True

        base = self._get_base()
        base.set_opponents(to_opponent_dict(mix, p_action=1.0))
        self._last_applied = mix

        if self.verbose:
            pretty = ", ".join(f"{k.name}:{v:.2f}" for k, v in mix.items())
            print(f"[curriculum] step={t} mix={{{pretty}}}")

        return True
