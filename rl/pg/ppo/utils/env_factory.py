from __future__ import annotations

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np

from game.types.players import PlayerId
from rl.env.config import OpponentName
from rl.env.env import Opponent, OrlogEnv
from rl.pg.ppo.utils.pool import SelfPlayPool, build_pool_selfplay_policy


def mask_fn(env: gym.Env) -> np.ndarray:
    cur = env
    while True:
        if isinstance(cur, OrlogEnv):
            return cur.get_action_mask()
        if isinstance(cur, gym.Wrapper):
            cur = cur.env
            continue
        break
    raise TypeError(f"mask_fn: OrlogEnv not found, got {type(env).__name__}")


def make_base_env(
    seed: int,
    opponents: dict[OpponentName, Opponent],
    terminal_only: bool,
    max_steps: int,
    selfplay_pool: SelfPlayPool | None = None,
) -> OrlogEnv:
    env = OrlogEnv(
        seed=seed,
        opponents=opponents,
        agent_player_id=PlayerId.P1,
        max_env_steps_per_episode=max_steps,
    )

    if selfplay_pool is not None:
        env.register_opponent_policy(
            OpponentName.SelfPlay,
            build_pool_selfplay_policy(selfplay_pool),
        )

    if terminal_only:
        env.use_terminal_only_reward = True
        env.shaping_scale = 0.0
        env.token_scale = 0.0
        if hasattr(env, "truncation_penalty"):
            env.truncation_penalty = 0.0
        env.max_env_steps_per_episode = max_steps
    else:
        env.use_terminal_only_reward = False
        env.shaping_scale = 0.05
        env.token_scale = 0.02
        if hasattr(env, "truncation_penalty"):
            env.truncation_penalty = -0.2

    return env


def make_env(
    seed: int,
    opponents: dict[OpponentName, Opponent],
    terminal_only: bool,
    max_steps: int,
    selfplay_pool: SelfPlayPool | None = None,
) -> ActionMasker:
    base = make_base_env(
        seed=seed,
        opponents=opponents,
        terminal_only=terminal_only,
        max_steps=max_steps,
        selfplay_pool=selfplay_pool,
    )
    base = Monitor(base)
    return ActionMasker(base, mask_fn)
