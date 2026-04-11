from __future__ import annotations

import copy
from pathlib import Path
import random
from typing import Callable, Literal
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from game.types.players import PlayerId
from rl.config import HistoryEntry
from rl.utils.curriculum import (
    MixPoint,
    normalize_mix,
    to_opponent_dict,
    to_weights_dict,
)
from rl.ddqn.config import DDQNConfig
from rl.ddqn.eval.evaluate_fixed_opponents import evaluate_fixed_opponents
from rl.ddqn.utils.write_eval_txt import write_eval_txt
from rl.ddqn.utils.actions import (
    linear_schedule,
    masked_greedy_action,
    masked_random_action,
)
from rl.ddqn.utils.dirs import build_best_for_stage_dir, build_checkpoint_dir
from rl.ddqn.utils.pool import SelfPlayPool
from rl.ddqn.utils.save_model import save_model_pt, save_model_zip
from rl.ddqn.utils.plot import (
    save_all_graphs,
)
from rl.env.actions import FAVOR_SKIP
from rl.env.config import OpponentName
from rl.env.env import Opponent, OrlogEnv
from rl.ddqn.replay_buffer import ReplayBuffer


device = "cuda" if torch.cuda.is_available() else "cpu"


class QNetConfigurable(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _opponents_set_to_mix(
    opponents: set[OpponentName], mix: dict[OpponentName, Opponent] | None = None
) -> dict[OpponentName, Opponent]:
    """
    Creates a mix dictionary from a set of opponents.
    If mix is provided, the values ovverride the uniform distribution for
    existing keys, others are set to 0. If mix is None, a uniform distribution
    over the opponents is returned.
    """
    if not opponents:
        raise ValueError("opponents set must not be empty")

    if mix is None:
        prob = 1.0 / len(opponents)
        return {opp: Opponent(p_mix=prob, p_action=1.0) for opp in opponents}
    else:
        total = sum(
            mix.get(opp, Opponent(p_mix=0.0, p_action=1.0)).p_mix for opp in opponents
        )
        if total == 0.0:
            raise ValueError("mix values for opponents cannot all be zero")
        return {
            opp: Opponent(
                p_mix=(mix.get(opp, Opponent(p_mix=0.0, p_action=1.0)).p_mix / total),
                p_action=1.0,
            )
            for opp in opponents
        }


class DDQN:
    def __init__(
        self,
        env: OrlogEnv,
        config: DDQNConfig,
    ):
        obs, info = env.reset()
        self.obs = obs
        self.info = info

        observation_space = gym.spaces.Box(
            low=0, high=999, shape=obs.shape, dtype=np.int32
        )
        action_space = gym.spaces.Discrete(FAVOR_SKIP + 1)

        obs_shape = getattr(observation_space, "shape", None)
        if obs_shape is None or len(obs_shape) == 0:
            raise ValueError("Could not infer obs_dim from env.observation_space.shape")
        obs_dim = int(obs_shape[0])

        n_actions_attr = getattr(action_space, "n", None)
        if n_actions_attr is None:
            raise ValueError("Could not infer n_actions from env.action_space.n")
        n_actions = int(n_actions_attr)

        q = QNetConfigurable(obs_dim, n_actions, config.hidden_dim, config.n_layers).to(
            device
        )
        q_target = QNetConfigurable(
            obs_dim, n_actions, config.hidden_dim, config.n_layers
        ).to(device)

        self.env = env

        self.q = q
        self.q_target = q_target
        self.q_target.load_state_dict(q.state_dict())
        self.q_target.eval()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.opponent_history: dict[OpponentName, list[HistoryEntry]] = {
            opp: [] for opp in env.opponents.keys()
        }
        self.aggregate_history: list[HistoryEntry] = []

        self.config = config
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.config.lr)
        self.rb = ReplayBuffer(self.config.buffer_size)
        self.total_steps_done = 0
        self.best_score_for_mix = float("-inf")
        self.current_mix_point: MixPoint | None = None

    def save(
        self,
        run_name: str,
        checkpoint: Literal["best", "last"],
        step: int,
        score: float,
        path: Path | str | None = None,
    ):
        if path is None:
            if self.current_mix_point is not None and checkpoint == "best":
                path = build_best_for_stage_dir(run_name, self.current_mix_point.name)
            else:
                path = build_checkpoint_dir(run_name, checkpoint)

        meta = {"step": step, "eval_score": score}
        pt_path = save_model_pt(
            path=path, checkpoint=checkpoint, q=self.q, params=self.config, meta=meta
        )
        saved_model_path = save_model_zip(
            path=path,
            pt_path=pt_path,
            params=self.config,
            meta=meta,
        )
        print(
            f"[saved] model checkpoint: {checkpoint} for '{run_name}': {saved_model_path}"
        )

    def predict(
        self,
        obs: np.ndarray,
        action_mask: list[bool] | np.ndarray | torch.Tensor | None = None,
        deterministic: bool = True,
        eps: float = 0.0,
    ) -> int:
        """
        Predict an action for a single observation.

        Args:
            obs:
                A single observation with shape (obs_dim,).
            action_mask:
                Optional boolean mask of valid actions with shape (n_actions,).
                True means the action is allowed.
                If None, all actions are considered valid.
            deterministic:
                If True, always pick the greedy valid action.
                If False, use epsilon-greedy according to `eps`.
            eps:
                Probability of picking a random valid action when deterministic=False.

        Returns:
            The selected action as an int.
        """
        if action_mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        elif isinstance(action_mask, torch.Tensor):
            mask = action_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(action_mask, dtype=bool)

        if mask.shape != (self.n_actions,):
            raise ValueError(
                f"action_mask must have shape ({self.n_actions},), got {mask.shape}"
            )

        if not mask.any():
            raise ValueError("action_mask does not allow any valid actions")

        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.shape != (self.obs_dim,):
            raise ValueError(
                f"obs must have shape ({self.obs_dim},), got {obs_arr.shape}"
            )

        bool_mask: list[bool] = mask.tolist()

        if (not deterministic) and random.random() < eps:
            return int(masked_random_action(bool_mask))

        was_training = self.q.training
        self.q.eval()

        try:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs_arr, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = self.q(obs_t)[0]
                action = masked_greedy_action(q_values, bool_mask)
            return int(action)
        finally:
            if was_training:
                self.q.train()

    def learn(
        self,
        total_env_steps: int,
        curriculum_schedule: list[MixPoint],
        eval_every: int,
        run_name: str,
        self_play_pool: SelfPlayPool | None = None,
        eval_configure_env: Callable[[OrlogEnv], None] | None = None,
        clone_env_factory: Callable[[], OrlogEnv] | None = None,
    ):
        all_opponent_types = set()
        for point in curriculum_schedule:
            all_opponent_types.update(point.mix.keys())

        schedule = sorted(curriculum_schedule, key=lambda p: p.t)
        self.obs, self.info = self.env.reset()
        self.q.train()

        ep_len = 0

        def _update_mix_point(schedule: list[MixPoint], t: int) -> None:
            if not schedule:
                raise ValueError("schedule must not be empty")

            for point in reversed(schedule):
                if t >= point.t:
                    self.current_mix_point = point
                    return

            self.current_mix_point = schedule[0]

        for _ in range(total_env_steps):
            self.total_steps_done += 1
            step = self.total_steps_done
            # curriculum
            if ep_len == 0:
                prev_mix_point = self.current_mix_point
                _update_mix_point(schedule, step)

                if self.current_mix_point is None:
                    raise RuntimeError(
                        "current_mix_point should not be None after update"
                    )

                if prev_mix_point != self.current_mix_point:
                    self.best_score_for_mix = float("-inf")
                    print(
                        f"[curriculum] step={step} updated mix: {prev_mix_point} -> {self.current_mix_point}"
                    )

                opps = to_opponent_dict(normalize_mix(self.current_mix_point.mix))
                self.env.set_opponents(opps)

            eps = linear_schedule(
                step,
                self.config.eps_start,
                self.config.eps_end,
                self.config.eps_decay_steps,
            )
            mask = list(self.info["action_mask"])
            strict_mask = self.env._action_mask()

            if mask != strict_mask:
                raise RuntimeError(
                    f"Mask mismatch before action selection.\n"
                    f"phase={self.env.state.phase}\n"
                    f"info_true={[i for i, x in enumerate(mask) if x]}\n"
                    f"env_true={[i for i, x in enumerate(strict_mask) if x]}"
                )
            if random.random() < eps:
                action = masked_random_action(mask)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(
                        self.obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action = masked_greedy_action(self.q(s_t)[0], mask)
            next_obs, r, done, trunc, next_info = self.env.step(action)

            if "action_mask" not in next_info:
                raise RuntimeError(
                    f"Env returned no action_mask. phase={self.env.state.phase}"
                )

            terminal = done or trunc

            next_mask = list(next_info["action_mask"])

            self.rb.add(self.obs, action, r, next_obs, terminal, next_mask)
            self.obs, self.info = next_obs, next_info
            ep_len += 1

            if terminal or ep_len >= 200:
                self.obs, self.info = self.env.reset()
                ep_len = 0

            # DDQN learn
            if (
                step >= self.config.learning_starts
                and len(self.rb) >= self.config.batch_size
                and step % self.config.train_freq == 0
            ):
                s, a, r_t, s2, done_t, mask_t = self.rb.sample(self.config.batch_size)
                with torch.no_grad():
                    q_online_next = self.q(s2).masked_fill(~mask_t, -1e9)
                    best_a = q_online_next.argmax(dim=1, keepdim=True)
                    q2 = self.q_target(s2).gather(1, best_a)
                    valid_next = mask_t.any(dim=1, keepdim=True)
                    q2 = torch.where(valid_next, q2, torch.zeros_like(q2))
                    y = r_t + self.config.gamma * (1.0 - done_t) * q2
                loss = torch.nn.functional.smooth_l1_loss(self.q(s).gather(1, a), y)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
                self.opt.step()

            if step % self.config.target_update_freq == 0:
                self.q_target.load_state_dict(self.q.state_dict())

            # Evaluation and checkpointing
            if step % eval_every == 0:
                print(f"\n{'='*60}")
                # Eval per opponent
                eval_opponent_types = set(all_opponent_types)

                if self_play_pool is None or len(self_play_pool) == 0:
                    eval_opponent_types.discard(OpponentName.SelfPlay)

                eval_opponents = _opponents_set_to_mix(
                    eval_opponent_types, self.env.opponents
                )

                results, aggregate = evaluate_fixed_opponents(
                    self.q,
                    step,
                    self.opponent_history,
                    self.aggregate_history,
                    opponents=eval_opponents,
                    masked_greedy_action=masked_greedy_action,
                    configure_env=eval_configure_env,
                )

                save_all_graphs(
                    run_name=run_name,
                    all_opponents=list(self.opponent_history.keys()),
                    opponent_history=self.opponent_history,
                    history=self.aggregate_history,
                )

                # Checkpoint if best
                if aggregate.winrate > self.best_score_for_mix:
                    print(
                        f"[checkpoint] new best: {aggregate.winrate:.4f} (prev {self.best_score_for_mix:.4f})"
                    )

                    # regular rolling "best" checkpoint
                    self.save(
                        run_name,
                        checkpoint="best",
                        step=step,
                        score=aggregate.winrate,
                    )

                    write_eval_txt(
                        base_path=build_best_for_stage_dir(
                            run_name,
                            (
                                self.current_mix_point.name
                                if self.current_mix_point
                                else None
                            ),
                        ),
                        weights=to_weights_dict(eval_opponents),
                        step=step,
                        aggregate=aggregate,
                        opponent_results=results,
                    )

                    # snapshot into self-play pool if it's not the first best for this mix
                    if self_play_pool is not None and self.best_score_for_mix > 0.0:
                        snapshot_path = self_play_pool.make_snapshot_path(
                            step=step,
                            score=aggregate.winrate,
                        )
                        self.save(
                            run_name,
                            checkpoint="best",
                            step=step,
                            score=aggregate.winrate,
                            path=snapshot_path,
                        )
                        if clone_env_factory is None:
                            raise RuntimeError(
                                "clone_env_factory is required when using self_play_pool"
                            )

                        pool_model = self.clone_for_env(clone_env_factory())

                        self_play_pool.add_checkpoint(
                            path=snapshot_path,
                            score=aggregate.winrate,
                            step=step,
                            model=pool_model,
                        )
                        print(f"[self-play pool] added: {snapshot_path}")

                    self.best_score_for_mix = aggregate.winrate
                print(f"{'='*60}\n")

                # Log per-opponent results and aggregate
                print(f"Run {run_name} - Step {step} evaluation:")
                print("mean_return", aggregate.mean_return)
                print("return_variance", aggregate.return_variance)
                print("winrate", aggregate.winrate)
                print(
                    f"winrate_variance",
                    aggregate.winrate_variance,
                )

                print("-" * 60)
                print(
                    f"{'WEIGHTED SCORE':<26} "
                    f"winrate={aggregate.winrate:.3f} "
                    f"return={aggregate.mean_return:.4f}"
                    "\n"
                )

            if step % 5000 == 0:
                print(f"[train] step={step} eps={eps:.3f} buffer={len(self.rb)}")

        aggregate = self.aggregate_history[-1]
        step = self.aggregate_history[-1].step
        self.save(run_name, checkpoint="last", step=step, score=aggregate.winrate)
        return self.q

    def clone_for_env(self, env: OrlogEnv) -> DDQN:
        cloned = DDQN(env, config=copy.deepcopy(self.config))

        cloned.q.load_state_dict(self.q.state_dict())
        cloned.q_target.load_state_dict(self.q_target.state_dict())

        cloned.q.eval()
        cloned.q_target.eval()

        return cloned

    def eval_greedy(
        self,
        n_games: int = 200,
        seed: int = 999,
    ):
        self.q.eval()  # important if you ever add dropout/bn later

        wins = {PlayerId.P1: 0, PlayerId.P2: 0, None: 0}
        returns = []

        for k in range(n_games):
            obs, info = self.env.reset(seed=seed + k)
            done = trunc = False
            ep_ret = 0.0

            while not (done or trunc):
                mask = info["action_mask"]

                # put state tensor on same device as q
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    q_vals = self.q(s_t)[0]
                a = masked_greedy_action(q_vals, mask)

                obs, r, done, trunc, info = self.env.step(int(a))
                ep_ret += float(r)

            wins[self.env.state.winner_player_id] += 1
            returns.append(ep_ret)

        agent_id = self.env.agent_id
        winrate = wins[agent_id] / n_games
        return winrate, float(np.mean(returns)), wins

    def eval_greedy_multi_seed(self, seed_blocks, games_per_block=200):
        wrs = []
        rets = []
        for base in seed_blocks:
            wr, ret, _ = self.eval_greedy(n_games=games_per_block, seed=base)
            wrs.append(wr)
            rets.append(ret)
        return float(np.mean(wrs)), float(np.mean(rets))
