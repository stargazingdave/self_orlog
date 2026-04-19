import random
import numpy as np
import optuna
import torch

from game.types.players import PlayerId
from rl.utils.curriculum import MixPoint
from rl.ddqn.config import DDQNConfig
from rl.ddqn.utils.init_model import init_model
from rl.env.config import OpponentName
from rl.env.env import Opponent, OrlogEnv


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(trial: optuna.Trial) -> float:
    print(f"[objective] trial {trial.number} start", flush=True)

    base_seed = 10_000 + trial.number
    seed_everything(base_seed)

    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    n_layers = trial.suggest_int("n_layers", 2, 4)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.96, 0.995)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [100_000, 200_000, 500_000])

    eps_start = trial.suggest_float("eps_start", 0.9, 1.0)
    eps_end = trial.suggest_float("eps_end", 0.02, 0.1)
    eps_decay_steps = trial.suggest_int("eps_decay_steps", 50_000, 150_000, step=25_000)

    target_update_freq = trial.suggest_categorical(
        "target_update_freq", [2_000, 5_000, 10_000]
    )
    learning_starts = trial.suggest_int("learning_starts", 2_000, 10_000, step=2_000)
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4])

    shaping_scale = 0.0
    win_reward = 1.0
    loss_reward = -1.0

    schedule: list[MixPoint] = [
        MixPoint(
            name="0",
            t=0,
            mix={
                OpponentName.Random: 0.10,
                OpponentName.Conservative: 0.20,
                OpponentName.MeleeAggressive: 0.20,
                OpponentName.RangedAggressive: 0.20,
                OpponentName.Aggressive: 0.20,
                OpponentName.HeuristicPressure: 0.10,
            },
        ),
    ]

    def env_ctor(seed: int):
        mix: dict[OpponentName, Opponent] = {
            OpponentName.Random: Opponent(p_mix=0.10, p_action=1.0),
            OpponentName.Conservative: Opponent(p_mix=0.20, p_action=1.0),
            OpponentName.MeleeAggressive: Opponent(p_mix=0.20, p_action=1.0),
            OpponentName.RangedAggressive: Opponent(p_mix=0.20, p_action=1.0),
            OpponentName.Aggressive: Opponent(p_mix=0.20, p_action=1.0),
            OpponentName.HeuristicPressure: Opponent(p_mix=0.10, p_action=1.0),
        }
        env = OrlogEnv(
            seed=seed,
            agent_player_id=PlayerId.P1,
            max_env_steps_per_episode=200,
            opponents=mix,
        )
        if hasattr(env, "shaping_scale"):
            env.shaping_scale = shaping_scale
        if hasattr(env, "win_reward"):
            env.win_reward = win_reward
        if hasattr(env, "loss_reward"):
            env.loss_reward = loss_reward
        return env

    total_env_steps = 150_000
    report_every = 25_000

    prune_eval_seed_blocks = [2000]
    prune_games_per_block = 50

    final_eval_seed_blocks = [2000, 3000, 4000]
    final_games_per_block = 100

    env = env_ctor(base_seed)

    # chunked training state (so you don't reset model each chunk)
    steps_done = 0

    config: DDQNConfig = DDQNConfig(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        target_update_freq=target_update_freq,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_steps=eps_decay_steps,
    )
    model = init_model(env, hyperparams=config)

    while steps_done < total_env_steps:
        chunk = min(report_every, total_env_steps - steps_done)

        model.learn(
            run_name=f"tune/trial_{trial.number}",
            total_env_steps=chunk,
            curriculum_schedule=schedule,
            eval_every=chunk,
        )

        steps_done += chunk

        wr, _ = model.eval_greedy_multi_seed(
            prune_eval_seed_blocks, games_per_block=prune_games_per_block
        )

        print(
            f"[objective] trial {trial.number} report: step={steps_done} wr={wr:.4f}",
            flush=True,
        )

        trial.report(wr, steps_done)
        if trial.should_prune():
            print(
                f"[objective] trial {trial.number} PRUNED at step={steps_done} wr={wr:.4f}",
                flush=True,
            )
            raise optuna.TrialPruned()

    eval_wr, _ = model.eval_greedy_multi_seed(
        final_eval_seed_blocks, games_per_block=final_games_per_block
    )

    print(f"[objective] trial {trial.number} end: wr={eval_wr:.4f}", flush=True)
    return eval_wr
