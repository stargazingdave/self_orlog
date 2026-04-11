from __future__ import annotations

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.pg.ppo.utils.dirs import build_checkpoint_dir, build_outputs_dir
from rl.pg.ppo.utils.env_factory import make_env
from rl.utils.curriculum import MixPoint, normalize_mix, to_opponent_dict
from rl.env.config import OpponentName
from rl.pg.ppo.train.callbacks.curriculum import CurriculumCallback
from rl.pg.ppo.train.callbacks.eval import BenchmarkEvalCallback, EvalBenchmark
from rl.pg.ppo.train.callbacks.logs import ProgressCallback
from rl.pg.ppo.utils.save_model import save_model
from rl.pg.ppo.utils.init_model import init_model

MAX_STEPS = 300
TOTAL_TIMESTEPS = 2_500_000
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 300


def main():
    # ---- curriculum schedule (smooth ramps) ----
    schedule = [
        MixPoint(
            name="0",
            t=0,
            mix={
                OpponentName.Conservative: 1.0,
            },
        ),
        MixPoint(
            name="1",
            t=300_000,
            mix={
                OpponentName.Conservative: 0.6,
                OpponentName.SemiAggressive: 0.2,
                OpponentName.TempoAggressive: 0.2,
            },
        ),
        MixPoint(
            name="2",
            t=600_000,
            mix={
                OpponentName.Aggressive: 0.40,
                OpponentName.TempoAggressive: 0.30,
                OpponentName.HeuristicPressure: 0.20,
                OpponentName.SemiAggressive: 0.10,
            },
        ),
    ]

    train_opps = to_opponent_dict(normalize_mix(schedule[0].mix))
    eval_mix = normalize_mix(
        {
            OpponentName.Conservative: 1.0,
        }
    )

    benchmarks: list[EvalBenchmark] = [
        EvalBenchmark(name="baseline", opponents=eval_mix),
    ]

    train_env = DummyVecEnv(
        [
            lambda: make_env(
                seed=0, opponents=train_opps, terminal_only=False, max_steps=MAX_STEPS
            )
        ]
    )

    # ---------------- LR = 2e-4 ----------------
    run_name = "tune"
    benchmark_name = "lr_2e-4"
    base_out = build_outputs_dir(run_name)
    last_dir = build_checkpoint_dir(run_name, benchmark_name, "last")

    model = init_model(train_env, n_steps=2048, learning_rate=2e-4)

    benchmark_eval_cb = BenchmarkEvalCallback(
        run_name=run_name,
        benchmarks=benchmarks,
        outputs_dir=base_out,
        deterministic=True,
        eval_freq=EVAL_FREQ,
        eval_seed=12345,
        max_steps=MAX_STEPS,
        n_eval_episodes=N_EVAL_EPISODES,
    )

    curriculum_cb = CurriculumCallback(
        train_env=train_env,
        schedule=schedule,
        ramp=50_000,
    )

    progress_cb = ProgressCallback(
        log_every=1_000,
    )

    cb = CallbackList(
        [
            curriculum_cb,
            benchmark_eval_cb,
            progress_cb,
        ]
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    last_path = last_dir / "model.zip"
    save_model(model, run_name=run_name, benchmark_name=run_name, checkpoint="last")
    print(f"[saved] {last_path}")

    # ---------------- LR = 1e-4 ----------------
    run_name = "tune"
    benchmark_name = "lr_1e-4"
    base_out = build_outputs_dir(run_name)
    last_dir = build_checkpoint_dir(run_name, benchmark_name, "last")

    model = init_model(train_env, n_steps=2048, learning_rate=1e-4)

    benchmark_eval_cb = BenchmarkEvalCallback(
        run_name=run_name,
        benchmarks=benchmarks,
        outputs_dir=base_out,
        deterministic=True,
        eval_freq=EVAL_FREQ,
        eval_seed=12345,
        max_steps=MAX_STEPS,
        n_eval_episodes=N_EVAL_EPISODES,
    )

    curriculum_cb = CurriculumCallback(
        train_env=train_env,
        schedule=schedule,
        ramp=50_000,
    )

    progress_cb = ProgressCallback(
        log_every=1_000,
    )

    cb = CallbackList(
        [
            curriculum_cb,
            benchmark_eval_cb,
            progress_cb,
        ]
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    last_path = last_dir / "model.zip"
    save_model(model, run_name=run_name, benchmark_name=run_name, checkpoint="last")
    print(f"[saved] {last_path}")


if __name__ == "__main__":
    main()
