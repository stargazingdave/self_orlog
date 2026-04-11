from __future__ import annotations

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.pg.ppo.utils.init_model import init_model
from rl.utils.curriculum import MixPoint, normalize_mix, to_opponent_dict
from rl.env.config import OpponentName
from rl.pg.ppo.train.callbacks.curriculum import CurriculumCallback
from rl.pg.ppo.train.callbacks.eval import BenchmarkEvalCallback, EvalBenchmark
from rl.pg.ppo.train.callbacks.logs import ProgressCallback
from rl.pg.ppo.train.callbacks.model_pool import (
    BestModelPoolSyncCallback,
    PoolBootstrapCallback,
    PoolSnapshotCallback,
)
from rl.pg.ppo.utils.pool import SelfPlayPool
from rl.pg.ppo.utils.dirs import build_checkpoint_dir, build_outputs_dir, build_pool_dir
from rl.pg.ppo.utils.env_factory import make_env
from rl.pg.ppo.utils.save_model import save_model


MAX_STEPS = 300
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 300


def _remove_self_play_from_mix(
    mix: dict[OpponentName, float],
) -> dict[OpponentName, float]:
    """Helper to remove self-play from the mix for the initial curriculum points."""
    if OpponentName.SelfPlay not in mix:
        return mix

    new_mix = {k: v for k, v in mix.items() if k != OpponentName.SelfPlay}
    total = sum(new_mix.values())
    normalized_mix = {k: v / total for k, v in new_mix.items()}
    return normalized_mix


# ---------------- main ----------------
def main():
    run_name = "full_1M"
    pool_benchmark_name = "baseline"
    base_out = build_outputs_dir(run_name)
    last_dir = build_checkpoint_dir(run_name, pool_benchmark_name, "last")
    best_dir = build_checkpoint_dir(run_name, pool_benchmark_name, "best")
    pool_dir = build_pool_dir(run_name)

    selfplay_pool = SelfPlayPool(
        pool_dir,
        seed=0,
        max_recent=6,
        max_historical=10,
        max_best=4,
        bucket_weights={
            "recent": 0.32,
            "historical": 0.33,
            "best": 0.35,
        },
    )

    schedule = [
        MixPoint(
            name="0",
            t=0,
            mix={
                OpponentName.Random: 1.0,
            },
        ),
        MixPoint(
            name="1",
            t=100_000,
            mix={
                OpponentName.Random: 0.1,
                OpponentName.Conservative: 0.5,
                OpponentName.TempoAggressive: 0.4,
            },
        ),
        MixPoint(
            name="2",
            t=200_000,
            mix={
                OpponentName.Random: 0.05,
                OpponentName.Conservative: 0.35,
                OpponentName.TempoAggressive: 0.2,
                OpponentName.MeleeAggressive: 0.2,
                OpponentName.RangedAggressive: 0.2,
            },
        ),
        MixPoint(
            name="3",
            t=300_000,
            mix={
                OpponentName.Random: 0.05,
                OpponentName.Conservative: 0.3,
                OpponentName.TempoAggressive: 0.2,
                OpponentName.MeleeAggressive: 0.2,
                OpponentName.RangedAggressive: 0.2,
                OpponentName.Aggressive: 0.25,
            },
        ),
        MixPoint(
            name="4",
            t=400_000,
            mix={
                OpponentName.Random: 0.05,
                OpponentName.Conservative: 0.2,
                OpponentName.TempoAggressive: 0.05,
                OpponentName.MeleeAggressive: 0.05,
                OpponentName.RangedAggressive: 0.05,
                OpponentName.Aggressive: 0.2,
                OpponentName.HeuristicPressure: 0.1,
                OpponentName.BalancedValueGreedy: 0.05,
                OpponentName.TokenHoarderBurst: 0.05,
            },
        ),
        MixPoint(
            name="5",
            t=500_000,
            mix={
                OpponentName.Random: 0.05,
                OpponentName.Conservative: 0.2,
                OpponentName.TempoAggressive: 0.05,
                OpponentName.MeleeAggressive: 0.05,
                OpponentName.RangedAggressive: 0.05,
                OpponentName.Aggressive: 0.2,
                OpponentName.HeuristicPressure: 0.1,
                OpponentName.BalancedValueGreedy: 0.1,
                OpponentName.TokenHoarderBurst: 0.1,
                OpponentName.GoldThor3: 0.05,
                OpponentName.ThorBurst: 0.05,
            },
        ),
        MixPoint(
            name="6",
            t=600_000,
            mix={
                OpponentName.Random: 0.05,
                OpponentName.Conservative: 0.15,
                OpponentName.TempoAggressive: 0.05,
                OpponentName.MeleeAggressive: 0.05,
                OpponentName.RangedAggressive: 0.05,
                OpponentName.Aggressive: 0.15,
                OpponentName.HeuristicPressure: 0.1,
                OpponentName.BalancedValueGreedy: 0.1,
                OpponentName.TokenHoarderBurst: 0.1,
                OpponentName.GoldThor3: 0.05,
                OpponentName.ThorBurst: 0.05,
                OpponentName.ArrowUllr: 0.05,
                OpponentName.ShieldCounterArcher: 0.05,
            },
        ),
        MixPoint(
            name="7",
            t=700_000,
            mix={
                OpponentName.Conservative: 0.1,
                OpponentName.TempoAggressive: 0.05,
                OpponentName.Aggressive: 0.1,
                OpponentName.HeuristicPressure: 0.1,
                OpponentName.BalancedValueGreedy: 0.1,
                OpponentName.TokenHoarderBurst: 0.1,
                OpponentName.GoldThor3: 0.1,
                OpponentName.ThorBurst: 0.1,
                OpponentName.ArrowUllr: 0.1,
                OpponentName.ShieldCounterArcher: 0.15,
            },
        ),
        MixPoint(
            name="8",
            t=800_000,
            mix={
                OpponentName.Conservative: 0.1,
                OpponentName.TempoAggressive: 0.05,
                OpponentName.Aggressive: 0.1,
                OpponentName.BalancedValueGreedy: 0.1,
                OpponentName.TokenHoarderBurst: 0.1,
                OpponentName.GoldThor3: 0.1,
                OpponentName.ThorBurst: 0.1,
                OpponentName.ArrowUllr: 0.1,
                OpponentName.ShieldCounterArcher: 0.15,
                OpponentName.SelfPlay: 0.1,
            },
        ),
    ]

    train_opps = to_opponent_dict(normalize_mix(schedule[0].mix))

    benchmarks: list[EvalBenchmark] = [
        EvalBenchmark(name=mp.name, opponents=_remove_self_play_from_mix(mp.mix))
        for mp in schedule
    ]

    train_env = DummyVecEnv(
        [
            lambda: make_env(
                seed=0,
                opponents=train_opps,
                terminal_only=False,
                max_steps=MAX_STEPS,
                selfplay_pool=selfplay_pool,
            )
        ]
    )

    model = init_model(train_env)

    pool_bootstrap_cb = PoolBootstrapCallback(selfplay_pool)

    pool_snapshot_cb = PoolSnapshotCallback(
        pool=selfplay_pool,
        save_freq=200_000,
    )

    best_pool_sync_cb = BestModelPoolSyncCallback(
        pool=selfplay_pool,
        best_model_path=best_dir / "model.zip",
        check_freq=EVAL_FREQ,
    )

    curriculum_cb = CurriculumCallback(
        train_env=train_env,
        schedule=schedule,
        ramp=100_000,
    )

    progress_cb = ProgressCallback(
        log_every=5_000,
    )

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

    cb = CallbackList(
        [
            curriculum_cb,
            pool_bootstrap_cb,
            pool_snapshot_cb,
            best_pool_sync_cb,
            benchmark_eval_cb,
            progress_cb,
        ]
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=cb,
        reset_num_timesteps=False,
    )

    last_path = last_dir / "model.zip"
    save_model(model, run_name=run_name, benchmark_name=run_name, checkpoint="last")
    print(f"[saved] {last_path}")


if __name__ == "__main__":
    main()
