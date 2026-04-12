from __future__ import annotations

from game.types.players import PlayerId
from rl.ddqn.utils.dirs import (
    build_pool_dir,
)
from rl.ddqn.utils.init_model import init_model
from rl.ddqn.utils.pool import SelfPlayPool
from rl.ddqn.utils.self_play_policy import make_self_play_policy
from rl.env.config import OpponentName
from rl.env.env import OrlogEnv
from rl.utils.curriculum import MixPoint, normalize_mix, to_opponent_dict

MAX_STEPS = 300
TOTAL_ENV_STEPS = 5_000_000
EVAL_FREQ = 25_000


def main():
    run_name = "full_5M"
    pool_dir = build_pool_dir(run_name)

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
            t=500_000,
            mix={
                OpponentName.Random: 0.1,
                OpponentName.Conservative: 0.5,
                OpponentName.TempoAggressive: 0.4,
            },
        ),
        MixPoint(
            name="2",
            t=1_000_000,
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
            t=1_500_000,
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
            t=2_000_000,
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
            t=2_500_000,
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
            t=3_000_000,
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
            t=3_500_000,
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
            t=4_000_000,
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

    def configure_env_for_self_play(env: OrlogEnv) -> None:
        env.register_opponent_policy(
            name=OpponentName.SelfPlay,
            policy_builder=make_self_play_policy(self_play_pool, env),
            p_action=1.0,
        )

    def make_clone_env() -> OrlogEnv:
        clone_env = OrlogEnv(
            seed=999999,
            agent_player_id=PlayerId.P1,
            max_env_steps_per_episode=MAX_STEPS,
            opponents=to_opponent_dict(normalize_mix(schedule[0].mix)),
        )
        configure_env_for_self_play(clone_env)
        return clone_env

    # start mix for train env = schedule[0]
    train_opps = to_opponent_dict(normalize_mix(schedule[0].mix))

    env = OrlogEnv(
        seed=42,
        agent_player_id=PlayerId.P1,
        max_env_steps_per_episode=200,
        opponents=train_opps,
    )

    model = init_model(env)

    self_play_pool = SelfPlayPool(pool_dir=pool_dir, max_size=8)

    configure_env_for_self_play(env)

    model.learn(
        run_name=run_name,
        total_env_steps=TOTAL_ENV_STEPS,
        curriculum_schedule=schedule,
        eval_every=EVAL_FREQ,
        self_play_pool=self_play_pool,
        eval_configure_env=configure_env_for_self_play,
        clone_env_factory=make_clone_env,
    )


if __name__ == "__main__":
    main()
