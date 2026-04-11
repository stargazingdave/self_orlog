from __future__ import annotations

from typing import Iterable

from game.types.game import GameState
from game.types.players import PlayerId
from game.types.randomizer import Randomizer
from rl.ddqn.utils.pool import SelfPlayPool
from rl.env.actions import (
    FAVOR_END,
    FAVOR_SKIP,
    FAVOR_START,
    FREYJA_MASK_END,
    FREYJA_MASK_START,
    ROLL_MASK_END,
    ROLL_MASK_START,
    get_action_mask_for_player,
)
from rl.env.config import OpponentPolicy
from rl.env.obs import obs_from_state
from rl.env.opponents.policies import OpponentPolicies


def make_self_play_policy(pool: SelfPlayPool, env):
    def policy_builder(
        rand: Randomizer,
        state: GameState,
        player_id: PlayerId,
        p_action: float,
    ) -> OpponentPolicy:
        def _predict_action_in_allowed(
            cur_state: GameState,
            allowed_actions: Iterable[int],
        ) -> int | None:
            full_mask = list(get_action_mask_for_player(cur_state, player_id))
            filtered_mask = [False] * len(full_mask)

            for a in allowed_actions:
                if 0 <= a < len(full_mask) and full_mask[a]:
                    filtered_mask[a] = True

            if not any(filtered_mask):
                return None

            if len(pool) == 0:
                return None

            model = pool.sample_model(env)
            obs = obs_from_state(cur_state)
            return int(
                model.predict(
                    obs,
                    action_mask=filtered_mask,
                    deterministic=True,
                )
            )

        def roll_policy(cur_state: GameState) -> int:
            action = _predict_action_in_allowed(
                cur_state,
                range(ROLL_MASK_START, ROLL_MASK_END + 1),
            )
            if action is None:
                return OpponentPolicies.random_policy(
                    rand, cur_state, player_id, p_action
                )["roll"](cur_state)
            return int(action)

        def freyja_policy(cur_state: GameState, max_dice: int) -> int:
            action = _predict_action_in_allowed(
                cur_state,
                range(FREYJA_MASK_START, FREYJA_MASK_END + 1),
            )

            if action is None:
                return OpponentPolicies.random_policy(
                    rand, cur_state, player_id, p_action
                )["freyja"](cur_state, max_dice)

            raw = int(action) - FREYJA_MASK_START
            if raw < 0:
                raw = 0

            bits = [i for i in range(6) if (raw >> i) & 1]
            if len(bits) > max_dice:
                raw = 0
                for i in bits[:max_dice]:
                    raw |= 1 << i

            return raw

        def favor_policy(cur_state: GameState) -> int:
            allowed_favor_actions = list(range(FAVOR_START, FAVOR_END + 1))
            allowed_favor_actions.append(FAVOR_SKIP)

            action = _predict_action_in_allowed(
                cur_state,
                allowed_favor_actions,
            )

            if action is None:
                return OpponentPolicies.random_policy(
                    rand, cur_state, player_id, p_action
                )["favor"](cur_state)

            return int(action)

        policy: OpponentPolicy = {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }
        return policy

    return policy_builder
