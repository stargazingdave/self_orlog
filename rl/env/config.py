from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypedDict

from game.types.game import GameState
from game.types.god_favors import GodFavorId
from game.types.players import PlayerId


FAVOR_TABLE = [
    (GodFavorId.THOR_STRIKE, 0),
    (GodFavorId.THOR_STRIKE, 1),
    (GodFavorId.THOR_STRIKE, 2),
    (GodFavorId.FREYJA_PLENTY, 0),
    (GodFavorId.FREYJA_PLENTY, 1),
    (GodFavorId.FREYJA_PLENTY, 2),
    (GodFavorId.ULLR_AIM, 0),
    (GodFavorId.ULLR_AIM, 1),
    (GodFavorId.ULLR_AIM, 2),
]


class OpponentName(str, Enum):
    Random = "random"
    Conservative = "conservative"
    MeleeAggressive = "melee_aggressive"
    RangedAggressive = "ranged_aggressive"
    Aggressive = "aggressive"
    SemiAggressive = "semi_aggressive"
    Defensive = "defensive"
    TempoAggressive = "tempo_aggressive"
    HeuristicPressure = "heuristic_pressure"
    HardMix = "hard_mix"
    SelfPlay = "self_play"
    GoldThor3 = "gold_thor3"
    ArrowUllr = "arrow_ullr"
    ThorBurst = "thor_burst"
    TokenHoarderBurst = "token_hoarder_burst"
    BalancedValueGreedy = "balanced_value_greedy"
    ShieldCounterArcher = "shield_counter_archer"


class OpponentPolicy(TypedDict):
    roll: Callable[[GameState], int]  # returns lock mask
    freyja: Callable[[GameState, int], int]  # returns reroll mask
    favor: Callable[[GameState], int]  # returns favor action index or skip


@dataclass(slots=True)
class OrlogEnvConfig:
    agent_player_id: PlayerId = PlayerId.P1
    max_env_steps_per_episode: int = 200
    debug: bool = False

    shaping_scale: float = 0.10
    win_reward: float = 1.0
    loss_reward: float = -1.0
    token_scale: float = 0.02
    shaping_clip: float = 0.20

    max_auto_advance_iters: int = 2000
    log_every_env_steps: int = 25
