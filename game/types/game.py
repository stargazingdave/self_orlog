from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .players import PlayerId, PlayerState
from .resolution import ResolutionState
from .god_favors import FreyjaRerollState


class GamePhase(str, Enum):
    SETUP = "SETUP"
    ROLLING = "ROLLING"
    GOD_FAVOR_SELECTION = "GOD_FAVOR_SELECTION"
    FREYJA_REROLL = "FREYJA_REROLL"
    RESOLUTION = "RESOLUTION"
    GAME_OVER = "GAME_OVER"


@dataclass
class RoundMeta:
    round_number: int
    starting_player_id: PlayerId
    current_player_id: PlayerId
    current_roll_number: int  # 1|2|3
    has_rolled_current_turn: bool
    god_favors_approved: dict[PlayerId, bool]


@dataclass
class GameState:
    phase: GamePhase
    players: tuple[PlayerState, PlayerState]
    round_meta: Optional[RoundMeta]
    winner_player_id: Optional[PlayerId]

    resolution: Optional[ResolutionState] = None
    freyja_reroll: Optional[FreyjaRerollState] = None
