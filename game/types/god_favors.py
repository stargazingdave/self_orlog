from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from .players import PlayerId


class GodFavorId(str, Enum):
    THOR_STRIKE = "THOR_STRIKE"
    FREYJA_PLENTY = "FREYJA_PLENTY"
    ULLR_AIM = "ULLR_AIM"


@dataclass
class GodFavorLevel:
    cost: int  # tokens

    # Only some of these are used depending on favor
    damage: Optional[int] = None  # Thor's Strike
    arrows_ignoring_shields: Optional[int] = None  # Ullr's Aim
    extra_reroll_dice: Optional[int] = None  # Freyja's Plenty


@dataclass
class GodFavorDefinition:
    id: GodFavorId
    name: str
    description: str
    levels: List[GodFavorLevel]


@dataclass
class ChosenGodFavor:
    favor_id: GodFavorId
    level_index: int  # 0-based


@dataclass
class FreyjaRerollState:
    player_id: PlayerId
    max_dice: int
