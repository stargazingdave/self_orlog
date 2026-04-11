from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .god_favors import ChosenGodFavor
    from .dice import DieState


class PlayerId(int, Enum):
    P1 = 1
    P2 = 2


@dataclass
class PlayerState:
    id: PlayerId
    name: str
    health: int
    tokens: int
    dice: List[DieState]
    chosen_god_favor_this_round: Optional[ChosenGodFavor]
