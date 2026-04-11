from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SymbolType(str, Enum):
    AXE = "AXE"
    ARROW = "ARROW"
    SHIELD = "SHIELD"
    HELMET = "HELMET"
    HAND = "HAND"


@dataclass
class DieDefinition:
    index: int  # 0–5
    gold_symbol: SymbolType


@dataclass
class DieState:
    index: int
    face: Optional[SymbolType]  # null in TS → Optional in Python
    is_golden: bool
    is_locked: bool
    perma_locked: bool  # mirrors TS


@dataclass
class SymbolCounts:
    axe: int = 0
    arrow: int = 0
    shield: int = 0
    helmet: int = 0
    hand: int = 0
