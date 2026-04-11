from dataclasses import dataclass
from enum import Enum
from .dice import SymbolCounts
from .players import PlayerId


class ResolutionStep(str, Enum):
    IMMEDIATE_FAVORS = "IMMEDIATE_FAVORS"
    P1_AXE = "P1_AXE"
    P1_ARROW = "P1_ARROW"
    P2_AXE = "P2_AXE"
    P2_ARROW = "P2_ARROW"
    P1_HANDS = "P1_HANDS"
    P2_HANDS = "P2_HANDS"
    RESOLUTION_FAVORS = "RESOLUTION_FAVORS"
    P1_THOR = "P1_THOR"
    P2_THOR = "P2_THOR"


@dataclass
class CombatModifiers:
    arrows_ignoring_shields: int = 0


@dataclass
class ResolutionState:
    starting_player_id: PlayerId
    step: ResolutionStep

    # TS has this and relies on it to avoid double-applying “finalization”
    finalized: bool

    # Frozen per-round rolls (filled on finalize)
    counts1: SymbolCounts
    counts2: SymbolCounts

    # Pre-combat modifiers per player (filled on finalize)
    mods_p1: CombatModifiers
    mods_p2: CombatModifiers
