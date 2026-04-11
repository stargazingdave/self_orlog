from typing import List, Tuple

from game.constants.dice import DICE_DEFINITIONS, NORMAL_FACES
from game.types.dice import DieState, SymbolCounts, SymbolType
from game.types.randomizer import Randomizer


def roll_die(die_index: int, rand: Randomizer) -> Tuple[SymbolType, bool]:
    r = rand.randrange(6)  # 0..5
    if r < 5:
        return NORMAL_FACES[r], False
    gold_symbol = DICE_DEFINITIONS[die_index].gold_symbol
    return gold_symbol, True


# Helpers
def count_gold_tokens(dice: List[DieState]) -> int:
    return sum(1 for d in dice if d.is_golden)


def count_symbols(dice: List[DieState]) -> SymbolCounts:
    counts = SymbolCounts()
    for d in dice:
        if d.face is None:
            continue
        if d.face == SymbolType.AXE:
            counts.axe += 1
        elif d.face == SymbolType.ARROW:
            counts.arrow += 1
        elif d.face == SymbolType.SHIELD:
            counts.shield += 1
        elif d.face == SymbolType.HELMET:
            counts.helmet += 1
        elif d.face == SymbolType.HAND:
            counts.hand += 1
    return counts
