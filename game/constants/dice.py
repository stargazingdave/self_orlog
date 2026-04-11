from typing import List

from game.types.dice import DieDefinition, SymbolType


DICE_DEFINITIONS: List[DieDefinition] = [
    DieDefinition(index=0, gold_symbol=SymbolType.AXE),  # Die 1: golden AXE
    DieDefinition(index=1, gold_symbol=SymbolType.AXE),  # Die 2: golden AXE
    DieDefinition(index=2, gold_symbol=SymbolType.ARROW),  # Die 3: golden ARROW
    DieDefinition(index=3, gold_symbol=SymbolType.SHIELD),  # Die 4: golden SHIELD
    DieDefinition(index=4, gold_symbol=SymbolType.HELMET),  # Die 5: golden HELMET
    DieDefinition(index=5, gold_symbol=SymbolType.HAND),  # Die 6: golden HAND
]

NORMAL_FACES: List[SymbolType] = [
    SymbolType.AXE,
    SymbolType.ARROW,
    SymbolType.SHIELD,
    SymbolType.HELMET,
    SymbolType.HAND,
]
