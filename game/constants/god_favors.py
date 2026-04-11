from typing import List

from game.types.god_favors import GodFavorDefinition, GodFavorId, GodFavorLevel


GOD_FAVORS: List[GodFavorDefinition] = [
    GodFavorDefinition(
        id=GodFavorId.THOR_STRIKE,
        name="Thor's Strike",
        description="Deal damage to the opponent after the resolution phase.",
        levels=[
            GodFavorLevel(cost=4, damage=2),
            GodFavorLevel(cost=8, damage=5),
            GodFavorLevel(cost=12, damage=8),
        ],
    ),
    GodFavorDefinition(
        id=GodFavorId.FREYJA_PLENTY,
        name="Freyja's Plenty",
        description="Roll additional dice this round.",
        levels=[
            GodFavorLevel(cost=2, extra_reroll_dice=1),
            GodFavorLevel(cost=4, extra_reroll_dice=2),
            GodFavorLevel(cost=6, extra_reroll_dice=3),
        ],
    ),
    GodFavorDefinition(
        id=GodFavorId.ULLR_AIM,
        name="Ullr's Aim",
        description="Arrows will ignore your opponent's shields.",
        levels=[
            GodFavorLevel(cost=2, arrows_ignoring_shields=2),
            GodFavorLevel(cost=3, arrows_ignoring_shields=3),
            GodFavorLevel(cost=4, arrows_ignoring_shields=6),
        ],
    ),
]
