from game.functions.utils import get_god_favor_def, get_player
from game.types.dice import SymbolType
from game.types.game import GameState
from game.types.god_favors import GodFavorId
from game.types.players import PlayerId
from game.types.randomizer import Randomizer
from rl.env.actions import FAVOR_SKIP, FAVOR_START


def ranged_aggressive_roll(
    s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float
):
    player = get_player(s, opp_id)
    mask = 0
    for i, d in enumerate(player.dice):
        # Lock only arrows
        if d.face == SymbolType.ARROW:
            if rand.random() < p_action:
                mask |= 1 << i
    return mask


def ranged_aggressive_favor(
    s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float
):
    player = get_player(s, opp_id)
    # Prefer Ullr
    ullr_favor = get_god_favor_def(GodFavorId.ULLR_AIM)
    l1_offset = 6  # index of Ullr L1 in action layout
    l2_offset = 7  # index of Ullr L2 in action layout
    l3_offset = 8  # index of Ullr L3 in action layout
    if rand.random() < p_action:
        if player.tokens >= ullr_favor.levels[2].cost:
            return FAVOR_START + l3_offset  # Ullr L3
        elif player.tokens >= ullr_favor.levels[1].cost:
            return FAVOR_START + l2_offset  # Ullr L2
        elif player.tokens >= ullr_favor.levels[0].cost:
            return FAVOR_START + l1_offset  # Ullr L1
    return FAVOR_SKIP
