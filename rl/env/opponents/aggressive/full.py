from game.functions.utils import get_player
from game.types.dice import SymbolType
from game.types.game import GameState
from game.types.players import PlayerId
from game.types.randomizer import Randomizer
from rl.env.actions import FAVOR_SKIP, FAVOR_START


def aggressive_roll(s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float):
    player = get_player(s, opp_id)
    mask = 0
    for i, d in enumerate(player.dice):
        # Lock golden, axes, and arrows
        if d.is_golden or d.face in [SymbolType.AXE, SymbolType.ARROW]:
            if rand.random() < p_action:
                mask |= 1 << i
            mask |= 1 << i
    return mask


def aggressive_favor(s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float):
    player = get_player(s, opp_id)
    # Prefer Thor or Ullr
    if player.tokens >= 4:
        # Thor L1 (0), Thor L2 (1), Ullr L1 (6), Ullr L2 (7)
        options = [0, 1, 6, 7]
        if rand.random() < p_action:
            return FAVOR_START + options[rand.randrange(len(options))]
    elif player.tokens >= 2:
        # Ullr L1 (6)
        return FAVOR_START + 6
    return FAVOR_SKIP
