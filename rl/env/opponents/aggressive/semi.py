from game.functions.utils import get_player
from game.types.dice import SymbolType
from game.types.game import GameState
from game.types.players import PlayerId
from game.types.randomizer import Randomizer
from rl.env.actions import FAVOR_SKIP, FAVOR_START


def semi_aggressive_roll(
    s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float
):
    player = get_player(s, opp_id)

    # --- knobs (tune to taste) ---
    p_lock_offense = 0.55  # lock AXE/ARROW/GOLDEN only ~half the time
    p_lock_defense = 0.25  # sometimes lock SHIELD/HELMET/HAND
    p_reroll_all = 0.20  # occasionally do something dumb/noisy
    # ----------------------------

    r = rand.random()
    if r < p_reroll_all:
        return 0  # reroll everything

    mask = 0

    # Always lock golden most of the time, but not always
    for i, d in enumerate(player.dice):
        if d.is_golden and rand.random() < 0.80 * p_action:
            mask |= 1 << i

    # Prefer offense, but only probabilistically
    for i, d in enumerate(player.dice):
        if d.face in [SymbolType.AXE, SymbolType.ARROW]:
            if rand.random() < p_lock_offense:
                mask |= 1 << i

    # Occasionally keep defense / hands instead of chasing damage
    for i, d in enumerate(player.dice):
        if d.face in [SymbolType.SHIELD, SymbolType.HELMET, SymbolType.HAND]:
            if rand.random() < p_lock_defense:
                if rand.random() < p_action:  # still make it less likely than offense
                    mask |= 1 << i

    return mask


def semi_aggressive_favor(
    s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float
):
    player = get_player(s, opp_id)

    # --- knobs ---
    p_skip_even_if_can = 0.35  # skip a lot even with tokens
    p_use_when_4plus = 0.55  # don't always fire at 4+
    p_use_when_2plus = 0.35  # rarely fire at 2+
    # -------------

    # Often skip to be weaker / less bursty
    if rand.random() < p_skip_even_if_can:
        return FAVOR_SKIP

    if player.tokens >= 4 and rand.random() < p_use_when_4plus * p_action:
        # Bias toward "less spiky" usage:
        # choose Ullr L1/L2 more often than Thor
        # Thor L1 (0), Thor L2 (1), Ullr L1 (6), Ullr L2 (7)
        options = [6, 7, 6, 7, 0, 1]  # duplicated to weight Ullr
        return FAVOR_START + options[rand.randrange(len(options))]

    if player.tokens >= 2 and rand.random() < p_use_when_2plus * p_action:
        # Sometimes even pick random among low options (still mostly Ullr L1)
        options = [6, 6, 6, 0]  # mostly Ullr L1, occasionally Thor L1
        return FAVOR_START + options[rand.randrange(len(options))]

    return FAVOR_SKIP
