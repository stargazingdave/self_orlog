from game.functions.utils import get_player
from game.types.dice import SymbolType
from game.types.game import GameState
from game.types.players import PlayerId
from game.types.randomizer import Randomizer
from rl.env.actions import FAVOR_SKIP, FAVOR_START


def tempo_aggressive_roll_policy(
    s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float
) -> int:
    opp = get_player(s, opp_id)
    mask = 0
    has_attack = False

    for i, d in enumerate(opp.dice):
        if d.face in (SymbolType.AXE, SymbolType.ARROW):
            if rand.random() < p_action:
                mask |= 1 << i
            has_attack = True

    if not has_attack:
        for i, d in enumerate(opp.dice):
            if d.is_golden:
                if rand.random() < p_action:
                    mask |= 1 << i

    return mask


def tempo_aggressive_freyja_policy(
    s: GameState, opp_id: PlayerId, max_dice: int, rand: Randomizer, p_action: float
) -> int:
    opp = get_player(s, opp_id)
    candidates = []

    for d in opp.dice:
        if d.face not in (SymbolType.AXE, SymbolType.ARROW):
            candidates.append(d.index)

    if candidates:
        candidates = rand.sample(candidates, len(candidates))
    picks = candidates[:max_dice]

    mask = 0
    for i in picks:
        mask |= 1 << i
    return mask


def tempo_aggressive_favor_policy(
    s: GameState, opp_id: PlayerId, rand: Randomizer, p_action: float
) -> int:
    opp = get_player(s, opp_id)
    tokens = opp.tokens

    if tokens >= 2 and rand.random() < p_action:
        return FAVOR_START + 6  # Ullr L1
    if tokens >= 4 and rand.random() < p_action:
        return FAVOR_START + 0  # Thor L1

    return FAVOR_SKIP
