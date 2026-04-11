from typing import Tuple

from game.functions.utils import get_god_favor_def

from game.types.god_favors import GodFavorId
from game.types.players import PlayerState


def apply_thors_strike(
    player: PlayerState, opponent: PlayerState
) -> Tuple[PlayerState, PlayerState]:
    chosen = player.chosen_god_favor_this_round
    if not chosen or chosen.favor_id != GodFavorId.THOR_STRIKE:
        return player, opponent

    favor_def = get_god_favor_def(chosen.favor_id)
    level = favor_def.levels[chosen.level_index]
    if level is None or (level.damage or 0) <= 0:
        return player, opponent

    if player.tokens < level.cost:
        return player, opponent

    new_player = PlayerState(**vars(player))
    new_opponent = PlayerState(**vars(opponent))

    new_player.tokens -= level.cost
    new_opponent.health -= level.damage or 0

    # Note: we do NOT clear chosen_god_favor_this_round (same as TS)
    return new_player, new_opponent
