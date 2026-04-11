from __future__ import annotations
import copy
from dataclasses import dataclass

from game.functions.utils import get_god_favor_def
from game.types.dice import SymbolCounts
from game.types.god_favors import GodFavorId

from game.types.resolution import CombatModifiers
from game.types.players import PlayerState


@dataclass
class PreCombatResult:
    player: PlayerState
    opponent: PlayerState
    mods: CombatModifiers


def apply_pre_combat_favors(
    player: PlayerState, opponent: PlayerState
) -> PreCombatResult:
    """
    Mirrors TS applyPreCombatFavors:

      - Only ULLR_AIM handled here.
      - If not enough tokens, nothing happens.
      - Thor and Freyja handled elsewhere.
      - Returns (player, opponent, mods) even if opponent unchanged.
    """
    new_player = copy.deepcopy(player)
    new_opponent = copy.deepcopy(opponent)

    mods = CombatModifiers(arrows_ignoring_shields=0)

    chosen = new_player.chosen_god_favor_this_round
    if not chosen:
        return PreCombatResult(player=new_player, opponent=new_opponent, mods=mods)

    favor_def = get_god_favor_def(chosen.favor_id)
    level = favor_def.levels[chosen.level_index] if favor_def.levels else None
    if level is None:
        return PreCombatResult(player=new_player, opponent=new_opponent, mods=mods)

    # Only activate if we still have enough tokens at this time
    if new_player.tokens < level.cost:
        return PreCombatResult(player=new_player, opponent=new_opponent, mods=mods)

    if chosen.favor_id == GodFavorId.ULLR_AIM:
        new_player.tokens -= level.cost
        mods.arrows_ignoring_shields += level.arrows_ignoring_shields or 0

    return PreCombatResult(player=new_player, opponent=new_opponent, mods=mods)


def compute_melee_damage(
    attacker: SymbolCounts, defender: SymbolCounts, mods: CombatModifiers
) -> int:
    base = attacker.axe
    blocked = defender.helmet
    return max(0, base - blocked)


def compute_ranged_damage(
    attacker: SymbolCounts, defender: SymbolCounts, mods: CombatModifiers
) -> int:
    arrows = attacker.arrow
    if arrows <= 0:
        return 0

    shields = defender.shield
    ignore = max(mods.arrows_ignoring_shields or 0, 0)

    blocked = min(arrows, shields)
    base_damage = max(arrows - shields, 0)

    # Ullr: some of the blocked arrows also deal damage
    extra_damage = min(ignore, blocked)

    return base_damage + extra_damage
