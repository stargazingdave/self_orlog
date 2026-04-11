from typing import List, Optional, Tuple
from dataclasses import replace

from game.constants.dice import DICE_DEFINITIONS
from game.constants.game import STARTING_HEALTH
from game.constants.god_favors import GOD_FAVORS
from game.types.god_favors import GodFavorDefinition, GodFavorId
from game.types.players import PlayerId, PlayerState
from game.types.game import GamePhase, GameState
from game.types.dice import DieState


def create_empty_dice() -> List[DieState]:
    """Equivalent to createEmptyDice() in TS."""
    dice: List[DieState] = []
    for ddef in DICE_DEFINITIONS:
        dice.append(
            DieState(
                index=ddef.index,
                face=None,
                is_golden=False,
                is_locked=False,
                perma_locked=False,
            )
        )
    return dice


def make_initial_game_state() -> GameState:
    """Factory for a fresh initial game state (TS initialGameState)."""
    players: Tuple[PlayerState, PlayerState] = (
        PlayerState(
            id=PlayerId.P1,
            name="",
            health=STARTING_HEALTH,
            tokens=0,
            dice=create_empty_dice(),
            chosen_god_favor_this_round=None,
        ),
        PlayerState(
            id=PlayerId.P2,
            name="",
            health=STARTING_HEALTH,
            tokens=0,
            dice=create_empty_dice(),
            chosen_god_favor_this_round=None,
        ),
    )

    return GameState(
        phase=GamePhase.SETUP,
        players=players,
        round_meta=None,
        winner_player_id=None,
    )


def get_god_favor_def(favor_id: GodFavorId) -> GodFavorDefinition:
    for f in GOD_FAVORS:
        if f.id == favor_id:
            return f
    raise ValueError(f"Unknown favorId: {favor_id}")


def other_player_id(player_id: PlayerId) -> PlayerId:
    return PlayerId.P2 if player_id == PlayerId.P1 else PlayerId.P1


def player_index_from_id(player_id: PlayerId) -> int:
    # P1 -> 0, P2 -> 1
    return 0 if player_id == PlayerId.P1 else 1


def get_player(state: GameState, pid: PlayerId) -> PlayerState:
    return state.players[player_index_from_id(pid)]
