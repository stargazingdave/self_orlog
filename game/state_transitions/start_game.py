import random
from typing import Optional

from game.constants.game import STARTING_HEALTH
from game.functions.utils import create_empty_dice

from game.types.players import PlayerId, PlayerState
from game.types.game import GamePhase, GameState, RoundMeta


def start_game(
    state: GameState,
    player1_name: str,
    player2_name: str,
    rng: Optional[random.Random] = None,
) -> GameState:
    if state.phase != GamePhase.SETUP:
        return state

    if rng is None:
        rng = random.Random()

    starting_player_id: PlayerId = PlayerId.P1 if rng.random() < 0.5 else PlayerId.P2

    players: tuple[PlayerState, PlayerState] = (
        PlayerState(
            id=PlayerId.P1,
            name=player1_name or "Player 1",
            health=STARTING_HEALTH,
            tokens=0,
            dice=create_empty_dice(),
            chosen_god_favor_this_round=None,
        ),
        PlayerState(
            id=PlayerId.P2,
            name=player2_name or "Player 2",
            health=STARTING_HEALTH,
            tokens=0,
            dice=create_empty_dice(),
            chosen_god_favor_this_round=None,
        ),
    )

    round_meta = RoundMeta(
        round_number=1,
        starting_player_id=starting_player_id,
        current_player_id=starting_player_id,
        current_roll_number=1,
        has_rolled_current_turn=False,
        god_favors_approved={
            PlayerId.P1: False,
            PlayerId.P2: False,
        },
    )

    return GameState(
        phase=GamePhase.ROLLING,
        players=players,
        round_meta=round_meta,
        winner_player_id=None,
    )
