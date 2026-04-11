from game.functions.utils import make_initial_game_state
from game.types.game import GameState


def create_new_game() -> GameState:
    """
    Python equivalent of TS newGameTransition():
      return { ...initialGameState }
    """
    # We return a fresh initial GameState each time:
    return make_initial_game_state()
