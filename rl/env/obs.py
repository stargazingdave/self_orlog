import numpy as np

from game.functions.utils import make_initial_game_state
from game.types.dice import SymbolType
from game.types.game import GameState
from game.types.god_favors import GodFavorId
from game.types.players import PlayerId, PlayerState

FACE_TO_INT = {
    None: 0,
    SymbolType.AXE: 1,
    SymbolType.ARROW: 2,
    SymbolType.SHIELD: 3,
    SymbolType.HELMET: 4,
    SymbolType.HAND: 5,
}

# Meta
OBS_P1_HP_INDEX = 0
OBS_P2_HP_INDEX = 1
OBS_ROUND_NUMBER_INDEX = 2
OBS_CURRENT_ROLL_INDEX = 3
OBS_HAS_ROLLED_INDEX = 4
OBS_CURRENT_PLAYER_INDEX = 5
OBS_STARTING_PLAYER_INDEX = 6

# Dice blocks: 18 ints per player
OBS_P1_DICE_START = 7
OBS_P1_DICE_END = 25  # exclusive

OBS_P2_DICE_START = 25
OBS_P2_DICE_END = 43  # exclusive

# Tokens
OBS_P1_TOKENS_INDEX = 43
OBS_P2_TOKENS_INDEX = 44

# Favors: 3 ints per player
OBS_P1_FAVOR_START = 45
OBS_P1_FAVOR_END = 48  # exclusive

OBS_P2_FAVOR_START = 48
OBS_P2_FAVOR_END = 51  # exclusive


def _encode_player_dice(p: PlayerState) -> list[int]:
    """
    Encodes dice as 18 ints:
    - faces (6 ints): 0 for empty, 1-5 for Axe/Arrow/Shield/Helmet/Hand
    - perma-locked (6 ints): 0/1 for each die
    - golden (6 ints): 0/1 for each die
    """
    out: list[int] = []

    # faces (6)
    for d in p.dice:
        out.append(FACE_TO_INT.get(d.face, 0))

    # perma-locked (6)
    for d in p.dice:
        out.append(1 if d.perma_locked else 0)

    # golden (6)
    for d in p.dice:
        out.append(1 if getattr(d, "is_golden", False) else 0)

    return out


def _encode_player_favors(p: PlayerState) -> list[int]:
    """
    Encodes chosen god favors as 3 ints:
    [Thor_level, Freyja_level, Ullr_level]

    Each value is:
    - 0 if not selected
    - 1..3 for selected level
    """
    out = [0, 0, 0]

    chosen = p.chosen_god_favor_this_round
    if not chosen:
        return out

    favor_id = chosen.favor_id
    level_index = chosen.level_index  # expected 0..2

    if level_index is None or not (0 <= int(level_index) <= 2):
        return out

    level = int(level_index) + 1  # convert 0..2 -> 1..3

    if favor_id == GodFavorId.THOR_STRIKE:
        out[0] = level
    elif favor_id == GodFavorId.FREYJA_PLENTY:
        out[1] = level
    elif favor_id == GodFavorId.ULLR_AIM:
        out[2] = level

    return out


def obs_from_state(state: GameState) -> np.ndarray:
    """
    Converts a game state to a numpy array observation.

    obs layout:
    - hp (2 ints)
    - round number (1 int)
    - current roll number (1 int)
    - has rolled this turn (1 int, 0/1)
    - current player (1 int, 0 for P1, 1 for P2)
    - starting player (1 int, 0 for P1, 1 for P2)
    - dice (36 ints): 18 for each player (faces, perma-locked, golden)
    - tokens (2 ints)
    - favors (6 ints): 3 for each player [Thor, Freyja, Ullr], each in 0..3

    Total: 51 ints
    """
    p1, p2 = state.players
    meta = state.round_meta

    if meta is None:
        round_number = 0
        current_roll = 0
        has_rolled = 0
        current_player = 0
        starting_player = 0
    else:
        round_number = int(meta.round_number)
        current_roll = int(meta.current_roll_number)
        has_rolled = 1 if meta.has_rolled_current_turn else 0
        current_player = 0 if meta.current_player_id == PlayerId.P1 else 1
        starting_player = 0 if meta.starting_player_id == PlayerId.P1 else 1

    obs: list[int] = [
        int(p1.health),
        int(p2.health),
        round_number,
        current_roll,
        has_rolled,
        current_player,
        starting_player,
    ]

    # dice
    obs += _encode_player_dice(p1)
    obs += _encode_player_dice(p2)

    # tokens
    obs += [int(p1.tokens), int(p2.tokens)]

    # favors
    obs += _encode_player_favors(p1)
    obs += _encode_player_favors(p2)

    return np.array(obs, dtype=np.int32)


def get_observation_for_player(state: GameState, player_id: PlayerId) -> np.ndarray:
    """
    Observation is symmetric and always encoded as (P1 then P2).
    For P2 perspective, swap P1/P2 fields and flip current/starting player bits.
    """
    obs = obs_from_state(state).astype(np.int32, copy=True)

    if player_id == PlayerId.P1:
        return obs

    # swap hp
    obs[OBS_P1_HP_INDEX], obs[OBS_P2_HP_INDEX] = (
        obs[OBS_P2_HP_INDEX],
        obs[OBS_P1_HP_INDEX],
    )

    # flip current_player and starting_player bits
    obs[OBS_CURRENT_PLAYER_INDEX] = 1 - int(obs[OBS_CURRENT_PLAYER_INDEX])
    obs[OBS_STARTING_PLAYER_INDEX] = 1 - int(obs[OBS_STARTING_PLAYER_INDEX])

    # swap dice blocks (18 ints each)
    p1_dice = obs[OBS_P1_DICE_START:OBS_P1_DICE_END].copy()
    p2_dice = obs[OBS_P2_DICE_START:OBS_P2_DICE_END].copy()
    obs[OBS_P1_DICE_START:OBS_P1_DICE_END] = p2_dice
    obs[OBS_P2_DICE_START:OBS_P2_DICE_END] = p1_dice

    # swap tokens
    obs[OBS_P1_TOKENS_INDEX], obs[OBS_P2_TOKENS_INDEX] = (
        obs[OBS_P2_TOKENS_INDEX],
        obs[OBS_P1_TOKENS_INDEX],
    )

    # swap favor blocks (3 ints each)
    p1_fav = obs[OBS_P1_FAVOR_START:OBS_P1_FAVOR_END].copy()
    p2_fav = obs[OBS_P2_FAVOR_START:OBS_P2_FAVOR_END].copy()
    obs[OBS_P1_FAVOR_START:OBS_P1_FAVOR_END] = p2_fav
    obs[OBS_P2_FAVOR_START:OBS_P2_FAVOR_END] = p1_fav

    return obs


_dummy_state = make_initial_game_state()
_dummy_obs = obs_from_state(_dummy_state)

OBS_DIM = len(_dummy_obs)
print(f"Observation dimension: {OBS_DIM}")
