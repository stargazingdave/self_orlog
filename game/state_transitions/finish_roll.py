import copy
from game.functions.utils import other_player_id, player_index_from_id
from game.types.game import GamePhase, GameState, RoundMeta
from game.types.players import PlayerId


def finish_roll(
    state: GameState,
    player_id: PlayerId,
) -> GameState:
    """
    Python equivalent of TS finishRollTransition.

    - Only in ROLLING phase.
    - Only currentPlayerId can finish.
    - 3rd roll is already handled inside roll_dice_transition.
    - Promotes current player's locked dice to perma_locked.
    - Switches turn or advances roll number.
    """
    if state.phase != GamePhase.ROLLING or state.round_meta is None:
        return state

    meta = state.round_meta

    if player_id != meta.current_player_id:
        return state

    # 3rd roll is auto-finished inside ROLL_DICE
    if meta.current_roll_number == 3:
        return state

    p_idx = player_index_from_id(player_id)
    other_id = other_player_id(player_id)

    # clone players & dice, and "freeze" current player's locks
    players_list = []
    for idx, p in enumerate(state.players):
        cloned = copy.deepcopy(p)
        for d in cloned.dice:
            if idx == p_idx:
                # promote locked → perma_locked for current player
                d.perma_locked = d.perma_locked or d.is_locked
            else:
                # ensure it's at least a bool (we already guarantee that in Python)
                d.perma_locked = bool(d.perma_locked)
        players_list.append(cloned)

    players_tuple = tuple(players_list)  # type: ignore[arg-type]

    new_meta = RoundMeta(
        round_number=meta.round_number,
        starting_player_id=meta.starting_player_id,
        current_player_id=meta.current_player_id,
        current_roll_number=meta.current_roll_number,
        has_rolled_current_turn=meta.has_rolled_current_turn,
        god_favors_approved=dict(meta.god_favors_approved),
    )

    if meta.current_roll_number < 3:
        if meta.current_player_id == meta.starting_player_id:
            new_meta.current_player_id = other_id
        else:
            new_meta.current_player_id = meta.starting_player_id
            new_meta.current_roll_number = meta.current_roll_number + 1
        new_meta.has_rolled_current_turn = False

        return GameState(
            phase=state.phase,
            players=players_tuple,
            round_meta=new_meta,
            winner_player_id=state.winner_player_id,
            resolution=state.resolution,
            freyja_reroll=state.freyja_reroll,
        )

    # no extra 3rd-roll logic anymore
    return state
