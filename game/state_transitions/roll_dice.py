import copy
from dataclasses import replace

from game.functions.dice import roll_die
from game.functions.utils import other_player_id, player_index_from_id
from game.types.resolution import ResolutionStep
from game.types.randomizer import Randomizer
from game.types.players import PlayerId
from game.types.game import GamePhase, GameState


def roll_dice(
    state: GameState,
    player_id: PlayerId,
    rand: Randomizer,
) -> GameState:
    """
    Python equivalent of TS rollDiceTransition(state, action).

    Handles:
      - FREYJA_REROLL special mode
      - Normal rolling in ROLLING phase
    """

    # --- Freyja reroll mode ---
    if state.phase == GamePhase.FREYJA_REROLL:
        freyja = state.freyja_reroll
        if freyja is None or freyja.player_id != player_id:
            return state

        max_dice = freyja.max_dice

        players_list = [copy.deepcopy(p) for p in state.players]
        p_idx = player_index_from_id(player_id)
        player = players_list[p_idx]

        unlocked = [d for d in player.dice if not d.is_locked]
        if len(unlocked) != max_dice:
            return state

        for die in player.dice:
            if not die.is_locked:
                face, is_golden = roll_die(die.index, rand)
                die.face = face
                die.is_golden = is_golden
            die.is_locked = True

        player.chosen_god_favor_this_round = None
        players_list[p_idx] = player

        resolution = (
            replace(state.resolution, step=ResolutionStep.IMMEDIATE_FAVORS)
            if state.resolution is not None
            else None
        )

        return GameState(
            phase=GamePhase.RESOLUTION,
            players=tuple(players_list),  # type: ignore[arg-type]
            round_meta=state.round_meta,
            winner_player_id=state.winner_player_id,
            resolution=resolution,
            freyja_reroll=None,
        )

    # --- Normal rolling mode ---
    if state.phase != GamePhase.ROLLING or state.round_meta is None:
        return state

    meta = state.round_meta

    if player_id != meta.current_player_id:
        return state

    if meta.has_rolled_current_turn:
        return state

    players_list = [copy.deepcopy(p) for p in state.players]
    p_idx = player_index_from_id(player_id)
    player = players_list[p_idx]

    for die in player.dice:
        if not die.is_locked:
            face, is_golden = roll_die(die.index, rand)
            die.face = face
            die.is_golden = is_golden

    if meta.current_roll_number == 3:
        for die in player.dice:
            die.is_locked = True

        other_id = other_player_id(player_id)
        other_idx = player_index_from_id(other_id)

        both_finished_third = all(
            d.is_locked for d in players_list[p_idx].dice
        ) and all(d.is_locked for d in players_list[other_idx].dice)

        if not both_finished_third:
            new_meta = replace(
                meta,
                current_player_id=other_id,
                has_rolled_current_turn=False,
            )
            return GameState(
                phase=state.phase,
                players=tuple(players_list),  # type: ignore[arg-type]
                round_meta=new_meta,
                winner_player_id=state.winner_player_id,
                resolution=state.resolution,
                freyja_reroll=state.freyja_reroll,
            )

        new_meta = replace(
            meta,
            current_player_id=meta.starting_player_id,
            has_rolled_current_turn=False,
            god_favors_approved={
                PlayerId.P1: False,
                PlayerId.P2: False,
            },
        )

        return GameState(
            phase=GamePhase.GOD_FAVOR_SELECTION,
            players=tuple(players_list),  # type: ignore[arg-type]
            round_meta=new_meta,
            winner_player_id=state.winner_player_id,
            resolution=state.resolution,
            freyja_reroll=state.freyja_reroll,
        )

    new_meta = replace(meta, has_rolled_current_turn=True)

    return GameState(
        phase=state.phase,
        players=tuple(players_list),  # type: ignore[arg-type]
        round_meta=new_meta,
        winner_player_id=state.winner_player_id,
        resolution=state.resolution,
        freyja_reroll=state.freyja_reroll,
    )
