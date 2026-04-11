import copy
from dataclasses import replace
from game.functions.utils import player_index_from_id
from game.state_transitions.resolve_round import resolve_round
from game.types.game import GamePhase, GameState
from game.types.god_favors import ChosenGodFavor
from game.types.players import PlayerId, PlayerState


def choose_god_favor(state: GameState, player_id, favor_id, level_index):
    if state.phase != GamePhase.GOD_FAVOR_SELECTION or state.round_meta is None:
        return state

    meta = state.round_meta
    if meta.god_favors_approved.get(player_id):
        return state  # already chose

    players = [copy.deepcopy(p) for p in state.players]
    idx = player_index_from_id(player_id)
    players[idx].chosen_god_favor_this_round = ChosenGodFavor(
        favor_id=favor_id,
        level_index=level_index,
    )

    approved = dict(meta.god_favors_approved)
    approved[player_id] = True

    new_players: tuple[PlayerState, PlayerState] = (players[0], players[1])

    new_state = GameState(
        phase=state.phase,
        players=new_players,
        round_meta=replace(meta, god_favors_approved=approved),
        winner_player_id=state.winner_player_id,
        resolution=state.resolution,
        freyja_reroll=state.freyja_reroll,
    )

    if approved[PlayerId.P1] and approved[PlayerId.P2]:
        return resolve_round(new_state)

    return new_state
