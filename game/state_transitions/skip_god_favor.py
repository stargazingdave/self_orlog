from dataclasses import replace
from game.state_transitions.resolve_round import resolve_round
from game.types.game import GamePhase, GameState
from game.types.players import PlayerId


def skip_god_favor(state, player_id):
    if state.phase != GamePhase.GOD_FAVOR_SELECTION or state.round_meta is None:
        return state

    meta = state.round_meta
    if meta.god_favors_approved.get(player_id):
        return state

    approved = dict(meta.god_favors_approved)
    approved[player_id] = True

    new_state = GameState(
        phase=state.phase,
        players=state.players,
        round_meta=replace(meta, god_favors_approved=approved),
        winner_player_id=state.winner_player_id,
        resolution=state.resolution,
        freyja_reroll=state.freyja_reroll,
    )

    if approved[PlayerId.P1] and approved[PlayerId.P2]:
        return resolve_round(new_state)

    return new_state
