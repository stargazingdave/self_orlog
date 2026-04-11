import copy
from game.types.dice import SymbolCounts
from game.types.game import GamePhase, GameState
from game.types.resolution import CombatModifiers, ResolutionState, ResolutionStep


def resolve_round(state: GameState) -> GameState:
    """
    TS resolveRoundTransition equivalent.

    TS behavior:
      - Only enters RESOLUTION and initializes a blank resolution object.
      - Does NOT award gold tokens, count symbols, or apply pre-combat favors here.
      - Those happen later in advance_resolution_transition during IMMEDIATE_FAVORS
        and only once guarded by resolution.finalized.
    """
    if state.phase != GamePhase.GOD_FAVOR_SELECTION or state.round_meta is None:
        return state

    starting_id = state.round_meta.starting_player_id

    # Clone players & dice (TS clones players + dice shallowly; deepcopy is fine for parity)
    p1 = copy.deepcopy(state.players[0])
    p2 = copy.deepcopy(state.players[1])

    resolution = ResolutionState(
        starting_player_id=starting_id,
        step=ResolutionStep.IMMEDIATE_FAVORS,
        finalized=False,
        counts1=SymbolCounts(),
        counts2=SymbolCounts(),
        mods_p1=CombatModifiers(),
        mods_p2=CombatModifiers(),
    )

    return GameState(
        phase=GamePhase.RESOLUTION,
        players=(p1, p2),
        round_meta=state.round_meta,
        winner_player_id=state.winner_player_id,
        resolution=resolution,
        freyja_reroll=state.freyja_reroll,
    )
