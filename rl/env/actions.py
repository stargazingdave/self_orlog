from game.functions.utils import get_god_favor_def, player_index_from_id
from game.types.game import GamePhase, GameState
from game.types.players import PlayerId
from rl.env.config import FAVOR_TABLE

N_ACTIONS = 138

ROLL_MASK_START = 0
ROLL_MASK_END = 63

FREYJA_MASK_START = 64
FREYJA_MASK_END = 127

FAVOR_START = 128  # 128..136 = 9 choose actions
FAVOR_END = 136
FAVOR_SKIP = 137


def get_action_mask_for_player(state: GameState, player_id: PlayerId) -> list[bool]:
    """
    Full discrete action space mask (0..FAVOR_SKIP) for ANY player.
    This matches OrlogEnv._action_mask() behavior, but using `player_id`
    instead of `self.agent_id`.
    """
    n = FAVOR_SKIP + 1
    mask = [False] * n

    s = state

    # terminal => no actions
    if s.phase == GamePhase.GAME_OVER:
        return mask

    rm = s.round_meta

    # Must be that player's decision point (mirror of _is_agent_decision_point)
    # ROLLING: current_player == player AND has_rolled_current_turn AND roll in (1,2)
    if s.phase == GamePhase.ROLLING and rm is not None:
        if (
            rm.current_player_id == player_id
            and rm.has_rolled_current_turn
            and rm.current_roll_number in (1, 2)
        ):
            for a in range(ROLL_MASK_START, ROLL_MASK_END + 1):
                mask[a] = True
        return mask

    # GOD_FAVOR_SELECTION: player not approved yet
    if s.phase == GamePhase.GOD_FAVOR_SELECTION and rm is not None:
        if not rm.god_favors_approved[player_id]:
            mask[FAVOR_SKIP] = True

            player = state.players[player_index_from_id(player_id)]
            tokens = int(getattr(player, "tokens", 0))
            if tokens <= 0:
                return mask

            # identical affordability check to your _action_mask()
            for i, (favor_id, level_index) in enumerate(FAVOR_TABLE):
                try:
                    level = get_god_favor_def(favor_id).levels[level_index]
                    cost = int(level.cost)
                except Exception:
                    cost = 10**9

                if tokens >= cost:
                    mask[FAVOR_START + i] = True
        return mask

    # FREYJA_REROLL: freyja.player_id == player_id
    if s.phase == GamePhase.FREYJA_REROLL and s.freyja_reroll is not None:
        if s.freyja_reroll.player_id == player_id:
            max_dice = int(getattr(s.freyja_reroll, "max_dice", 0))

            for raw in range(0, 64):
                if max_dice <= 0 or (raw & 0b111111).bit_count() <= max_dice:
                    mask[FREYJA_MASK_START + raw] = True
        return mask

    return mask
