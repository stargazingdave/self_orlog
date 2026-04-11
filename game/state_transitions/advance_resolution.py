from game.functions.combat import (
    apply_pre_combat_favors,
    compute_melee_damage,
    compute_ranged_damage,
)
from game.functions.dice import count_gold_tokens, count_symbols
from game.functions.god_favors import apply_thors_strike
from game.functions.utils import create_empty_dice, get_god_favor_def, other_player_id
from game.types.dice import SymbolCounts
from game.types.game import GamePhase, GameState, RoundMeta
import copy

from game.types.god_favors import FreyjaRerollState, GodFavorId
from game.types.players import PlayerId
from game.types.resolution import CombatModifiers, ResolutionState, ResolutionStep


def advance_resolution(state: GameState) -> GameState:
    if state.phase != GamePhase.RESOLUTION or state.resolution is None:
        return state

    res = state.resolution
    starting_id = res.starting_player_id
    step = res.step

    counts1 = res.counts1
    counts2 = res.counts2
    mods_p1 = res.mods_p1
    mods_p2 = res.mods_p2

    finalized = res.finalized

    p1 = copy.deepcopy(state.players[0])
    p2 = copy.deepcopy(state.players[1])

    first_id = starting_id
    second_id = other_player_id(first_id)

    def get_counts(pid: PlayerId) -> SymbolCounts:
        return counts1 if pid == PlayerId.P1 else counts2

    def get_mods(pid: PlayerId) -> CombatModifiers:
        return mods_p1 if pid == PlayerId.P1 else mods_p2

    def apply_damage_to(pid: PlayerId, damage: int) -> None:
        if damage <= 0:
            return
        if pid == PlayerId.P1:
            p1.health -= damage
        else:
            p2.health -= damage

    def is_dead(pid: PlayerId) -> bool:
        return (p1.health if pid == PlayerId.P1 else p2.health) <= 0

    next_step = step

    # --- IMMEDIATE_FAVORS (matches TS) ---
    if step == ResolutionStep.IMMEDIATE_FAVORS:

        def try_start_freyja(pid: PlayerId) -> GameState | None:
            nonlocal p1, p2

            player = p1 if pid == PlayerId.P1 else p2
            chosen = player.chosen_god_favor_this_round
            if not chosen or chosen.favor_id != GodFavorId.FREYJA_PLENTY:
                return None

            favor_def = get_god_favor_def(chosen.favor_id)
            level = favor_def.levels[chosen.level_index] if favor_def.levels else None
            if level is None:
                return None

            # Not enough tokens → Freyja fizzles, clear favor (TS does this)
            if player.tokens < level.cost:
                cleared = copy.deepcopy(player)
                cleared.chosen_god_favor_this_round = None
                if pid == PlayerId.P1:
                    p1 = cleared
                else:
                    p2 = cleared
                return None

            max_dice = level.extra_reroll_dice or 0
            if max_dice <= 0:
                cleared = copy.deepcopy(player)
                cleared.chosen_god_favor_this_round = None
                if pid == PlayerId.P1:
                    p1 = cleared
                else:
                    p2 = cleared
                return None

            # Spend tokens now (TS does this before entering FREYJA_REROLL)
            updated = copy.deepcopy(player)
            updated.tokens -= level.cost
            if pid == PlayerId.P1:
                p1 = updated
            else:
                p2 = updated

            # Enter Freyja reroll phase; keep resolution.step == IMMEDIATE_FAVORS
            return GameState(
                phase=GamePhase.FREYJA_REROLL,
                players=(p1, p2),
                round_meta=state.round_meta,
                winner_player_id=state.winner_player_id,
                resolution=state.resolution,  # keep same object/step
                freyja_reroll=FreyjaRerollState(player_id=pid, max_dice=max_dice),
            )

        # Order: starting player first, then the other (TS)
        for pid in (first_id, second_id):
            out = try_start_freyja(pid)
            if out is not None:
                return out

        # No Freyja started
        if not finalized:
            # 1) award gold tokens once (final dice)
            p1.tokens += count_gold_tokens(p1.dice)
            p2.tokens += count_gold_tokens(p2.dice)

            # 2) apply pre-combat favors sequentially (important; TS uses this order)
            pre1 = apply_pre_combat_favors(p1, p2)
            p1 = pre1.player
            p2 = pre1.opponent

            pre2 = apply_pre_combat_favors(p2, p1)
            p2 = pre2.player
            p1 = pre2.opponent

            # 3) compute counts once (final dice)
            final_counts1 = count_symbols(p1.dice)
            final_counts2 = count_symbols(p2.dice)

            new_res = ResolutionState(
                starting_player_id=starting_id,
                step=ResolutionStep.P1_AXE,
                finalized=True,
                counts1=final_counts1,
                counts2=final_counts2,
                mods_p1=pre1.mods,
                mods_p2=pre2.mods,
            )

            return GameState(
                phase=GamePhase.RESOLUTION,
                players=(p1, p2),
                round_meta=state.round_meta,
                winner_player_id=state.winner_player_id,
                resolution=new_res,
                freyja_reroll=state.freyja_reroll,
            )

        # already finalized → just go to combat (TS)
        next_step = ResolutionStep.P1_AXE

    # --- everything else: keep your existing branches, BUT: ---
    # IMPORTANT: when you return GAME_OVER, keep round_meta like TS does.

    if step == ResolutionStep.P1_AXE:
        attacker_id = first_id
        defender_id = second_id

        attacker_counts = get_counts(attacker_id)
        defender_counts = get_counts(defender_id)
        mods = get_mods(attacker_id)

        dmg = compute_melee_damage(attacker_counts, defender_counts, mods)
        apply_damage_to(defender_id, dmg)

        if is_dead(defender_id):
            return GameState(
                phase=GamePhase.GAME_OVER,
                players=(p1, p2),
                winner_player_id=attacker_id,
                resolution=None,
                freyja_reroll=state.freyja_reroll,
                round_meta=state.round_meta,
            )

        next_step = ResolutionStep.P1_ARROW

    if step == ResolutionStep.P1_ARROW:
        attacker_id = first_id
        defender_id = second_id

        attacker_counts = get_counts(attacker_id)
        defender_counts = get_counts(defender_id)
        mods = get_mods(attacker_id)

        dmg = compute_ranged_damage(attacker_counts, defender_counts, mods)
        apply_damage_to(defender_id, dmg)

        if is_dead(defender_id):
            return GameState(
                phase=GamePhase.GAME_OVER,
                players=(p1, p2),
                winner_player_id=attacker_id,
                resolution=None,
                freyja_reroll=state.freyja_reroll,
                round_meta=state.round_meta,
            )

        next_step = ResolutionStep.P2_AXE

    if step == ResolutionStep.P2_AXE:
        attacker_id = second_id
        defender_id = first_id

        attacker_counts = get_counts(attacker_id)
        defender_counts = get_counts(defender_id)
        mods = get_mods(attacker_id)

        dmg = compute_melee_damage(attacker_counts, defender_counts, mods)
        apply_damage_to(defender_id, dmg)

        if is_dead(defender_id):
            return GameState(
                phase=GamePhase.GAME_OVER,
                players=(p1, p2),
                winner_player_id=attacker_id,
                resolution=None,
                freyja_reroll=state.freyja_reroll,
                round_meta=state.round_meta,
            )

        next_step = ResolutionStep.P2_ARROW

    if step == ResolutionStep.P2_ARROW:
        attacker_id = second_id
        defender_id = first_id

        attacker_counts = get_counts(attacker_id)
        defender_counts = get_counts(defender_id)
        mods = get_mods(attacker_id)

        dmg = compute_ranged_damage(attacker_counts, defender_counts, mods)
        apply_damage_to(defender_id, dmg)

        if is_dead(defender_id):
            return GameState(
                phase=GamePhase.GAME_OVER,
                players=(p1, p2),
                winner_player_id=attacker_id,
                resolution=None,
                freyja_reroll=state.freyja_reroll,
                round_meta=state.round_meta,
            )

        next_step = ResolutionStep.P1_HANDS

        # --- HANDS: first player steals ---
    if step == ResolutionStep.P1_HANDS:
        # FIX: counts2 (not PlayerId.P2)
        first_counts = counts1 if first_id == PlayerId.P1 else counts2
        first_hands = first_counts.hand

        if first_hands > 0:
            if first_id == PlayerId.P1:
                stealable = min(first_hands, p2.tokens)
                if stealable > 0:
                    p1.tokens += stealable
                    p2.tokens -= stealable
            else:
                stealable = min(first_hands, p1.tokens)
                if stealable > 0:
                    p2.tokens += stealable
                    p1.tokens -= stealable

        next_step = ResolutionStep.P2_HANDS

    # --- HANDS: second player steals ---
    if step == ResolutionStep.P2_HANDS:
        # FIX: counts2 (not PlayerId.P2)
        second_counts = counts1 if second_id == PlayerId.P1 else counts2
        second_hands = second_counts.hand

        if second_hands > 0:
            if second_id == PlayerId.P1:
                stealable = min(second_hands, p2.tokens)
                if stealable > 0:
                    p1.tokens += stealable
                    p2.tokens -= stealable
            else:
                # FIX: second_hands (not first_hands)
                stealable = min(second_hands, p1.tokens)
                if stealable > 0:
                    p2.tokens += stealable
                    p1.tokens -= stealable

        next_step = ResolutionStep.P1_THOR

    if step == ResolutionStep.P1_THOR:
        attacker_id = first_id
        defender_id = second_id

        if attacker_id == PlayerId.P1:
            updated_attacker, updated_defender = apply_thors_strike(p1, p2)
            p1 = updated_attacker
            p2 = updated_defender
        else:
            updated_attacker, updated_defender = apply_thors_strike(p2, p1)
            p2 = updated_attacker
            p1 = updated_defender

        if is_dead(defender_id):
            return GameState(
                phase=GamePhase.GAME_OVER,
                players=(p1, p2),
                winner_player_id=attacker_id,
                resolution=None,
                freyja_reroll=state.freyja_reroll,
                round_meta=state.round_meta,
            )

        next_step = ResolutionStep.P2_THOR

    if step == ResolutionStep.P2_THOR:
        attacker_id = second_id
        defender_id = first_id

        if attacker_id == PlayerId.P1:
            updated_attacker, updated_defender = apply_thors_strike(p1, p2)
            p1 = updated_attacker
            p2 = updated_defender
        else:
            updated_attacker, updated_defender = apply_thors_strike(p2, p1)
            p2 = updated_attacker
            p1 = updated_defender

        if is_dead(defender_id):
            return GameState(
                phase=GamePhase.GAME_OVER,
                players=(p1, p2),
                winner_player_id=attacker_id,
                resolution=None,
                freyja_reroll=state.freyja_reroll,
                round_meta=state.round_meta,
            )

        # No one died this round -> Next round
        current_round = state.round_meta.round_number if state.round_meta else 1
        next_starting_id = other_player_id(starting_id)

        next_p1 = copy.deepcopy(p1)
        next_p2 = copy.deepcopy(p2)
        next_p1.dice = create_empty_dice()
        next_p2.dice = create_empty_dice()
        next_p1.chosen_god_favor_this_round = None
        next_p2.chosen_god_favor_this_round = None
        next_players = (next_p1, next_p2)

        next_round_meta = RoundMeta(
            round_number=current_round + 1,
            starting_player_id=next_starting_id,
            current_player_id=next_starting_id,
            current_roll_number=1,
            has_rolled_current_turn=False,
            god_favors_approved={
                PlayerId.P1: False,
                PlayerId.P2: False,
            },
        )

        return GameState(
            phase=GamePhase.ROLLING,
            players=next_players,
            round_meta=next_round_meta,
            winner_player_id=None,
            resolution=None,
            freyja_reroll=None,
        )

    # Non-terminal step: advance step + updated players
    new_res = ResolutionState(
        starting_player_id=starting_id,
        step=next_step,
        finalized=finalized,  # preserve finalized flag
        counts1=counts1,
        counts2=counts2,
        mods_p1=mods_p1,
        mods_p2=mods_p2,
    )

    return GameState(
        phase=GamePhase.RESOLUTION,
        players=(p1, p2),
        round_meta=state.round_meta,
        winner_player_id=state.winner_player_id,
        resolution=new_res,
        freyja_reroll=state.freyja_reroll,
    )
