from typing import List

from game.functions.utils import get_player, other_player_id, player_index_from_id
from rl.env.actions import FAVOR_END, FAVOR_SKIP, FAVOR_START
from rl.env.config import FAVOR_TABLE, OpponentPolicy
from game.constants.god_favors import GOD_FAVORS
from game.types.dice import SymbolCounts, SymbolType
from game.types.game import GameState
from game.types.god_favors import GodFavorId
from game.types.players import PlayerId, PlayerState
from game.types.randomizer import Randomizer
from rl.env.opponents.aggressive.full import aggressive_favor, aggressive_roll
from rl.env.opponents.aggressive.melee import (
    melee_aggressive_favor,
    melee_aggressive_roll,
)
from rl.env.opponents.aggressive.ranged import (
    ranged_aggressive_favor,
    ranged_aggressive_roll,
)
from rl.env.opponents.aggressive.semi import semi_aggressive_favor, semi_aggressive_roll
from rl.env.opponents.aggressive.tempo import (
    tempo_aggressive_favor_policy,
    tempo_aggressive_freyja_policy,
    tempo_aggressive_roll_policy,
)


def _favor_action(favor_id: GodFavorId, level_idx_0based: int) -> int:
    for j, (fid, lvl) in enumerate(FAVOR_TABLE):
        if fid == favor_id and lvl == level_idx_0based:
            return FAVOR_START + j
    return FAVOR_SKIP


def _favor_cost(favor_id: GodFavorId, level_idx_0based: int) -> int:
    for gf in GOD_FAVORS:
        if gf.id == favor_id:
            return int(gf.levels[level_idx_0based].cost)
    raise KeyError(f"favor_id not found in GOD_FAVORS: {favor_id}")


def _best_affordable_level(
    tokens: int,
    pid: PlayerId,
    favor_id: GodFavorId,
) -> int:
    for lvl in (2, 1, 0):
        if tokens >= _favor_cost(favor_id, lvl):
            return lvl
    return -1


def _count_faces(player: PlayerState) -> dict[SymbolType, int]:
    out: dict[SymbolType, int] = {
        SymbolType.AXE: 0,
        SymbolType.ARROW: 0,
        SymbolType.HELMET: 0,
        SymbolType.SHIELD: 0,
        SymbolType.HAND: 0,
    }
    for d in player.dice:
        if d.face is None:
            continue
        if d.face in out:
            out[d.face] += 1
    return out


def _choose_best_indices(scored: list[tuple[int, int]], k: int) -> list[int]:
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: max(0, min(k, len(scored)))]]


def sample_k_without_replacement(
    items: List[int], k: int, rand: Randomizer
) -> List[int]:
    """RNG-protocol-safe sampling (no rng.sample dependency)."""
    k = max(0, min(k, len(items)))
    pool = list(items)
    out = []
    for _ in range(k):
        j = rand.randrange(len(pool))
        out.append(pool.pop(j))
    return out


def _random_exact_k_mask(k: int, rand: Randomizer) -> int:
    k = max(0, min(6, int(k)))
    pool = list(range(6))
    chosen: list[int] = []

    for _ in range(k):
        j = rand.randrange(len(pool))
        chosen.append(pool.pop(j))

    m = 0
    for i in chosen:
        m |= 1 << i
    return m


def _mask_to_exact_k_indices(mask: int, k: int) -> list[int]:
    k = max(0, min(6, int(k)))

    ones = [i for i in range(6) if ((mask >> i) & 1) == 1]
    if len(ones) > k:
        return ones[:k]

    if len(ones) < k:
        # fill with lowest indices not already chosen
        for i in range(6):
            if i not in ones:
                ones.append(i)
                if len(ones) == k:
                    break
    return ones


def _random_favor_action_for_player(
    rand: Randomizer, state: GameState, pid: PlayerId
) -> int:
    """
    Cheap default policy:
    - If player has 0 tokens: skip.
    - Else randomly choose among 9 favor/level actions + skip.
    (Engine will naturally fizzle if not enough tokens for a given level.)
    """
    p = get_player(state, pid)
    if int(p.tokens) <= 0:
        return FAVOR_SKIP
    # include skip sometimes so it can “save” tokens
    actions = list(range(FAVOR_START, FAVOR_END + 1)) + [FAVOR_SKIP]
    return actions[rand.randrange(len(actions))]


# --- Opponent Policy Library ---
class OpponentPolicies:
    """Collection of opponent policies for mixed training."""

    @staticmethod
    def random_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """Fully random opponent."""
        return {
            "roll": lambda s: rand.randrange(64),
            "freyja": lambda s, max_dice: _random_exact_k_mask(max_dice, rand),
            "favor": lambda s: _random_favor_action_for_player(rand, state, pid),
        }

    @staticmethod
    def conservative_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """Lock valuable dice (golden/axe/arrow), use cheap favors."""

        other = other_player_id(pid)

        def _should_lock(
            golden: bool,
            face: SymbolType | None,
            shields_to_lock: int,
            helmets_to_lock: int,
        ) -> bool:
            if golden:
                return True
            if face in (SymbolType.AXE, SymbolType.ARROW):
                return True
            if face == SymbolType.HELMET and helmets_to_lock > 0:
                return True
            if face == SymbolType.SHIELD and shields_to_lock > 0:
                return True
            return False

        # Compute counters based on current state (do it inside roll, not once)
        def conservative_roll(s):
            # Read both players from state for THIS decision
            my_player = get_player(state, pid)
            other_player = get_player(state, other)

            other_axes = sum(
                1
                for d in other_player.dice
                if d.face == SymbolType.AXE and d.perma_locked
            )
            other_arrows = sum(
                1
                for d in other_player.dice
                if d.face == SymbolType.ARROW and d.perma_locked
            )

            my_helmets = sum(1 for d in my_player.dice if d.face == SymbolType.HELMET)
            my_shields = sum(1 for d in my_player.dice if d.face == SymbolType.SHIELD)

            shields_to_lock = max(0, other_arrows - my_shields)
            helmets_to_lock = max(0, other_axes - my_helmets)

            mask = 0
            local_shields_to_lock = shields_to_lock
            local_helmets_to_lock = helmets_to_lock

            for i, d in enumerate(my_player.dice):
                if _should_lock(
                    d.is_golden, d.face, local_shields_to_lock, local_helmets_to_lock
                ):
                    if d.face == SymbolType.HELMET and local_helmets_to_lock > 0:
                        local_helmets_to_lock -= 1
                    if d.face == SymbolType.SHIELD and local_shields_to_lock > 0:
                        local_shields_to_lock -= 1
                    if rand.random() < p_action:
                        mask |= 1 << i

            return mask

        def conservative_favor(s):
            my_player = get_player(state, pid)
            if my_player.tokens >= 2:
                choices = [3, 6]
                if my_player.tokens >= 3 and rand.random() < p_action:
                    choices.append(7)
                return FAVOR_START + choices[rand.randrange(len(choices))]
            return FAVOR_SKIP

        return {
            "roll": conservative_roll,
            "freyja": lambda s, max_dice: _random_exact_k_mask(max_dice, rand),
            "favor": conservative_favor,
        }

    @staticmethod
    def melee_aggressive_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """Maximize damage output, prefer offensive favors."""
        opp_id = other_player_id(pid)

        return {
            "roll": lambda s: melee_aggressive_roll(s, opp_id, rand, p_action),
            "freyja": lambda s, max_dice: _random_exact_k_mask(max_dice, rand),
            "favor": lambda s: melee_aggressive_favor(s, opp_id, rand, p_action),
        }

    @staticmethod
    def ranged_aggressive_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """Maximize damage output, prefer offensive favors."""
        opp_id = other_player_id(pid)

        return {
            "roll": lambda s: ranged_aggressive_roll(s, opp_id, rand, p_action),
            "freyja": lambda s, max_dice: _random_exact_k_mask(max_dice, rand),
            "favor": lambda s: ranged_aggressive_favor(s, opp_id, rand, p_action),
        }

    @staticmethod
    def aggressive_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """Maximize damage output, prefer offensive favors."""
        opp_id = other_player_id(pid)

        return {
            "roll": lambda s: aggressive_roll(s, opp_id, rand, p_action),
            "freyja": lambda s, max_dice: _random_exact_k_mask(max_dice, rand),
            "favor": lambda s: aggressive_favor(s, opp_id, rand, p_action),
        }

    @staticmethod
    def semi_aggressive_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Offense-leaning, but intentionally weaker and noisier than aggressive.
        Good as an intermediate ladder opponent.
        """
        opp_id = other_player_id(pid)

        return {
            "roll": lambda s: semi_aggressive_roll(s, opp_id, rand, p_action),
            "freyja": lambda s, max_dice: _random_exact_k_mask(max_dice, rand),
            "favor": lambda s: semi_aggressive_favor(s, opp_id, rand, p_action),
        }

    @staticmethod
    def tempo_aggressive_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Tempo-aggressive opponent:
        - Locks AXE / ARROW immediately
        - Locks GOLD only if no attacks
        - Spends tokens ASAP (Ullr L1 > Thor L1)
        """
        opp_id = other_player_id(pid)

        return {
            "roll": lambda s: tempo_aggressive_roll_policy(s, opp_id, rand, p_action),
            "freyja": lambda s, max_dice: tempo_aggressive_freyja_policy(
                s, opp_id, max_dice, rand, p_action
            ),
            "favor": lambda s: tempo_aggressive_favor_policy(s, opp_id, rand, p_action),
        }

    @staticmethod
    def heuristic_pressure_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Heuristic opponent:
        ROLL:
          - Gold die: 90% lock
          - Axes lock prob depends on #axes: 1->0.7, 2->0.8, 3+->1.0 (per die)
          - Same for arrows
          - If agent has axes/arrows, lock matching defense (helmets/shields) up to agent's attack counts with 90%

        GOD FAVOR SELECTION (favor_policy):
          1) If opp has arrows and agent has shields -> prefer Ullr's Aim
             level depends on overlap min(opp_arrows, agent_shields) and available tokens.
          2) Else, if opp HP <= 5 AND agent has attacks that opp can't currently block (unblocked > 0)
             -> prefer Freyja's Plenty (highest affordable).
          3) Else, if affordable -> Thor's Strike with probabilities by best affordable level:
             L1=0.5, L2=0.8, L3=1.0
          Otherwise -> skip.

        FREYJA REROLL (freyja_policy):
          - Reroll up to max_dice dice that are NOT relevant defense:
              * always reroll HAND
              * reroll SHIELD if agent has no arrows
              * reroll HELMET if agent has no axes
              * avoid rerolling relevant defense: HELMET if agent has axes, SHIELD if agent has arrows
          - respects perma_locked and is_locked (won't reroll them)
        """
        player_id = pid
        other_id = other_player_id(pid)

        # -------------------- constants --------------------
        P_GOLD = 1.00
        P_DEF_MATCH = 1.00

        def p_attack_lock(n: int) -> float:
            if n <= 0:
                return 0.0
            if n == 1:
                return 0.60
            if n == 2:
                return 0.90
            return 1.00  # 3+

        def counts_from_faces(dice) -> SymbolCounts:
            c = SymbolCounts()
            for d in dice:
                if d.face == SymbolType.AXE:
                    c.axe += 1
                elif d.face == SymbolType.ARROW:
                    c.arrow += 1
                elif d.face == SymbolType.SHIELD:
                    c.shield += 1
                elif d.face == SymbolType.HELMET:
                    c.helmet += 1
                elif d.face == SymbolType.HAND:
                    c.hand += 1
            return c

        # -------------------- helpers: favor action mapping --------------------
        def favor_action(favor_id: GodFavorId, level_idx_0based: int) -> int:
            for j, (fid, lvl) in enumerate(FAVOR_TABLE):
                if fid == favor_id and lvl == level_idx_0based:
                    return FAVOR_START + j
            return FAVOR_SKIP

        # -------------------- FULL FIX: use your real GOD_FAVORS cost table --------------------
        def favor_cost(favor_id: GodFavorId, level_idx_0based: int) -> int:
            for gf in GOD_FAVORS:
                if gf.id == favor_id:
                    return int(gf.levels[level_idx_0based].cost)
            raise KeyError(f"favor_id not found in GOD_FAVORS: {favor_id}")

        def best_affordable_level(
            state: GameState, pid: PlayerId, favor_id: GodFavorId
        ) -> int:
            tokens = get_player(state, pid).tokens
            for lvl in (2, 1, 0):
                if tokens >= favor_cost(favor_id, lvl):
                    return lvl
            return -1

        # -------------------- ROLL policy (lock mask) --------------------
        def roll_policy(state: GameState) -> int:
            player = get_player(state, player_id)
            other = get_player(state, other_id)

            other_axes = sum(1 for d in other.dice if d.face == SymbolType.AXE)
            other_arrows = sum(1 for d in other.dice if d.face == SymbolType.ARROW)

            idx_gold = [
                d.index for d in player.dice if d.is_golden and not d.perma_locked
            ]
            idx_axes = [
                d.index
                for d in player.dice
                if d.face == SymbolType.AXE and not d.perma_locked
            ]

            idx_arrows = [
                d.index
                for d in player.dice
                if d.face == SymbolType.ARROW and not d.perma_locked
            ]
            idx_helm = [
                d.index
                for d in player.dice
                if d.face == SymbolType.HELMET and not d.perma_locked
            ]
            idx_shield = [
                d.index
                for d in player.dice
                if d.face == SymbolType.SHIELD and not d.perma_locked
            ]
            lock = [False] * 6
            need_helm = min(other_axes, len(idx_helm))
            need_shld = min(other_arrows, len(idx_shield))

            if need_helm > 0:
                idx_golden_helm = [i for i in idx_helm if player.dice[i].is_golden]
                idx_non_golden_helm = [
                    i for i in idx_helm if not player.dice[i].is_golden
                ]
                for i in idx_golden_helm:
                    if len(idx_golden_helm) <= 0 or need_helm <= 0:
                        break
                    if rand.random() < P_DEF_MATCH:
                        lock[i] = True
                        need_helm -= 1
                        idx_golden_helm.remove(i)
                for i in idx_non_golden_helm:
                    if len(idx_non_golden_helm) <= 0 or need_helm <= 0:
                        break
                    if rand.random() < P_DEF_MATCH:
                        lock[i] = True
                        need_helm -= 1
                        idx_non_golden_helm.remove(i)

            if need_shld > 0:
                idx_golden_shld = [i for i in idx_shield if player.dice[i].is_golden]
                idx_non_golden_shld = [
                    i for i in idx_shield if not player.dice[i].is_golden
                ]
                for i in idx_golden_shld:
                    if len(idx_golden_shld) <= 0 or need_shld <= 0:
                        break
                    if rand.random() < P_DEF_MATCH:
                        lock[i] = True
                        need_shld -= 1
                        idx_golden_shld.remove(i)
                for i in idx_non_golden_shld:
                    if len(idx_non_golden_shld) <= 0 or need_shld <= 0:
                        break
                    if rand.random() < P_DEF_MATCH:
                        lock[i] = True
                        need_shld -= 1
                        idx_non_golden_shld.remove(i)

            # Priority 2: gold
            for i in idx_gold:
                if rand.random() < P_GOLD:
                    lock[i] = True

            # Priority 3: attacks by count
            p_axes = p_attack_lock(len(idx_axes))
            p_arrs = p_attack_lock(len(idx_arrows))

            for i in idx_axes:
                if rand.random() < p_axes:
                    lock[i] = True
            for i in idx_arrows:
                if rand.random() < p_arrs:
                    lock[i] = True

            mask = 0
            for i, v in enumerate(lock):
                if v:
                    mask |= 1 << i
            return mask

        # -------------------- FREYJA reroll policy (reroll mask) --------------------
        def freyja_policy(state: GameState, max_dice: int) -> int:
            player = get_player(state, player_id)
            other = get_player(state, other_id)

            a = counts_from_faces(other.dice)

            keep_helmet = a.axe > 0  # relevant defense vs axes
            keep_shield = a.arrow > 0  # relevant defense vs arrows

            candidates = []
            for d in player.dice:
                score = 0

                # always reroll hands
                if d.face == SymbolType.HAND:
                    score += 100

                # reroll irrelevant defenses
                if d.face == SymbolType.SHIELD and not keep_shield:
                    score += 80
                if d.face == SymbolType.HELMET and not keep_helmet:
                    score += 80

                # avoid rerolling relevant defense
                if d.face == SymbolType.SHIELD and keep_shield:
                    score -= 50
                if d.face == SymbolType.HELMET and keep_helmet:
                    score -= 50

                # mild preference to keep golden (optional)
                if d.is_golden:
                    score -= 20

                candidates.append((score, d.index))

            candidates.sort(reverse=True)
            picks = [
                idx for _, idx in candidates[: max(0, min(max_dice, len(candidates)))]
            ]

            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        # -------------------- FAVOR selection policy (choose/skip) --------------------
        def favor_policy(state: GameState) -> int:
            player = get_player(state, player_id)
            other = get_player(state, other_id)

            o = counts_from_faces(player.dice)
            a = counts_from_faces(other.dice)

            unblocked_axes = max(0, a.axe - o.helmet)
            unblocked_arrows = max(0, a.arrow - o.shield)
            unblocked_total = unblocked_axes + unblocked_arrows

            # 1) Ullr's Aim: opp has arrows AND agent has shields
            # "Prefer using Ullr's Aim according to number of arrows and shields numbers and available tokens."
            if o.arrow > 0 and a.shield > 0:
                lvl_max = best_affordable_level(state, player_id, GodFavorId.ULLR_AIM)
                if lvl_max >= 0:
                    overlap = min(
                        o.arrow, a.shield
                    )  # the interaction that Ullr affects
                    if overlap >= 3:
                        desired = 2
                    elif overlap >= 2:
                        desired = 1
                    else:
                        desired = 0
                    lvl = min(desired, lvl_max)
                    return favor_action(GodFavorId.ULLR_AIM, lvl)

            # 2) Freyja's Plenty: low HP and agent has attacks opp can't block
            if player.health <= 5 and unblocked_total > 0:
                lvl = best_affordable_level(state, player_id, GodFavorId.FREYJA_PLENTY)
                if lvl >= 0:
                    return favor_action(GodFavorId.FREYJA_PLENTY, lvl)

            # 3) Thor's Strike: if enough tokens, choose with probability by best affordable level
            lvl = best_affordable_level(state, player_id, GodFavorId.THOR_STRIKE)
            if lvl >= 0:
                p = [0.5, 0.8, 1.0][lvl]
                if rand.random() < p:
                    return favor_action(GodFavorId.THOR_STRIKE, lvl)

            return FAVOR_SKIP

        return {"roll": roll_policy, "freyja": freyja_policy, "favor": favor_policy}

    @staticmethod
    def gold_thor3_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Narrow specialist:
        - roll: lock only golden dice
        - freyja: reroll any non-gold unlocked dice up to max_dice
        - favor: choose Thor L3 only if affordable, else skip
        """

        def roll_policy(state: GameState) -> int:
            me = get_player(state, pid)
            mask = 0
            for d in me.dice:
                if d.perma_locked:
                    continue
                if d.is_golden and rand.random() < p_action:
                    mask |= 1 << d.index
            return mask

        def freyja_policy(state: GameState, max_dice: int) -> int:
            me = get_player(state, pid)
            candidates: list[int] = []
            for d in me.dice:
                if not d.is_golden:
                    candidates.append(d.index)

            rand.shuffle(candidates)
            picks = candidates[: max(0, min(max_dice, len(candidates)))]
            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        def favor_policy(state: GameState) -> int:
            tokens = get_player(state, pid).tokens
            lvl = _best_affordable_level(tokens, pid, GodFavorId.THOR_STRIKE)
            if lvl == 2:
                return _favor_action(GodFavorId.THOR_STRIKE, 2)
            return FAVOR_SKIP

        return {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }

    @staticmethod
    def arrow_ullr_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Punishes shield-heavy play:
        - roll: prioritize arrows, then gold
        - freyja: reroll non-arrow / non-gold dice
        - favor: if opponent has shields and I have arrows, choose highest affordable Ullr
                otherwise choose highest affordable Thor
        """
        other = other_player_id(pid)

        def roll_policy(state: GameState) -> int:
            me = get_player(state, pid)
            mask = 0
            for d in me.dice:
                if d.perma_locked:
                    continue

                lock = False
                if d.face == SymbolType.ARROW:
                    lock = rand.random() < p_action
                elif d.is_golden:
                    lock = rand.random() < 0.90

                if lock:
                    mask |= 1 << d.index
            return mask

        def freyja_policy(state: GameState, max_dice: int) -> int:
            me = get_player(state, pid)
            candidates: list[int] = []
            for d in me.dice:
                if d.face != SymbolType.ARROW and not d.is_golden:
                    candidates.append(d.index)

            rand.shuffle(candidates)
            picks = candidates[: max(0, min(max_dice, len(candidates)))]
            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        def favor_policy(state: GameState) -> int:
            me = get_player(state, pid)
            opp = get_player(state, other)

            my_counts = _count_faces(me)
            opp_counts = _count_faces(opp)

            if my_counts[SymbolType.ARROW] > 0 and opp_counts[SymbolType.SHIELD] > 0:
                ullr_lvl = _best_affordable_level(
                    get_player(state, pid).tokens, pid, GodFavorId.ULLR_AIM
                )
                if ullr_lvl >= 0:
                    return _favor_action(GodFavorId.ULLR_AIM, ullr_lvl)

            thor_lvl = _best_affordable_level(
                get_player(state, pid).tokens, pid, GodFavorId.THOR_STRIKE
            )
            if thor_lvl >= 0:
                return _favor_action(GodFavorId.THOR_STRIKE, thor_lvl)

            return FAVOR_SKIP

        return {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }

    @staticmethod
    def thor_burst_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        More active Thor specialist:
        - roll: prioritize gold + axes + arrows
        - freyja: reroll helmets / shields / hands first
        - favor: choose highest affordable Thor immediately
        """

        def roll_policy(state: GameState) -> int:
            me = get_player(state, pid)
            mask = 0
            for d in me.dice:
                if d.perma_locked:
                    continue

                lock_prob = 0.0
                if d.is_golden:
                    lock_prob = 0.95
                elif d.face == SymbolType.AXE:
                    lock_prob = 0.85
                elif d.face == SymbolType.ARROW:
                    lock_prob = 0.85

                if rand.random() < lock_prob * p_action:
                    mask |= 1 << d.index
            return mask

        def freyja_policy(state: GameState, max_dice: int) -> int:
            me = get_player(state, pid)

            scored: list[tuple[int, int]] = []
            for d in me.dice:
                score = 0
                if d.face == SymbolType.HAND:
                    score += 100
                elif d.face == SymbolType.SHIELD:
                    score += 80
                elif d.face == SymbolType.HELMET:
                    score += 80
                elif d.face == SymbolType.ARROW:
                    score -= 40
                elif d.face == SymbolType.AXE:
                    score -= 40

                if d.is_golden:
                    score -= 30

                scored.append((score, d.index))

            picks = _choose_best_indices(scored, max_dice)

            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        def favor_policy(state: GameState) -> int:
            thor_lvl = _best_affordable_level(
                get_player(state, pid).tokens, pid, GodFavorId.THOR_STRIKE
            )
            if thor_lvl >= 0:
                return _favor_action(GodFavorId.THOR_STRIKE, thor_lvl)
            return FAVOR_SKIP

        return {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }

    @staticmethod
    def token_hoarder_burst_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Token Hoarder / Big Favor Burst
        - roll: strongly prioritize HAND + gold, keep some arrows/axes secondarily
        - freyja: reroll weak defense first, keep hands/gold
        - favor:
            * save until a strong spend is available
            * prefer Ullr if arrows can punish shields
            * otherwise use Thor only when at least L2 is affordable
        """
        other = other_player_id(pid)

        def roll_policy(state: GameState) -> int:
            me = get_player(state, pid)
            mask = 0

            for d in me.dice:
                if d.perma_locked:
                    continue

                lock_prob = 0.0
                if d.face == SymbolType.HAND:
                    lock_prob = 0.95
                elif d.is_golden:
                    lock_prob = 0.90
                elif d.face == SymbolType.ARROW:
                    lock_prob = 0.45
                elif d.face == SymbolType.AXE:
                    lock_prob = 0.35
                elif d.face in (SymbolType.HELMET, SymbolType.SHIELD):
                    lock_prob = 0.20

                if rand.random() < lock_prob * p_action:
                    mask |= 1 << d.index

            return mask

        def freyja_policy(state: GameState, max_dice: int) -> int:
            me = get_player(state, pid)
            scored: list[tuple[int, int]] = []

            for d in me.dice:
                score = 0
                if d.face == SymbolType.SHIELD:
                    score += 90
                elif d.face == SymbolType.HELMET:
                    score += 85
                elif d.face == SymbolType.AXE:
                    score += 55
                elif d.face == SymbolType.ARROW:
                    score += 25
                elif d.face == SymbolType.HAND:
                    score -= 100

                if d.is_golden:
                    score -= 80

                scored.append((score, d.index))

            picks = _choose_best_indices(scored, max_dice)

            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        def favor_policy(state: GameState) -> int:
            me = get_player(state, pid)
            opp = get_player(state, other)
            my_counts = _count_faces(me)
            opp_counts = _count_faces(opp)

            ullr_lvl = _best_affordable_level(me.tokens, pid, GodFavorId.ULLR_AIM)
            thor_lvl = _best_affordable_level(me.tokens, pid, GodFavorId.THOR_STRIKE)

            # Big delayed Ullr punish
            if (
                ullr_lvl >= 1
                and my_counts[SymbolType.ARROW] >= 2
                and opp_counts[SymbolType.SHIELD] >= 1
            ):
                return _favor_action(GodFavorId.ULLR_AIM, ullr_lvl)

            # Big delayed Thor spend
            if thor_lvl >= 1:
                return _favor_action(GodFavorId.THOR_STRIKE, thor_lvl)

            return FAVOR_SKIP

        return {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }

    @staticmethod
    def balanced_value_greedy_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Balanced Value Greedy
        - context-aware lock policy
        - context-aware Freyja rerolls
        - favor:
            * Ullr if arrows vs shields is strong
            * Thor if immediate damage looks better
            * Freyja as value fallback if affordable and board is weak
        """
        other = other_player_id(pid)

        def _die_keep_score(state: GameState, face: SymbolType, is_golden: bool) -> int:
            me = get_player(state, pid)
            opp = get_player(state, other)
            my_counts = _count_faces(me)
            opp_counts = _count_faces(opp)

            score = 0

            if face == SymbolType.AXE:
                score += 55
                if opp_counts[SymbolType.HELMET] == 0:
                    score += 18
                if opp.health <= 5:
                    score += 15

            elif face == SymbolType.ARROW:
                score += 55
                if opp_counts[SymbolType.SHIELD] == 0:
                    score += 18
                if opp_counts[SymbolType.SHIELD] >= 2:
                    score += 8  # still useful if we may use Ullr

            elif face == SymbolType.HAND:
                score += 45
                if my_counts[SymbolType.HAND] == 0:
                    score += 10

            elif face == SymbolType.HELMET:
                score += 38
                if opp_counts[SymbolType.AXE] >= 2:
                    score += 16

            elif face == SymbolType.SHIELD:
                score += 38
                if opp_counts[SymbolType.ARROW] >= 2:
                    score += 16

            if is_golden:
                score += 20

            return score

        def roll_policy(state: GameState) -> int:
            me = get_player(state, pid)
            mask = 0

            for d in me.dice:
                if d.perma_locked:
                    continue
                if d.face is None:
                    continue

                keep_score = _die_keep_score(state, d.face, d.is_golden)

                # map score into rough keep probability
                prob = 0.0
                if keep_score >= 75:
                    prob = 0.95
                elif keep_score >= 60:
                    prob = 0.85
                elif keep_score >= 45:
                    prob = 0.65
                elif keep_score >= 30:
                    prob = 0.45
                else:
                    prob = 0.20

                if rand.random() < prob * p_action:
                    mask |= 1 << d.index

            return mask

        def freyja_policy(state: GameState, max_dice: int) -> int:
            me = get_player(state, pid)
            scored: list[tuple[int, int]] = []

            for d in me.dice:
                if d.face is None:
                    continue

                reroll_score = 100 - _die_keep_score(state, d.face, d.is_golden)
                scored.append((reroll_score, d.index))

            picks = _choose_best_indices(scored, max_dice)

            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        def favor_policy(state: GameState) -> int:
            me = get_player(state, pid)
            opp = get_player(state, other)

            my_counts = _count_faces(me)
            opp_counts = _count_faces(opp)

            thor_lvl = _best_affordable_level(me.tokens, pid, GodFavorId.THOR_STRIKE)
            ullr_lvl = _best_affordable_level(me.tokens, pid, GodFavorId.ULLR_AIM)
            freyja_lvl = _best_affordable_level(
                me.tokens, pid, GodFavorId.FREYJA_PLENTY
            )

            # Strong arrow punish
            if (
                ullr_lvl >= 0
                and my_counts[SymbolType.ARROW] >= 2
                and opp_counts[SymbolType.SHIELD] >= 1
            ):
                return _favor_action(GodFavorId.ULLR_AIM, ullr_lvl)

            # Good direct damage turn
            if thor_lvl >= 0 and (
                my_counts[SymbolType.AXE] + my_counts[SymbolType.ARROW] >= 3
                or opp.health <= 6
            ):
                return _favor_action(GodFavorId.THOR_STRIKE, thor_lvl)

            # Weak board: invest in reroll value
            if freyja_lvl >= 0 and (
                my_counts[SymbolType.AXE]
                + my_counts[SymbolType.ARROW]
                + my_counts[SymbolType.HAND]
                <= 1
            ):
                return _favor_action(GodFavorId.FREYJA_PLENTY, freyja_lvl)

            # fallback
            if thor_lvl >= 0:
                return _favor_action(GodFavorId.THOR_STRIKE, thor_lvl)
            if ullr_lvl >= 0 and my_counts[SymbolType.ARROW] >= 1:
                return _favor_action(GodFavorId.ULLR_AIM, ullr_lvl)

            return FAVOR_SKIP

        return {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }

    @staticmethod
    def shield_counter_archer_policy(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        """
        Shield-Counter Archer
        - roll: strongly prefer arrows, then gold, then a little hand
        - freyja: reroll everything that isn't arrow/gold
        - favor: aggressively use Ullr when shields exist, otherwise Thor fallback
        """
        other = other_player_id(pid)

        def roll_policy(state: GameState) -> int:
            me = get_player(state, pid)
            mask = 0

            for d in me.dice:
                if d.perma_locked:
                    continue

                lock_prob = 0.0
                if d.face == SymbolType.ARROW:
                    lock_prob = 0.98
                elif d.is_golden:
                    lock_prob = 0.92
                elif d.face == SymbolType.HAND:
                    lock_prob = 0.42
                elif d.face == SymbolType.AXE:
                    lock_prob = 0.25
                else:
                    lock_prob = 0.10

                if rand.random() < lock_prob * p_action:
                    mask |= 1 << d.index

            return mask

        def freyja_policy(state: GameState, max_dice: int) -> int:
            me = get_player(state, pid)
            scored: list[tuple[int, int]] = []

            for d in me.dice:
                if d.perma_locked or d.is_locked:
                    continue

                score = 0
                if d.face == SymbolType.SHIELD:
                    score += 95
                elif d.face == SymbolType.HELMET:
                    score += 90
                elif d.face == SymbolType.AXE:
                    score += 65
                elif d.face == SymbolType.HAND:
                    score += 35
                elif d.face == SymbolType.ARROW:
                    score -= 100

                if d.is_golden:
                    score -= 80

                scored.append((score, d.index))

            picks = _choose_best_indices(scored, max_dice)

            mask = 0
            for i in picks:
                mask |= 1 << i
            return mask

        def favor_policy(state: GameState) -> int:
            me = get_player(state, pid)
            opp = get_player(state, other)

            my_counts = _count_faces(me)
            opp_counts = _count_faces(opp)

            ullr_lvl = _best_affordable_level(me.tokens, pid, GodFavorId.ULLR_AIM)
            thor_lvl = _best_affordable_level(me.tokens, pid, GodFavorId.THOR_STRIKE)

            if (
                ullr_lvl >= 0
                and my_counts[SymbolType.ARROW] >= 1
                and opp_counts[SymbolType.SHIELD] >= 1
            ):
                return _favor_action(GodFavorId.ULLR_AIM, ullr_lvl)

            if (
                thor_lvl >= 0
                and my_counts[SymbolType.ARROW] + my_counts[SymbolType.AXE] >= 3
            ):
                return _favor_action(GodFavorId.THOR_STRIKE, thor_lvl)

            if ullr_lvl >= 0 and my_counts[SymbolType.ARROW] >= 2:
                return _favor_action(GodFavorId.ULLR_AIM, ullr_lvl)

            return FAVOR_SKIP

        return {
            "roll": roll_policy,
            "freyja": freyja_policy,
            "favor": favor_policy,
        }
