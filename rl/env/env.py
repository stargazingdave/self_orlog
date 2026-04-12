import random
import copy
import numpy as np
import gymnasium as gym
from rl.env.actions import (
    FAVOR_END,
    FAVOR_SKIP,
    FAVOR_START,
    FREYJA_MASK_END,
    FREYJA_MASK_START,
    ROLL_MASK_END,
    ROLL_MASK_START,
    get_action_mask_for_player,
)
from rl.env.obs import obs_from_state
from game.functions.utils import (
    get_god_favor_def,
    get_player,
    other_player_id,
)
from game.state_transitions.advance_resolution import advance_resolution
from game.state_transitions.choose_god_favor import choose_god_favor
from game.state_transitions.finish_roll import finish_roll
from game.state_transitions.create_new_game import create_new_game
from game.state_transitions.roll_dice import roll_dice
from game.state_transitions.skip_god_favor import skip_god_favor
from game.state_transitions.start_game import start_game
from game.types.game import GamePhase, GameState
from game.types.players import PlayerId, PlayerState
from game.types.randomizer import Randomizer
from rl.env.config import *
from rl.env.opponents.policies import OpponentPolicies


@dataclass
class Opponent:
    p_mix: float
    p_action: float


@dataclass
class OpponentConfig:
    policy: Callable[[Randomizer, GameState, PlayerId, float], OpponentPolicy]
    p_action: float


class OrlogEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int,
        opponents: dict[OpponentName, Opponent],
        agent_player_id: PlayerId = PlayerId.P1,
        max_env_steps_per_episode: int = 200,
        debug: bool = False,
    ):
        super().__init__()
        self.use_terminal_only_reward = False  # toggle for diagnostics
        self.truncation_penalty = -0.2  # used when truncated and not terminated
        self.debug = bool(debug)
        self.agent_id = agent_player_id
        self.opp_id = other_player_id(agent_player_id)
        self.rand: Randomizer = Randomizer(random.Random(seed), seed)

        self.state: GameState = create_new_game()

        dummy = obs_from_state(self.state)
        self.observation_space = gym.spaces.Box(
            low=0, high=999, shape=dummy.shape, dtype=np.int32
        )

        # 0..137 (138 actions)
        self.action_space = gym.spaces.Discrete(FAVOR_SKIP + 1)

        # rewards
        self.shaping_scale = 0.10
        self.win_reward = 1.0
        self.loss_reward = -1.0

        # token shaping + clipping
        self.token_scale = 0.02  # start here (try 0.01..0.04)
        self.shaping_clip = 0.20  # cap per-step shaping so it can't dominate

        # episode control
        self.max_env_steps_per_episode = int(max_env_steps_per_episode)
        self._env_steps = 0
        # episode control
        self.max_env_steps_per_episode = int(max_env_steps_per_episode)
        self.max_auto_advance_iters = 2000
        self._env_steps = 0
        self.log_every_env_steps = 25
        self.log_round_resolves = False

        # ---- opponent policies ----
        # rolling lock mask policy: callable(state)->int 0..63
        self.opp_roll_policy = lambda s: self.rand.randrange(64)

        # freyja mask policy: callable(state, max_dice)->int 0..63 (bit=1 means reroll)
        self.opp_freyja_policy = lambda s, max_dice: self._random_exact_k_mask(max_dice)

        # god favor policy: callable(state)->action int in [FAVOR_START..FAVOR_SKIP]
        self.opp_favor_policy = lambda s: self._random_favor_action_for_player(
            self.opp_id
        )

        self.opponent_policies: dict[OpponentName, OpponentConfig] = {
            OpponentName.Random: OpponentConfig(
                policy=OpponentPolicies.random_policy, p_action=1.0
            ),
            OpponentName.Conservative: OpponentConfig(
                policy=OpponentPolicies.conservative_policy, p_action=1.0
            ),
            OpponentName.MeleeAggressive: OpponentConfig(
                policy=OpponentPolicies.melee_aggressive_policy, p_action=1.0
            ),
            OpponentName.RangedAggressive: OpponentConfig(
                policy=OpponentPolicies.ranged_aggressive_policy, p_action=1.0
            ),
            OpponentName.Aggressive: OpponentConfig(
                policy=OpponentPolicies.aggressive_policy, p_action=1.0
            ),
            OpponentName.SemiAggressive: OpponentConfig(
                policy=OpponentPolicies.semi_aggressive_policy, p_action=1.0
            ),
            OpponentName.TempoAggressive: OpponentConfig(
                policy=OpponentPolicies.tempo_aggressive_policy, p_action=1.0
            ),
            OpponentName.HeuristicPressure: OpponentConfig(
                policy=OpponentPolicies.heuristic_pressure_policy, p_action=1.0
            ),
            OpponentName.GoldThor3: OpponentConfig(
                policy=OpponentPolicies.gold_thor3_policy, p_action=1.0
            ),
            OpponentName.ArrowUllr: OpponentConfig(
                policy=OpponentPolicies.arrow_ullr_policy, p_action=1.0
            ),
            OpponentName.ThorBurst: OpponentConfig(
                policy=OpponentPolicies.thor_burst_policy, p_action=1.0
            ),
            OpponentName.TokenHoarderBurst: OpponentConfig(
                policy=OpponentPolicies.token_hoarder_burst_policy, p_action=1.0
            ),
            OpponentName.BalancedValueGreedy: OpponentConfig(
                policy=OpponentPolicies.balanced_value_greedy_policy, p_action=1.0
            ),
            OpponentName.ShieldCounterArcher: OpponentConfig(
                policy=OpponentPolicies.shield_counter_archer_policy, p_action=1.0
            ),
        }

        self.current_opponent_name = None

        # set + normalize mix (use the argument!)
        self.opponents = opponents
        total = sum([cfg.p_mix for cfg in self.opponents.values()])
        if total <= 0:
            raise ValueError("opponents total weight must be > 0")
        # normalize mix weights to sum to 1
        self.opponents = {
            k: Opponent(p_mix=v.p_mix / total, p_action=v.p_action)
            for k, v in self.opponents.items()
        }

    # ---------------- gym API ----------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        for name in self.opponents:
            if name not in self.opponent_policies:
                raise RuntimeError(
                    f"Opponent '{name}' is in mix but not registered. "
                    f"Did you forget to pass selfplay_pool?"
                )

        # Select opponent policy for this episode
        policy = self._select_opponent_policy()
        self.opp_roll_policy = policy["roll"]
        self.opp_freyja_policy = policy["freyja"]
        self.opp_favor_policy = policy["favor"]

        super().reset(seed=seed)

        if seed is not None:
            self.rand = Randomizer(random.Random(seed), seed)

        self.state = create_new_game()
        self.state = start_game(
            self.state,
            player1_name="Agent",
            player2_name="Opponent",
        )

        self._env_steps = 0
        self._ep_shaping = 0.0
        self._ep_terminal = 0.0
        self._ep_steps = 0
        self._ep_hp0 = (
            int(self.state.players[0].health),
            int(self.state.players[1].health),
        )
        self._ep_tok0 = (
            int(self.state.players[0].tokens),
            int(self.state.players[1].tokens),
        )

        self._auto_advance()

        # ---- invariant: after auto-advance we must be terminal or have at least 1 valid action
        m = self._action_mask()

        if self.debug:
            self._debug_print_mask_ranges(m, label=f"[reset] phase={self.state.phase}")

        if self.state.phase != GamePhase.GAME_OVER and (not any(m)):
            meta = self.state.round_meta
            raise RuntimeError(
                "RESET STALL: non-terminal state has no valid actions. "
                f"phase={self.state.phase} "
                f"cp={getattr(meta,'current_player_id',None)} "
                f"roll={getattr(meta,'current_roll_number',None)} "
                f"has_rolled={getattr(meta,'has_rolled_current_turn',None)}"
            )

        if self.debug:
            sp = (
                self.state.round_meta.starting_player_id
                if self.state.round_meta
                else None
            )
            print(
                f"[reset] HP P1={self.state.players[0].health} "
                f"P2={self.state.players[1].health} start_player={sp}"
            )

        info = {}
        info["action_mask"] = (
            self._action_mask()
        )  # mask for the *current* state after auto-advance
        return obs_from_state(self.state), info

    def step(self, action: int):
        self._env_steps += 1
        self._ep_steps += 1

        prev_p1 = self.state.players[0].health
        prev_p2 = self.state.players[1].health

        prev_t1 = self.state.players[0].tokens
        prev_t2 = self.state.players[1].tokens

        reward = 0.0
        info = {}

        phase = self.state.phase
        atype = self._action_type(int(action))

        if phase == GamePhase.GOD_FAVOR_SELECTION:
            if atype != "FAVOR_SELECT":
                raise RuntimeError(
                    f"Illegal action for GOD_FAVOR_SELECTION: a={action} type={atype}"
                )

        elif phase == GamePhase.ROLLING:
            if atype != "LOCK_MASK":
                raise RuntimeError(
                    f"Illegal action for ROLLING: a={action} type={atype}"
                )

        elif phase == GamePhase.FREYJA_REROLL:
            if atype != "FREYJA_MASK":
                raise RuntimeError(
                    f"Illegal action for FREYJA_REROLL: a={action} type={atype}"
                )

        else:
            raise RuntimeError(
                f"step() called in non-decision phase: phase={phase} a={action} type={atype}"
            )

        # Always include a mask in the returned info (mask corresponds to the returned state)
        def _finish_step(obs, reward, terminated, truncated):
            m = self._action_mask()

            # If episode ended, never return an all-false mask (maskable policies safety)
            if (terminated or truncated) and not any(m):
                # ensure at least one valid action to avoid downstream issues
                m[0] = True

            info["action_mask"] = m

            # Invariant: if not done, must have at least 1 valid action
            if (not terminated) and (not truncated) and (not any(m)):
                meta = self.state.round_meta
                raise RuntimeError(
                    "STEP STALL: non-terminal state has no valid actions. "
                    f"phase={self.state.phase} "
                    f"cp={getattr(meta,'current_player_id',None)} "
                    f"roll={getattr(meta,'current_roll_number',None)} "
                    f"has_rolled={getattr(meta,'has_rolled_current_turn',None)}"
                )

            if terminated or truncated:
                info["ep_steps"] = int(self._ep_steps)
                info["ep_shaping"] = float(getattr(self, "_ep_shaping", 0.0))
                info["ep_terminal"] = float(getattr(self, "_ep_terminal", 0.0))
                info["ep_truncated"] = bool(truncated)

            return obs, float(reward), bool(terminated), bool(truncated), info

        if self.state.phase == GamePhase.GAME_OVER:
            info["already_done"] = True
            return _finish_step(obs_from_state(self.state), 0.0, True, False)

        if self.debug and (self._env_steps % self.log_every_env_steps == 0):
            meta = self.state.round_meta
            print(
                f"[env-step {self._env_steps}] phase={self.state.phase} "
                f"cp={getattr(meta,'current_player_id',None)} "
                f"roll={getattr(meta,'current_roll_number',None)} "
                f"has_rolled={getattr(meta,'has_rolled_current_turn',None)} "
                f"HP: P1={self.state.players[0].health} P2={self.state.players[1].health} "
                f"TOK: P1={self.state.players[0].tokens} P2={self.state.players[1].tokens}"
            )

        # Guard: if not at an agent decision point, advance and return (no-op step)
        if not self._is_agent_decision_point():
            self._auto_advance()
            terminated = self.state.phase == GamePhase.GAME_OVER

            if self._env_steps >= self.max_env_steps_per_episode and not terminated:
                info["time_limit"] = True
                return _finish_step(obs_from_state(self.state), 0.0, False, True)

            info["auto_advanced"] = True
            return _finish_step(obs_from_state(self.state), 0.0, terminated, False)

        # --- Apply action depending on phase ---
        if self.state.phase == GamePhase.ROLLING:
            # expects 0..63
            mask = int(action) & 0b111111
            self._do_player_roll_decision(self.agent_id, mask)

        elif self.state.phase == GamePhase.GOD_FAVOR_SELECTION:
            self._do_player_favor_decision(self.agent_id, int(action))

        elif self.state.phase == GamePhase.FREYJA_REROLL:
            # expects 64..127 (we’ll accept anything and remap)
            freyja = self.state.freyja_reroll
            max_dice = getattr(freyja, "max_dice", 0) if freyja is not None else 0
            raw = int(action) - FREYJA_MASK_START
            mask = raw & 0b111111
            self._do_player_freyja_decision(self.agent_id, mask, max_dice=max_dice)

        else:
            # should never be a decision point for the agent
            pass

        # Advance opponent + resolution until next agent decision / terminal
        self._auto_advance()

        terminated = self.state.phase == GamePhase.GAME_OVER
        truncated = False

        if self._env_steps >= self.max_env_steps_per_episode and not terminated:
            truncated = True
            info["time_limit"] = True

        # Terminal reward
        step_shaping = 0.0
        step_terminal = 0.0

        if terminated:
            # NEW: expose winner for evaluation
            info["winner_player_id"] = self.state.winner_player_id

            if self.state.winner_player_id == self.agent_id:
                step_terminal = self.win_reward
            else:
                step_terminal = self.loss_reward
        else:
            if not self.use_terminal_only_reward:
                # existing HP shaping (damage advantage)
                d1 = prev_p1 - self.state.players[0].health  # damage to P1
                d2 = prev_p2 - self.state.players[1].health  # damage to P2

                if self.agent_id == PlayerId.P1:
                    damage_adv = d2 - d1  # damage_to_opp - damage_to_me
                else:
                    damage_adv = d1 - d2

                step_shaping += self.shaping_scale * float(damage_adv)

                # NEW: token advantage delta shaping
                new_t1 = self.state.players[0].tokens
                new_t2 = self.state.players[1].tokens
                dt1 = new_t1 - prev_t1
                dt2 = new_t2 - prev_t2

                if self.agent_id == PlayerId.P1:
                    token_adv_delta = dt1 - dt2
                else:
                    token_adv_delta = dt2 - dt1

                step_shaping += self.token_scale * float(token_adv_delta)

                # NEW: clip shaping
                if step_shaping > self.shaping_clip:
                    step_shaping = self.shaping_clip
                elif step_shaping < -self.shaping_clip:
                    step_shaping = -self.shaping_clip

        # NEW: truncation penalty (prevents stalling to time-limit)
        if truncated and not terminated:
            reward += float(self.truncation_penalty)

        reward += step_terminal + step_shaping
        self._ep_terminal += step_terminal
        self._ep_shaping += step_shaping

        return _finish_step(obs_from_state(self.state), reward, terminated, truncated)

    def _action_type(self, a: int) -> str:
        if ROLL_MASK_START <= a <= ROLL_MASK_END:
            return "LOCK_MASK"
        if FREYJA_MASK_START <= a <= FREYJA_MASK_END:
            return "FREYJA_MASK"
        if FAVOR_START <= a <= FAVOR_END or a == FAVOR_SKIP:
            return "FAVOR_SELECT"
        return "UNKNOWN"

    def _debug_print_mask_ranges(self, m, label=""):
        idx = [i for i, ok in enumerate(m) if ok]
        if not idx:
            print(label, "no valid actions")
            return
        # group contiguous ranges
        ranges = []
        start = prev = idx[0]
        for x in idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                ranges.append((start, prev))
                start = prev = x
        ranges.append((start, prev))
        print(label, "valid_count=", len(idx), "ranges=", ranges, "sample=", idx[:20])

    # ---------------- decision point detection ----------------
    def _is_agent_decision_point(self) -> bool:
        s = self.state
        meta = s.round_meta

        if s.phase == GamePhase.ROLLING:
            return (
                meta is not None
                and meta.current_player_id == self.agent_id
                and meta.has_rolled_current_turn is True
                and meta.current_roll_number in (1, 2)
            )

        if s.phase == GamePhase.GOD_FAVOR_SELECTION:
            if s.round_meta is None:
                return False
            return not s.round_meta.god_favors_approved[self.agent_id]

        if s.phase == GamePhase.FREYJA_REROLL:
            freyja = s.freyja_reroll
            return freyja is not None and freyja.player_id == self.agent_id

        return False

    # ---------------- core helpers ----------------

    def _set_player(self, pid: PlayerId, new_player: PlayerState):
        p1, p2 = self.state.players
        if pid == PlayerId.P1:
            players = (new_player, p2)
        else:
            players = (p1, new_player)
        self.state = GameState(
            phase=self.state.phase,
            players=players,
            round_meta=self.state.round_meta,
            winner_player_id=self.state.winner_player_id,
            resolution=self.state.resolution,
            freyja_reroll=self.state.freyja_reroll,
        )

    def _action_mask(self):
        # Full discrete action space: 0..137
        n = FAVOR_SKIP + 1
        mask = [False] * n

        # ✅ STRICT: only expose actions when the agent is actually allowed to act
        if not self._is_agent_decision_point():
            return mask

        s = self.state

        if s.phase == GamePhase.ROLLING:
            # allow only lock masks 0..63
            for a in range(0, 64):
                mask[a] = True
            return mask

        if s.phase == GamePhase.GOD_FAVOR_SELECTION:
            # allow skip always
            mask[FAVOR_SKIP] = True

            # allow favor choices only if player has tokens AND those favors are affordable
            player = get_player(self.state, self.agent_id)
            tokens = int(getattr(player, "tokens", 0))

            if tokens <= 0:
                return mask

            # FAVOR_TABLE entries are (favor_id, level_index)
            for i, (favor_id, level_index) in enumerate(FAVOR_TABLE):
                try:
                    level = get_god_favor_def(favor_id).levels[level_index]
                    cost = int(level.cost)
                except Exception:
                    cost = 10**9
                if tokens >= cost:
                    mask[FAVOR_START + i] = True

            return mask

        if s.phase == GamePhase.FREYJA_REROLL:
            # allow only 64..127, but also only masks with <= max_dice bits (optional strictness)
            freyja = s.freyja_reroll
            max_dice = int(getattr(freyja, "max_dice", 0)) if freyja is not None else 0

            for raw in range(0, 64):
                # optional: enforce "<= max_dice dice" reroll
                if max_dice <= 0 or (raw & 0b111111).bit_count() <= max_dice:
                    mask[FREYJA_MASK_START + raw] = True
            return mask

        # Not a decision point: no valid actions
        return mask

    def _do_player_roll_decision(self, player_id: PlayerId, mask: int):
        # lock mask then finish roll
        self._apply_lock_mask(mask, player_id)
        self.state = finish_roll(self.state, player_id)

    def _apply_lock_mask(self, mask: int, player_id: PlayerId):
        p = copy.deepcopy(get_player(self.state, player_id))
        for i, d in enumerate(p.dice):
            d.is_locked = bool((mask >> i) & 1)
        self._set_player(player_id, p)

    def _do_player_favor_decision(self, player_id: PlayerId, action: int):
        # Accept any int; clamp into valid favor actions.
        if action == FAVOR_SKIP:
            self.state = skip_god_favor(self.state, player_id)
            return

        # If action not in favor range, treat as skip (keeps env “no invalid actions”)
        if action < FAVOR_START or action > FAVOR_END:
            self.state = skip_god_favor(self.state, player_id)
            return

        idx = action - FAVOR_START
        favor_id, level_index = FAVOR_TABLE[idx]
        self.state = choose_god_favor(self.state, player_id, favor_id, level_index)

    def _do_player_freyja_decision(
        self, player_id: PlayerId, reroll_mask: int, max_dice: int
    ):
        """
        Freyja reroll: choose exactly max_dice dice to reroll.
        reroll_mask bit=1 => reroll that die (i.e. unlock it before roll_dice).
        """
        p = copy.deepcopy(get_player(self.state, player_id))

        # normalize mask to exact K bits (no invalids)
        chosen = self._mask_to_exact_k_indices(reroll_mask, k=max(0, int(max_dice)))

        # Set all locked first, then unlock chosen indices
        for d in p.dice:
            d.is_locked = True
        for i in chosen:
            if 0 <= i < len(p.dice):
                p.dice[i].is_locked = False

        self._set_player(player_id, p)

        # execute reroll (engine checks unlocked==max_dice)
        self.state = roll_dice(self.state, player_id, rand=self.rand)

    # ---------------- auto-advance ----------------
    def _auto_advance(self):
        """
        Advance until:
        - agent decision point, OR
        - terminal.
        """
        for _ in range(self.max_auto_advance_iters):
            s = self.state
            meta = s.round_meta

            if s.phase == GamePhase.GAME_OVER:
                return

            # If we reached a point where the agent needs to act, stop.
            if self._is_agent_decision_point():
                return

            # ---- ROLLING ----
            if s.phase == GamePhase.ROLLING and meta is not None:
                cp = meta.current_player_id

                # Auto-roll if current player hasn't rolled yet
                if not meta.has_rolled_current_turn:
                    self.state = roll_dice(self.state, cp, rand=self.rand)
                    continue

                # After rolling on roll 1/2: if opponent, pick mask then finish; if agent, we would have returned above.
                if meta.current_roll_number in (1, 2):
                    if cp == self.opp_id:
                        opp_mask = int(self.opp_roll_policy(self.state)) & 0b111111
                        self._do_player_roll_decision(self.opp_id, opp_mask)
                        continue
                    else:
                        # agent decision point would have returned already
                        return

                # roll 3: no lock decision; must finish roll to advance phase/turn
                if meta.current_roll_number == 3:
                    self.state = finish_roll(self.state, cp)
                    continue

                # safety fallback (shouldn't be reached)
            # ---- GOD FAVOR SELECTION ----
            if s.phase == GamePhase.GOD_FAVOR_SELECTION:
                rm = s.round_meta
                if rm is None:
                    return

                if not rm.god_favors_approved[self.opp_id]:
                    a = self.opp_favor_policy(self.state)
                    self._do_player_favor_decision(self.opp_id, a)
                    continue
                if not rm.god_favors_approved[self.agent_id]:
                    return
                continue

            # ---- FREYJA REROLL ----
            if s.phase == GamePhase.FREYJA_REROLL:
                freyja = s.freyja_reroll
                if freyja is None:
                    # shouldn’t happen, but don’t deadlock
                    self.state = GameState(
                        phase=GamePhase.RESOLUTION,
                        players=s.players,
                        round_meta=s.round_meta,
                        winner_player_id=s.winner_player_id,
                        resolution=s.resolution,
                        freyja_reroll=None,
                    )
                    continue

                pid = freyja.player_id
                max_dice = freyja.max_dice

                # If agent: decision point (returned earlier)
                if pid == self.agent_id:
                    return

                # Opponent: pick reroll mask and execute reroll immediately
                mask = int(self.opp_freyja_policy(self.state, max_dice)) & 0b111111
                self._do_player_freyja_decision(self.opp_id, mask, max_dice=max_dice)
                continue

            # ---- RESOLUTION ----
            if s.phase == GamePhase.RESOLUTION:
                before = self.state
                self.state = advance_resolution(self.state)

                # If resolution is “stuck” (shouldn’t), bail to avoid infinite loop
                if (
                    self.state.phase == before.phase
                    and self.state.resolution == before.resolution
                ):
                    return

                # If resolution switched to FREYJA_REROLL / GAME_OVER / ROLLING, loop handles next
                continue

            # Any other phase: just stop (prevents runaway)
            return

        meta = self.state.round_meta
        raise RuntimeError(
            f"_auto_advance exceeded {self.max_auto_advance_iters} iters; "
            f"phase={self.state.phase} cp={getattr(meta,'current_player_id',None)} "
            f"roll={getattr(meta,'current_roll_number',None)} has_rolled={getattr(meta,'has_rolled_current_turn',None)}"
        )

    # ---------------- opponent / random helpers ----------------
    def _random_exact_k_mask(self, k: int) -> int:
        k = max(0, min(6, int(k)))
        pool = list(range(6))
        chosen: list[int] = []

        for _ in range(k):
            j = self.rand.randrange(len(pool))
            chosen.append(pool.pop(j))

        m = 0
        for i in chosen:
            m |= 1 << i
        return m

    def _mask_to_exact_k_indices(self, mask: int, k: int) -> list[int]:
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

    def _random_favor_action_for_player(self, pid: PlayerId) -> int:
        """
        Cheap default policy:
        - If player has 0 tokens: skip.
        - Else randomly choose among 9 favor/level actions + skip.
        (Engine will naturally fizzle if not enough tokens for a given level.)
        """
        p = get_player(self.state, pid)
        if int(p.tokens) <= 0:
            return FAVOR_SKIP
        # include skip sometimes so it can “save” tokens
        actions = list(range(FAVOR_START, FAVOR_END + 1)) + [FAVOR_SKIP]
        return actions[self.rand.randrange(len(actions))]

    def _select_opponent_policy(self):
        """Sample opponent policy based on mix weights."""
        policies = list(self.opponents.keys())
        weights = [self.opponents[policy].p_mix for policy in policies]

        r = random.Random(self.rand.randrange(1 << 30)).random()
        cumulative = 0.0
        chosen = policies[-1]
        for policy_name, w in zip(policies, weights):
            cumulative += w
            if r <= cumulative:
                chosen = policy_name
                break

        self.current_opponent_name = chosen

        if chosen not in self.opponent_policies:
            raise KeyError(f"Opponent policy {chosen} is not registered")

        policy_cfg = self.opponent_policies[chosen]
        return policy_cfg.policy(
            self.rand, self.state, self.opp_id, self.opponents[chosen].p_action
        )

    def set_opponents(self, opponents: dict[OpponentName, Opponent]):
        self.opponents = dict(opponents)
        total = sum([cfg.p_mix for cfg in self.opponents.values()])
        if total <= 0:
            raise ValueError("opponents total weight must be > 0")
        self.opponents = {
            k: Opponent(p_mix=v.p_mix / total, p_action=v.p_action)
            for k, v in self.opponents.items()
        }

    def get_action_mask(self) -> np.ndarray:
        return np.array(
            get_action_mask_for_player(self.state, self.agent_id), dtype=np.bool_
        )

    def register_opponent_policy(
        self,
        name: OpponentName,
        policy_builder: Callable[
            [Randomizer, GameState, PlayerId, float], OpponentPolicy
        ],
        p_action: float = 1.0,
    ):
        self.opponent_policies[name] = OpponentConfig(
            policy=policy_builder,
            p_action=p_action,
        )
