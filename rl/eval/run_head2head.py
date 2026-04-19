from __future__ import annotations

import copy
import io
import json
import random
from typing import Literal
import zipfile
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from torch.serialization import safe_globals

from game.functions.utils import get_player
from game.state_transitions.advance_resolution import advance_resolution
from game.state_transitions.choose_god_favor import choose_god_favor
from game.state_transitions.create_new_game import create_new_game
from game.state_transitions.finish_roll import finish_roll
from game.state_transitions.roll_dice import roll_dice
from game.state_transitions.skip_god_favor import skip_god_favor
from game.state_transitions.start_game import start_game
from game.types.game import GamePhase, GameState
from game.types.players import PlayerId, PlayerState
from game.types.randomizer import Randomizer
from rl.ddqn.config import DDQNConfig
from rl.ddqn.model import QNet
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
from rl.env.config import FAVOR_TABLE
from rl.env.obs import get_observation_for_player


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_AUTO_ADVANCE_ITERS = 2000


@dataclass
class GameStats:
    games: int
    left_wins: int
    right_wins: int
    truncations: int
    steps_total: int


@dataclass
class ModelMeta:
    name: str
    architecture: Literal["ppo", "ddqn"]
    source_path: str


@dataclass
class GameHistory:
    seed: int
    left_name: str
    right_name: str
    left_source_path: str
    right_source_path: str
    left_architecture: str
    right_architecture: str
    deterministic: bool
    max_steps_per_episode: int
    steps: list[dict]
    result: dict


def to_jsonable(obj):
    if isinstance(obj, Enum):
        return obj.name
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def pair_dir_name(meta1: ModelMeta, meta2: ModelMeta) -> str:
    return f"{sanitize_name(meta1.name)}__vs__{sanitize_name(meta2.name)}"


def stats_to_dict(stats: GameStats, left_name: str, right_name: str) -> dict:
    games = stats.games
    decided = stats.left_wins + stats.right_wins

    return {
        "left_name": left_name,
        "right_name": right_name,
        "games": stats.games,
        "left_wins": stats.left_wins,
        "right_wins": stats.right_wins,
        "truncations": stats.truncations,
        "steps_total": stats.steps_total,
        "avg_steps": (stats.steps_total / games) if games else 0.0,
        "left_winrate_all": (stats.left_wins / games) if games else 0.0,
        "right_winrate_all": (stats.right_wins / games) if games else 0.0,
        "truncation_rate": (stats.truncations / games) if games else 0.0,
        "left_winrate_decided": (stats.left_wins / decided) if decided else 0.0,
        "right_winrate_decided": (stats.right_wins / decided) if decided else 0.0,
    }


def save_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def build_overall_stats(
    stats_a: GameStats,
    stats_b: GameStats,
    meta1: ModelMeta,
    meta2: ModelMeta,
) -> dict:
    total_games = stats_a.games + stats_b.games
    model1_wins = stats_a.left_wins + stats_b.right_wins
    model2_wins = stats_a.right_wins + stats_b.left_wins
    truncations = stats_a.truncations + stats_b.truncations
    steps_total = stats_a.steps_total + stats_b.steps_total
    decided = model1_wins + model2_wins

    return {
        "model1_name": meta1.name,
        "model2_name": meta2.name,
        "total_games": total_games,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "truncations": truncations,
        "steps_total": steps_total,
        "avg_steps": (steps_total / total_games) if total_games else 0.0,
        "model1_winrate_all": (model1_wins / total_games) if total_games else 0.0,
        "model2_winrate_all": (model2_wins / total_games) if total_games else 0.0,
        "truncation_rate": (truncations / total_games) if total_games else 0.0,
        "model1_winrate_decided": (model1_wins / decided) if decided else 0.0,
        "model2_winrate_decided": (model2_wins / decided) if decided else 0.0,
    }


# =========================
# Models
# =========================


class PolicyAdapter:
    architecture: str
    source_path: str

    def predict_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        raise NotImplementedError


class PPOPolicy(PolicyAdapter):
    def __init__(self, model: MaskablePPO, source_path: Path):
        self.model = model
        self.architecture = "ppo"
        self.source_path = str(source_path)

    def predict_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        action, _ = self.model.predict(
            obs,
            deterministic=deterministic,
            action_masks=np.asarray(mask, dtype=bool),
        )
        return int(action)


class DDQNPolicy(PolicyAdapter):
    def __init__(
        self,
        model_state_dict: dict,
        hidden_dim: int,
        n_layers: int,
        source_path: Path,
        device: str = "cpu",
    ):
        self.architecture = "ddqn"
        self.source_path = str(source_path)
        self.device = device
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.model_state_dict = model_state_dict

        self.q_net: nn.Module | None = None
        self.obs_dim: int | None = None
        self.n_actions: int | None = None

    def _ensure_loaded(self, obs_dim: int, n_actions: int) -> None:
        if self.q_net is not None:
            if self.obs_dim != obs_dim or self.n_actions != n_actions:
                raise RuntimeError(
                    f"DDQN already initialized with obs_dim={self.obs_dim}, "
                    f"n_actions={self.n_actions}, got obs_dim={obs_dim}, "
                    f"n_actions={n_actions}"
                )
            return

        q_net = QNet(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        q_net.load_state_dict(self.model_state_dict)
        q_net.eval()

        self.q_net = q_net
        self.obs_dim = obs_dim
        self.n_actions = n_actions

    def predict_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)

        if obs.ndim != 1:
            raise RuntimeError(f"DDQN expected 1D obs, got shape={obs.shape}")
        if mask.ndim != 1:
            raise RuntimeError(f"DDQN expected 1D mask, got shape={mask.shape}")
        if not mask.any():
            raise RuntimeError("DDQN received empty valid-action mask")

        self._ensure_loaded(obs_dim=int(obs.shape[0]), n_actions=int(mask.shape[0]))
        assert self.q_net is not None

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_values = self.q_net(obs_t)[0]

        q_values = q_values.masked_fill(~mask_t, -1e9)
        return int(torch.argmax(q_values).item())


def load_ddqn_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        model_bytes = zf.read("model.pt")
        params = json.loads(zf.read("params.json").decode("utf-8"))
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

    with safe_globals([DDQNConfig]):
        ckpt = torch.load(io.BytesIO(model_bytes), map_location=DEVICE)

    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{zip_path} missing 'model_state_dict' in model.pt")

    return ckpt, params, meta


def load_policy(path: str | Path, architecture: str) -> PolicyAdapter:
    p = Path(path)

    if architecture == "ppo":
        model = MaskablePPO.load(str(p), device=DEVICE, print_system_info=True)
        return PPOPolicy(model=model, source_path=p)

    if architecture == "ddqn":
        ckpt, params, _meta = load_ddqn_zip(p)
        return DDQNPolicy(
            model_state_dict=ckpt["model_state_dict"],
            hidden_dim=int(params["hidden_dim"]),
            n_layers=int(params["n_layers"]),
            source_path=p,
            device=DEVICE,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")


# =========================
# Pure game helpers
# =========================


def action_type(a: int) -> str:
    if ROLL_MASK_START <= a <= ROLL_MASK_END:
        return "LOCK_MASK"
    if FREYJA_MASK_START <= a <= FREYJA_MASK_END:
        return "FREYJA_MASK"
    if FAVOR_START <= a <= FAVOR_END or a == FAVOR_SKIP:
        return "FAVOR_SELECT"
    return "UNKNOWN"


def get_decision_player_id(state: GameState) -> PlayerId | None:
    rm = state.round_meta

    if state.phase == GamePhase.GAME_OVER:
        return None

    if state.phase == GamePhase.ROLLING:
        if (
            rm is not None
            and rm.current_player_id is not None
            and rm.has_rolled_current_turn is True
            and rm.current_roll_number in (1, 2)
        ):
            return rm.current_player_id
        return None

    if state.phase == GamePhase.GOD_FAVOR_SELECTION:
        if rm is None:
            return None

        cp = rm.current_player_id
        if cp is not None and not rm.god_favors_approved[cp]:
            return cp

        if not rm.god_favors_approved[PlayerId.P1]:
            return PlayerId.P1
        if not rm.god_favors_approved[PlayerId.P2]:
            return PlayerId.P2

        return None

    if state.phase == GamePhase.FREYJA_REROLL:
        if state.freyja_reroll is None:
            return None
        return state.freyja_reroll.player_id

    return None


def set_player(state: GameState, pid: PlayerId, new_player: PlayerState) -> GameState:
    p1, p2 = state.players
    if pid == PlayerId.P1:
        players = (new_player, p2)
    else:
        players = (p1, new_player)

    return GameState(
        phase=state.phase,
        players=players,
        round_meta=state.round_meta,
        winner_player_id=state.winner_player_id,
        resolution=state.resolution,
        freyja_reroll=state.freyja_reroll,
    )


def apply_lock_mask(state: GameState, player_id: PlayerId, mask: int) -> GameState:
    p = copy.deepcopy(get_player(state, player_id))
    for i, d in enumerate(p.dice):
        d.is_locked = bool((mask >> i) & 1)
    return set_player(state, player_id, p)


def mask_to_exact_k_indices(mask: int, k: int) -> list[int]:
    k = max(0, min(6, int(k)))

    ones = [i for i in range(6) if ((mask >> i) & 1) == 1]
    if len(ones) > k:
        return ones[:k]

    if len(ones) < k:
        for i in range(6):
            if i not in ones:
                ones.append(i)
                if len(ones) == k:
                    break

    return ones


def apply_player_action(
    state: GameState,
    player_id: PlayerId,
    action: int,
    rand: Randomizer,
) -> GameState:
    phase = state.phase
    atype = action_type(int(action))

    if phase == GamePhase.ROLLING:
        if atype != "LOCK_MASK":
            raise RuntimeError(f"Illegal action for ROLLING: {action} ({atype})")
        lock_mask = int(action) & 0b111111
        state = apply_lock_mask(state, player_id, lock_mask)
        state = finish_roll(state, player_id)
        return state

    if phase == GamePhase.GOD_FAVOR_SELECTION:
        if atype != "FAVOR_SELECT":
            raise RuntimeError(
                f"Illegal action for GOD_FAVOR_SELECTION: {action} ({atype})"
            )

        if action == FAVOR_SKIP:
            return skip_god_favor(state, player_id)

        if action < FAVOR_START or action > FAVOR_END:
            return skip_god_favor(state, player_id)

        idx = action - FAVOR_START
        favor_id, level_index = FAVOR_TABLE[idx]
        return choose_god_favor(state, player_id, favor_id, level_index)

    if phase == GamePhase.FREYJA_REROLL:
        if atype != "FREYJA_MASK":
            raise RuntimeError(f"Illegal action for FREYJA_REROLL: {action} ({atype})")

        freyja = state.freyja_reroll
        max_dice = int(getattr(freyja, "max_dice", 0)) if freyja is not None else 0
        raw = int(action) - FREYJA_MASK_START
        reroll_mask = raw & 0b111111
        chosen = mask_to_exact_k_indices(reroll_mask, k=max_dice)

        p = copy.deepcopy(get_player(state, player_id))
        for d in p.dice:
            d.is_locked = True
        for i in chosen:
            if 0 <= i < len(p.dice):
                p.dice[i].is_locked = False

        state = set_player(state, player_id, p)
        state = roll_dice(state, player_id, rand=rand)
        return state

    raise RuntimeError(f"apply_player_action called in non-decision phase: {phase}")


def auto_advance(
    state: GameState,
    rand: Randomizer,
    debug: bool = False,
) -> GameState:
    for _ in range(MAX_AUTO_ADVANCE_ITERS):
        if state.phase == GamePhase.GAME_OVER:
            return state

        decision_player = get_decision_player_id(state)
        if decision_player is not None:
            return state

        rm = state.round_meta
        if rm is None:
            raise RuntimeError(
                "auto_advance encountered round_meta=None after start_game"
            )

        if state.phase == GamePhase.ROLLING:
            cp = rm.current_player_id

            if not rm.has_rolled_current_turn:
                state = roll_dice(state, cp, rand=rand)
                continue

            if rm.current_roll_number == 3:
                state = finish_roll(state, cp)
                continue

            return state

        if state.phase == GamePhase.RESOLUTION:
            before_phase = state.phase
            before_resolution = state.resolution
            before_winner = state.winner_player_id

            state = advance_resolution(state)

            if (
                state.phase == before_phase
                and state.resolution == before_resolution
                and state.winner_player_id == before_winner
            ):
                raise RuntimeError(
                    "advance_resolution made no progress during RESOLUTION"
                )
            continue

        # If your engine has some additional forced phase later, add it here.
        return state

    raise RuntimeError("auto_advance exceeded iteration limit")


# =========================
# Match loop
# =========================


@dataclass
class MatchResult:
    winner: PlayerId | None
    truncated: bool
    steps: int
    history: GameHistory


def choose_action_for_player(
    state: GameState,
    decision_player: PlayerId,
    p1_policy: PolicyAdapter,
    p2_policy: PolicyAdapter,
    deterministic: bool = True,
) -> int:
    obs = get_observation_for_player(state, decision_player)
    mask = np.asarray(get_action_mask_for_player(state, decision_player), dtype=bool)

    if not mask.any():
        raise RuntimeError(
            f"No valid actions for player {decision_player} in phase={state.phase}"
        )

    policy = p1_policy if decision_player == PlayerId.P1 else p2_policy
    action = policy.predict_action(obs, mask, deterministic=deterministic)

    if action < 0 or action >= len(mask) or not mask[action]:
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            raise RuntimeError(
                "Policy produced invalid action and no valid fallback exists"
            )
        return int(valid_actions[0])

    return int(action)


def play_one_game(
    seed: int,
    p1_policy: PolicyAdapter,
    p2_policy: PolicyAdapter,
    p1_name: str = "P1",
    p2_name: str = "P2",
    deterministic: bool = True,
    debug: bool = False,
    max_steps_per_episode: int = 1000,
) -> MatchResult:
    rand = Randomizer(random.Random(seed), seed)

    state = create_new_game()
    state = start_game(
        state,
        player1_name=p1_name,
        player2_name=p2_name,
    )
    state = auto_advance(state, rand, debug=debug)

    history_steps: list[dict] = [
        {
            "event": "initial_state",
            "state": to_jsonable(state),
        }
    ]

    step_count = 0

    while state.phase != GamePhase.GAME_OVER:
        if step_count >= max_steps_per_episode:
            history = GameHistory(
                seed=seed,
                left_name=p1_name,
                right_name=p2_name,
                left_source_path=p1_policy.source_path,
                right_source_path=p2_policy.source_path,
                left_architecture=p1_policy.architecture,
                right_architecture=p2_policy.architecture,
                deterministic=deterministic,
                max_steps_per_episode=max_steps_per_episode,
                steps=history_steps,
                result={
                    "winner": None,
                    "truncated": True,
                    "steps": step_count,
                    "final_phase": state.phase.name,
                    "final_state": to_jsonable(state),
                },
            )
            return MatchResult(
                winner=None,
                truncated=True,
                steps=step_count,
                history=history,
            )

        decision_player = get_decision_player_id(state)
        if decision_player is None:
            raise RuntimeError(
                f"No decision player in non-terminal state. phase={state.phase}"
            )

        obs = get_observation_for_player(state, decision_player)
        mask = np.asarray(
            get_action_mask_for_player(state, decision_player), dtype=bool
        )

        action = choose_action_for_player(
            state=state,
            decision_player=decision_player,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
            deterministic=deterministic,
        )

        before_state = state

        state = apply_player_action(
            state=state,
            player_id=decision_player,
            action=action,
            rand=rand,
        )
        state = auto_advance(state, rand, debug=debug)

        history_steps.append(
            {
                "step_index": step_count,
                "phase_before": before_state.phase.name,
                "decision_player": decision_player.name,
                "action": int(action),
                "action_type": action_type(int(action)),
                "valid_actions": np.flatnonzero(mask).tolist(),
                "observation": np.asarray(obs, dtype=np.float32).tolist(),
                "state_before": to_jsonable(before_state),
                "state_after": to_jsonable(state),
            }
        )

        step_count += 1

    history = GameHistory(
        seed=seed,
        left_name=p1_name,
        right_name=p2_name,
        left_source_path=p1_policy.source_path,
        right_source_path=p2_policy.source_path,
        left_architecture=p1_policy.architecture,
        right_architecture=p2_policy.architecture,
        deterministic=deterministic,
        max_steps_per_episode=max_steps_per_episode,
        steps=history_steps,
        result={
            "winner": state.winner_player_id.name if state.winner_player_id else None,
            "truncated": False,
            "steps": step_count,
            "final_phase": state.phase.name,
            "final_state": to_jsonable(state),
        },
    )

    return MatchResult(
        winner=state.winner_player_id,
        truncated=False,
        steps=step_count,
        history=history,
    )


def run_seated_series(
    n_games: int,
    seed_start: int,
    left_policy: PolicyAdapter,
    right_policy: PolicyAdapter,
    left_name: str,
    right_name: str,
    deterministic: bool,
    debug: bool,
    max_steps_per_episode: int,
    histories_dir: Path | None = None,
) -> GameStats:
    stats = GameStats(
        games=0,
        left_wins=0,
        right_wins=0,
        truncations=0,
        steps_total=0,
    )

    if histories_dir is not None:
        ensure_dir(histories_dir)

    for i in range(n_games):
        seed = seed_start + i
        result = play_one_game(
            seed=seed,
            p1_policy=left_policy,
            p2_policy=right_policy,
            p1_name=left_name,
            p2_name=right_name,
            deterministic=deterministic,
            debug=debug,
            max_steps_per_episode=max_steps_per_episode,
        )

        stats.games += 1
        stats.steps_total += result.steps

        if result.truncated:
            stats.truncations += 1
        elif result.winner == PlayerId.P1:
            stats.left_wins += 1
        elif result.winner == PlayerId.P2:
            stats.right_wins += 1

        if histories_dir is not None:
            history_path = histories_dir / f"{i + 1:05d}.json"
            save_json(history_path, to_jsonable(result.history))

        if i % 50 == 0 and debug:
            print(
                f"Game {i}/{n_games} completed. "
                f"Current stats: {stats.left_wins} left wins, "
                f"{stats.right_wins} right wins, "
                f"{stats.truncations} truncations."
            )

    return stats


def print_series(title: str, stats: GameStats):
    games = stats.games
    left_wins = stats.left_wins
    right_wins = stats.right_wins
    truncations = stats.truncations
    decided = left_wins + right_wins
    avg_steps = stats.steps_total / games if games else 0.0

    print(f"\n=== {title} ===")
    print(f"games:        {games}")
    print(f"left wins:    {left_wins}")
    print(f"right wins:   {right_wins}")
    print(f"truncations:  {truncations}")
    print(f"avg steps:    {avg_steps:.2f}")

    if games > 0:
        print(f"left winrate over all games:   {left_wins / games:.4f}")
        print(f"right winrate over all games:  {right_wins / games:.4f}")
        print(f"truncation rate:               {truncations / games:.4f}")

    if decided > 0:
        print(f"left winrate over decided:     {left_wins / decided:.4f}")
        print(f"right winrate over decided:    {right_wins / decided:.4f}")


def main():
    output_root = Path("outputs") / "head2head"

    models_meta: list[ModelMeta] = [
        ModelMeta(
            name="DDQN_5M",
            architecture="ddqn",
            source_path=str(
                Path("outputs") / "rl" / "ddqn" / "full_5M" / "best" / "8" / "model.zip"
            ),
        ),
        ModelMeta(
            name="PPO_5M",
            architecture="ppo",
            source_path=str(
                Path("outputs")
                / "rl"
                / "pg"
                / "ppo"
                / "full_5M"
                / "8"
                / "best"
                / "model.zip"
            ),
        ),
        ModelMeta(
            name="DDQN_1M",
            architecture="ddqn",
            source_path=str(
                Path("outputs") / "rl" / "ddqn" / "full_1M" / "best" / "8" / "model.zip"
            ),
        ),
        ModelMeta(
            name="PPO_1M",
            architecture="ppo",
            source_path=str(
                Path("outputs")
                / "rl"
                / "pg"
                / "ppo"
                / "full_1M"
                / "8"
                / "best"
                / "model.zip"
            ),
        ),
    ]

    matches_to_run: list[tuple[ModelMeta, ModelMeta]] = []

    # Add each unique pair of models in either seating (A vs B and B vs A count as same pair)
    for i in range(len(models_meta)):
        for j in range(i + 1, len(models_meta)):
            m1 = models_meta[i]
            m2 = models_meta[j]
            matches_to_run.append((m1, m2))

    stochastic = False
    max_steps = 200
    games_per_seat = 1_000
    seed_start = 1000
    debug = True

    models: dict[str, PolicyAdapter] = {
        meta.name: load_policy(meta.source_path, meta.architecture)
        for meta in models_meta
    }

    for meta1, meta2 in matches_to_run:
        print(f"\n\n=== Starting match: {meta1.name} vs {meta2.name} ===")

        pair_name = pair_dir_name(meta1, meta2)
        pair_dir = output_root / pair_name
        seat_a_histories_dir = pair_dir / "histories" / "seat_a"
        seat_b_histories_dir = pair_dir / "histories" / "seat_b"
        ensure_dir(pair_dir)

        model1 = models[meta1.name]
        model2 = models[meta2.name]

        deterministic = not stochastic

        stats_a = run_seated_series(
            n_games=games_per_seat,
            seed_start=seed_start,
            left_policy=model1,
            right_policy=model2,
            left_name=meta1.name,
            right_name=meta2.name,
            deterministic=deterministic,
            debug=debug,
            max_steps_per_episode=max_steps,
            histories_dir=seat_a_histories_dir,
        )

        stats_b = run_seated_series(
            n_games=games_per_seat,
            seed_start=seed_start + games_per_seat,
            left_policy=model2,
            right_policy=model1,
            left_name=meta2.name,
            right_name=meta1.name,
            deterministic=deterministic,
            debug=debug,
            max_steps_per_episode=max_steps,
            histories_dir=seat_b_histories_dir,
        )

        print_series(f"Seat A: {meta1.name} vs {meta2.name}", stats_a)
        print_series(f"Seat B: {meta2.name} vs {meta1.name}", stats_b)

        overall_stats = build_overall_stats(stats_a, stats_b, meta1, meta2)

        seat_a_stats = {
            "pair": pair_name,
            "seat": "seat_a",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_left": asdict(meta1),
            "model_right": asdict(meta2),
            "deterministic": deterministic,
            "max_steps_per_episode": max_steps,
            "games_per_seat": games_per_seat,
            "seed_start": seed_start,
            **stats_to_dict(stats_a, meta1.name, meta2.name),
        }

        seat_b_stats = {
            "pair": pair_name,
            "seat": "seat_b",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_left": asdict(meta2),
            "model_right": asdict(meta1),
            "deterministic": deterministic,
            "max_steps_per_episode": max_steps,
            "games_per_seat": games_per_seat,
            "seed_start": seed_start + games_per_seat,
            **stats_to_dict(stats_b, meta2.name, meta1.name),
        }

        overall_payload = {
            "pair": pair_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model1": asdict(meta1),
            "model2": asdict(meta2),
            "deterministic": deterministic,
            "max_steps_per_episode": max_steps,
            "games_per_seat": games_per_seat,
            "seat_a": {
                "left": meta1.name,
                "right": meta2.name,
            },
            "seat_b": {
                "left": meta2.name,
                "right": meta1.name,
            },
            **overall_stats,
        }

        save_json(pair_dir / "seat_a_stats.json", seat_a_stats)
        save_json(pair_dir / "seat_b_stats.json", seat_b_stats)
        save_json(pair_dir / "overall_stats.json", overall_payload)

        print("\n=== Overall ===")
        print(f"total games:                    {overall_stats['total_games']}")
        print(f"{meta1.name} wins:              {overall_stats['model1_wins']}")
        print(f"{meta2.name} wins:              {overall_stats['model2_wins']}")
        print(f"truncations:                    {overall_stats['truncations']}")

        if overall_stats["total_games"] > 0:
            print(
                f"{meta1.name} winrate all:       "
                f"{overall_stats['model1_winrate_all']:.4f}"
            )
            print(
                f"{meta2.name} winrate all:       "
                f"{overall_stats['model2_winrate_all']:.4f}"
            )
            print(
                f"truncation rate:                "
                f"{overall_stats['truncation_rate']:.4f}"
            )

        decided = overall_stats["model1_wins"] + overall_stats["model2_wins"]
        if decided > 0:
            print(
                f"{meta1.name} winrate decided:   "
                f"{overall_stats['model1_winrate_decided']:.4f}"
            )
            print(
                f"{meta2.name} winrate decided:   "
                f"{overall_stats['model2_winrate_decided']:.4f}"
            )


if __name__ == "__main__":
    main()
