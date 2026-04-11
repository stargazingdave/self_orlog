from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from game.types.game import GameState
from game.types.players import PlayerId
from game.types.randomizer import Randomizer
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
from rl.env.config import OpponentPolicy
from rl.env.obs import get_observation_for_player
from rl.env.opponents.policies import (
    _random_exact_k_mask,
    _random_favor_action_for_player,
)
from rl.utils.dirs import (
    ensure_dir,
)


@dataclass
class PoolEntry:
    kind: str
    step: int
    path: Path
    source: str


class SelfPlayPool:
    def __init__(
        self,
        base_dir: Path,
        *,
        seed: int = 0,
        max_recent: int = 5,
        max_historical: int = 8,
        max_best: int = 3,
        bucket_weights: dict[str, float] | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.recent_dir = ensure_dir(self.base_dir / "recent")
        self.historical_dir = ensure_dir(self.base_dir / "historical")
        self.best_dir = ensure_dir(self.base_dir / "best")
        self.manifest_path = self.base_dir / "manifest.json"

        self.max_recent = int(max_recent)
        self.max_historical = int(max_historical)
        self.max_best = int(max_best)

        self.rng = np.random.default_rng(seed)
        self.bucket_weights = bucket_weights or {
            "recent": 0.30,
            "historical": 0.40,
            "best": 0.30,
        }

        self._cache: dict[str, MaskablePPO] = {}
        self._write_manifest()

    def _sorted_zip_files(self, d: Path) -> list[Path]:
        return sorted(d.glob("*.zip"))

    def _prune_oldest(self, d: Path, max_keep: int) -> None:
        files = self._sorted_zip_files(d)
        while len(files) > max_keep:
            victim = files.pop(0)
            self._cache.pop(str(victim.resolve()), None)
            victim.unlink(missing_ok=True)

    def _move_oldest_recent_to_historical_if_needed(self) -> None:
        recent_files = self._sorted_zip_files(self.recent_dir)
        while len(recent_files) > self.max_recent:
            victim = recent_files.pop(0)
            target = self.historical_dir / victim.name
            if target.exists():
                target.unlink()
            shutil.move(str(victim), str(target))
            self._cache.pop(str(victim.resolve()), None)

        self._prune_oldest(self.historical_dir, self.max_historical)

    def _write_manifest(self) -> None:
        data = {
            "recent": [p.name for p in self._sorted_zip_files(self.recent_dir)],
            "historical": [p.name for p in self._sorted_zip_files(self.historical_dir)],
            "best": [p.name for p in self._sorted_zip_files(self.best_dir)],
            "bucket_weights": self.bucket_weights,
            "max_recent": self.max_recent,
            "max_historical": self.max_historical,
            "max_best": self.max_best,
        }
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def bootstrap_from_existing_best(self, best_model_path: Path) -> Path:
        """
        Bootstrap the pool with an initial best model copied from an existing path.
        This is useful for starting a new self-play league from a strong pre-trained model.
        """
        best_model_path = Path(best_model_path)
        if not best_model_path.exists():
            raise FileNotFoundError(f"bootstrap model not found: {best_model_path}")

        target = self.best_dir / "bootstrap_best_step0.zip"
        shutil.copy2(best_model_path, target)
        self._prune_oldest(self.best_dir, self.max_best)
        self._write_manifest()
        return target

    def add_recent_snapshot_from_model(self, model: MaskablePPO, step: int) -> Path:
        target = self.recent_dir / f"recent_step_{int(step):09d}.zip"
        model.save(str(target.with_suffix("")))
        self._move_oldest_recent_to_historical_if_needed()
        self._write_manifest()
        return target

    def sync_best_snapshot(self, best_model_path: Path, step: int) -> Path:
        best_model_path = Path(best_model_path)
        if not best_model_path.exists():
            raise FileNotFoundError(f"best model not found: {best_model_path}")

        target = self.best_dir / f"best_step_{int(step):09d}.zip"
        if not target.exists():
            shutil.copy2(best_model_path, target)
            self._prune_oldest(self.best_dir, self.max_best)
            self._write_manifest()
        return target

    def _entries_by_kind(self, kind: str) -> list[PoolEntry]:
        d = {
            "recent": self.recent_dir,
            "historical": self.historical_dir,
            "best": self.best_dir,
        }[kind]
        out: list[PoolEntry] = []
        for p in self._sorted_zip_files(d):
            step = 0
            stem = p.stem
            if "_step_" in stem:
                try:
                    step = int(stem.rsplit("_step_", 1)[1])
                except ValueError:
                    step = 0
            out.append(PoolEntry(kind=kind, step=step, path=p, source=stem))
        return out

    def sample_entry(self) -> PoolEntry | None:
        buckets = {
            "recent": self._entries_by_kind("recent"),
            "historical": self._entries_by_kind("historical"),
            "best": self._entries_by_kind("best"),
        }
        available_kinds = [k for k, v in buckets.items() if len(v) > 0]
        if not available_kinds:
            return None

        weights = np.array(
            [float(self.bucket_weights.get(k, 0.0)) for k in available_kinds],
            dtype=np.float64,
        )
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        chosen_kind = available_kinds[
            int(self.rng.choice(len(available_kinds), p=weights))
        ]
        chosen_bucket = buckets[chosen_kind]
        idx = int(self.rng.integers(0, len(chosen_bucket)))
        return chosen_bucket[idx]

    def load_model(self, path: Path) -> MaskablePPO:
        key = str(Path(path).resolve())
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        model = MaskablePPO.load(str(path))
        self._cache[key] = model
        return model


# ---------------- self play policy ----------------
def build_pool_selfplay_policy(pool: SelfPlayPool):
    def selfplay_policy_builder(
        rand: Randomizer, state: GameState, pid: PlayerId, p_action: float
    ) -> OpponentPolicy:
        episode_state_id: int | None = None
        current_entry: PoolEntry | None = None

        def _ensure_episode_entry(state) -> PoolEntry | None:
            nonlocal episode_state_id, current_entry
            cur_state_id = id(state)
            if episode_state_id != cur_state_id:
                episode_state_id = cur_state_id
                current_entry = pool.sample_entry()
            return current_entry

        def _random_valid_action_for_player() -> int:
            valid = get_action_mask_for_player(state, pid)
            valid_actions = [i for i, ok in enumerate(valid) if ok]
            if not valid_actions:
                return 0
            j = rand.randrange(len(valid_actions))
            return int(valid_actions[j])

        def _predict_action_for_player(state) -> int:
            entry = _ensure_episode_entry(state)
            if entry is None:
                return _random_valid_action_for_player()

            model = pool.load_model(entry.path)
            obs = get_observation_for_player(state, pid)
            mask = np.array(get_action_mask_for_player(state, pid), dtype=np.bool_)

            if not mask.any():
                return 0

            action, _ = model.predict(
                obs,
                deterministic=False,
                action_masks=mask,
            )
            return int(action)

        def roll_fn(state) -> int:
            if rand.random() > p_action:
                return rand.randrange(64)

            a = _predict_action_for_player(state)
            if ROLL_MASK_START <= a <= ROLL_MASK_END:
                return int(a)
            return rand.randrange(64)

        def freyja_fn(state, max_dice) -> int:
            if rand.random() > p_action:
                return _random_exact_k_mask(max_dice, rand)

            a = _predict_action_for_player(state)
            if FREYJA_MASK_START <= a <= FREYJA_MASK_END:
                return int(a - FREYJA_MASK_START)

            return _random_exact_k_mask(max_dice, rand)

        def favor_fn(state) -> int:
            if rand.random() > p_action:
                return _random_favor_action_for_player(rand, state, pid)

            a = _predict_action_for_player(state)
            if FAVOR_START <= a <= FAVOR_END or a == FAVOR_SKIP:
                return int(a)

            return _random_favor_action_for_player(rand, state, pid)

        policy: OpponentPolicy = {
            "roll": roll_fn,
            "freyja": freyja_fn,
            "favor": favor_fn,
        }
        return policy

    return selfplay_policy_builder
