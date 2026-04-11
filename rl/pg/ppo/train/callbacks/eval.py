from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from game.types.players import PlayerId
from rl.config import HistoryEntry
from rl.env.config import OpponentName
from rl.env.env import Opponent
from rl.pg.ppo.utils.plot import save_all_graphs
from rl.pg.ppo.utils.write_eval_txt import write_eval_txt
from rl.pg.ppo.utils.dirs import build_checkpoint_dir
from rl.pg.ppo.utils.env_factory import make_env
from rl.pg.ppo.utils.save_model import save_model
from rl.utils.dirs import ensure_dir


@dataclass(slots=True)
class EvalBenchmark:
    name: str
    opponents: dict[OpponentName, float]


class BenchmarkEvalCallback(BaseCallback):
    """
    Evaluates named benchmark groups made of weighted opponents.

    For each eval step:
      - evaluates every unique opponent used by any benchmark
      - logs per-opponent metrics
      - computes weighted benchmark metrics
      - logs per-benchmark metrics
      - if a benchmark improves, saves:
            <outputs_dir>/<benchmark_name>/best/model.zip
            <outputs_dir>/<benchmark_name>/best/eval.txt

    At training end:
      - saves graphs for each opponent:
            <outputs_dir>/graphs/<opponent>.png
      - saves graphs for each benchmark:
            <outputs_dir>/graphs/<benchmark>.png

    Each graph contains 4 curves:
      - mean return
      - winrate
      - return variance
      - winrate variance
    """

    def __init__(
        self,
        run_name: str,
        benchmarks: list[EvalBenchmark],
        outputs_dir: str | Path,
        eval_freq: int,
        n_eval_episodes: int,
        max_steps: int,
        eval_seed: int = 12345,
        deterministic: bool = True,
        p_action: float = 1.0,
        verbose_print: bool = True,
    ):
        super().__init__()

        if not benchmarks:
            raise ValueError("benchmarks must not be empty")

        self.run_name = run_name
        self.benchmarks = benchmarks
        self.outputs_dir = ensure_dir(Path(outputs_dir))
        self.graphs_dir = ensure_dir(self.outputs_dir / "graphs")

        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_steps = int(max_steps)
        self.eval_seed = int(eval_seed)
        self.deterministic = bool(deterministic)
        self.p_action = float(p_action)
        self.verbose_print = bool(verbose_print)

        # Validate and normalize benchmark weights
        self._benchmark_weights: dict[str, dict[OpponentName, float]] = {}
        for bench in self.benchmarks:
            if not bench.opponents:
                raise ValueError(f"benchmark '{bench.name}' has no opponents")
            total = float(sum(bench.opponents.values()))
            if total <= 0:
                raise ValueError(
                    f"benchmark '{bench.name}' has non-positive total weight"
                )
            self._benchmark_weights[bench.name] = {
                opp: float(w) / total for opp, w in bench.opponents.items()
            }

        # All unique opponents across all benchmarks
        unique_opponents: set[OpponentName] = set()
        for weights in self._benchmark_weights.values():
            unique_opponents.update(weights.keys())
        self._all_opponents = sorted(unique_opponents, key=lambda x: x.name)

        # Cache single-opponent eval envs
        self._opponent_envs: dict[OpponentName, DummyVecEnv] = {
            opp: self._build_single_opponent_env(opp) for opp in self._all_opponents
        }

        # Best score per benchmark
        self._best_scores: dict[str, float] = {
            bench.name: -float("inf") for bench in self.benchmarks
        }

        # Histories
        self.opponent_history: dict[str, list[HistoryEntry]] = {
            opp.value: [] for opp in self._all_opponents
        }
        self.benchmark_history: dict[str, list[HistoryEntry]] = {
            bench.name: [] for bench in self.benchmarks
        }

    # ---------------------------------------------------------
    # env creation
    # ---------------------------------------------------------

    def _build_single_opponent_env(self, opponent_name: OpponentName) -> DummyVecEnv:
        return DummyVecEnv(
            [
                lambda opponent_name=opponent_name: make_env(
                    seed=self.eval_seed,
                    opponents={
                        opponent_name: Opponent(p_mix=1.0, p_action=self.p_action)
                    },
                    terminal_only=True,
                    max_steps=self.max_steps,
                )
            ]
        )

    # ---------------------------------------------------------
    # evaluation
    # ---------------------------------------------------------

    def _eval_single_opponent(
        self,
        opponent_name: OpponentName,
    ) -> dict[str, float]:
        terminal_infos: list[dict[str, Any]] = []

        def hook(locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
            infos = locals_.get("infos")
            dones = locals_.get("dones")
            if infos is None or dones is None:
                return
            for info, done in zip(infos, dones):
                if done and info is not None:
                    terminal_infos.append(info)

        if not isinstance(self.model, MaskablePPO):
            raise TypeError(
                f"BenchmarkEvalCallback requires MaskablePPO, got {type(self.model).__name__}"
            )

        env = self._opponent_envs[opponent_name]

        episode_rewards, _episode_lengths = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            callback=hook,
        )

        rewards = np.asarray(episode_rewards, dtype=np.float64)

        # Prefer winner_player_id if present; otherwise fall back to reward > 0
        if terminal_infos and all(
            "winner_player_id" in info for info in terminal_infos
        ):
            win_flags = np.array(
                [
                    1.0 if info.get("winner_player_id") == PlayerId.P1 else 0.0
                    for info in terminal_infos
                ],
                dtype=np.float64,
            )
        else:
            win_flags = (rewards > 0.0).astype(np.float64)

        mean_return = float(np.mean(rewards))
        return_variance = float(np.var(rewards))
        winrate = float(np.mean(win_flags))
        winrate_variance = float(np.var(win_flags))

        return {
            "mean_return": mean_return,
            "return_variance": return_variance,
            "winrate": winrate,
            "winrate_variance": winrate_variance,
        }

    def _aggregate_benchmark(
        self,
        benchmark_name: str,
        opponent_results: dict[OpponentName, dict[str, float]],
    ) -> dict[str, float]:
        weights = self._benchmark_weights[benchmark_name]

        mean_return = sum(
            weights[opp] * opponent_results[opp]["mean_return"] for opp in weights
        )
        winrate = sum(
            weights[opp] * opponent_results[opp]["winrate"] for opp in weights
        )

        # Mixture variance using second moments:
        # Var(X) = E[X^2] - (E[X])^2
        return_second_moment = sum(
            weights[opp]
            * (
                opponent_results[opp]["return_variance"]
                + opponent_results[opp]["mean_return"] ** 2
            )
            for opp in weights
        )
        return_variance = float(return_second_moment - mean_return**2)

        # For win indicator, mixture Bernoulli variance with p = weighted winrate
        winrate_variance = float(winrate * (1.0 - winrate))

        return {
            "mean_return": float(mean_return),
            "return_variance": float(max(0.0, return_variance)),
            "winrate": float(winrate),
            "winrate_variance": float(max(0.0, winrate_variance)),
        }

    def _save_best_for_benchmark(
        self,
        benchmark_name: str,
        aggregate: dict[str, float],
        opponent_results: dict[OpponentName, dict[str, float]],
    ) -> None:
        score = aggregate["winrate"]
        if score <= self._best_scores[benchmark_name]:
            return

        if not isinstance(self.model, MaskablePPO):
            raise TypeError(
                f"BenchmarkEvalCallback requires MaskablePPO, got {type(self.model).__name__}"
            )

        self._best_scores[benchmark_name] = score
        model_path = (
            build_checkpoint_dir(self.run_name, benchmark_name, checkpoint="best")
            / "model.zip"
        )
        save_model(
            self.model,
            benchmark_name=benchmark_name,
            checkpoint="best",
            run_name=self.run_name,
        )
        write_eval_txt(
            self.run_name,
            self._benchmark_weights,
            benchmark_name,
            "best",
            self.num_timesteps,
            aggregate,
            opponent_results,
        )

        if self.verbose_print:
            print(
                f"[saved] new best for '{benchmark_name}': "
                f"{model_path}.zip (winrate={score:.3f})"
            )

    # ---------------------------------------------------------
    # callback hooks
    # ---------------------------------------------------------

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Evaluate each unique opponent once
        opponent_results: dict[OpponentName, dict[str, float]] = {}
        for opp in self._all_opponents:
            result = self._eval_single_opponent(opp)
            opponent_results[opp] = result

            slug = opp.value

            self.opponent_history[slug].append(
                HistoryEntry(
                    step=int(self.num_timesteps),
                    mean_return=result["mean_return"],
                    return_variance=result["return_variance"],
                    winrate=result["winrate"],
                    winrate_variance=result["winrate_variance"],
                )
            )

            self.logger.record(
                f"bench/opponent/{slug}_mean_return", result["mean_return"]
            )
            self.logger.record(
                f"bench/opponent/{slug}_return_variance", result["return_variance"]
            )
            self.logger.record(f"bench/opponent/{slug}_winrate", result["winrate"])
            self.logger.record(
                f"bench/opponent/{slug}_winrate_variance", result["winrate_variance"]
            )

            if self.verbose_print:
                print(
                    f"{slug:<26} "
                    f"wr={result['winrate']:.3f} "
                    f"r={result['mean_return']:.4f}"
                )

        # Aggregate each benchmark
        for bench in self.benchmarks:
            aggregate = self._aggregate_benchmark(bench.name, opponent_results)

            self.benchmark_history[bench.name].append(
                HistoryEntry(
                    step=int(self.num_timesteps),
                    mean_return=aggregate["mean_return"],
                    return_variance=aggregate["return_variance"],
                    winrate=aggregate["winrate"],
                    winrate_variance=aggregate["winrate_variance"],
                )
            )

            self.logger.record(
                f"bench/{bench.name}/mean_return", aggregate["mean_return"]
            )
            self.logger.record(
                f"bench/{bench.name}/return_variance", aggregate["return_variance"]
            )
            self.logger.record(f"bench/{bench.name}/winrate", aggregate["winrate"])
            self.logger.record(
                f"bench/{bench.name}/winrate_variance",
                aggregate["winrate_variance"],
            )

            self._save_best_for_benchmark(
                benchmark_name=bench.name,
                aggregate=aggregate,
                opponent_results=opponent_results,
            )

            if self.verbose_print:
                parts = [f"[bench:{bench.name}] step={self.num_timesteps}"]
                for opp, weight in self._benchmark_weights[bench.name].items():
                    slug = opp.value
                    row = opponent_results[opp]
                    parts.append(f"{slug}={row['winrate']:.3f}@{weight:.2f}")
                parts.append(f"weighted_winrate={aggregate['winrate']:.3f}")
                parts.append(f"weighted_mean_return={aggregate['mean_return']:.4f}")

                if self.verbose_print:
                    print(
                        f"\n=== Benchmark: {bench.name} | step={self.num_timesteps} ==="
                    )

                    for opp, weight in self._benchmark_weights[bench.name].items():
                        slug = opp.value
                        row = opponent_results[opp]

                        print(
                            f"{slug:<26} "
                            f"w={weight:.2f} "
                            f"wr={row['winrate']:.3f} "
                            f"r={row['mean_return']:.4f}"
                        )

                    print("-" * 60)
                    print(
                        f"{'WEIGHTED SCORE':<26} "
                        f"winrate={aggregate['winrate']:.3f} "
                        f"mean_return={aggregate['mean_return']:.4f}"
                        "\n"
                    )

        self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        save_all_graphs(
            run_name=self.run_name,
            all_opponents=self._all_opponents,
            opponent_history=self.opponent_history,
            benchmarks=self.benchmarks,
            benchmark_history=self.benchmark_history,
        )

        # Close cached envs
        for env in self._opponent_envs.values():
            env.close()
