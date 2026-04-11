from __future__ import annotations

from typing import Callable
import numpy as np
import torch

from rl.config import HistoryEntry
from rl.utils.curriculum import normalize_mix
from rl.ddqn.eval.aggregate_eval_score import aggregate_eval_score
from rl.env.config import OpponentName
from game.types.players import PlayerId
from rl.env.env import Opponent, OrlogEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_fixed_opponents(
    q,
    step: int,
    opponents_history: dict[OpponentName, list[HistoryEntry]],
    aggregate_history: list[HistoryEntry],
    opponents: dict[OpponentName, Opponent],
    masked_greedy_action,
    seeds: tuple[int, ...] = (5000, 6000, 7000),
    games_per_opponent: int = 10,
    max_env_steps_per_episode: int = 200,
    print_summary: bool = True,
    configure_env: Callable[[OrlogEnv], None] | None = None,
) -> tuple[
    dict[OpponentName, HistoryEntry],
    HistoryEntry,
]:
    """
    Best-of-both:
      - q.eval() for deterministic inference (important if Dropout/BN exist)
      - handles truncation (done = done or trunc)
      - robust n_actions inference (doesn't assume action_space.n exists)
      - updates history
      - optionally saves plots as PNG (script-friendly)
      - optionally prints summary
    """
    q.eval()
    results: dict[
        OpponentName,
        HistoryEntry,
    ] = {}

    for opponent in opponents:
        wins: list[int] = []
        returns: list[float] = []

        for seed in seeds:
            env = OrlogEnv(
                seed=seed,
                agent_player_id=PlayerId.P1,
                max_env_steps_per_episode=max_env_steps_per_episode,
                opponents={opponent: Opponent(p_mix=1.0, p_action=1.0)},
            )

            if configure_env is not None:
                configure_env(env)

            # Infer n_actions once per env instance
            n_actions_attr = getattr(getattr(env, "action_space", None), "n", None)
            if n_actions_attr is not None:
                n_actions = int(n_actions_attr)
            else:
                # Fallback for non-Discrete spaces that might expose .shape
                shape = getattr(getattr(env, "action_space", None), "shape", None)
                if shape and len(shape) == 1:
                    n_actions = int(shape[0])
                else:
                    raise ValueError(
                        "Could not infer number of actions from env.action_space"
                    )

            for _ in range(games_per_opponent):
                obs, info = env.reset()
                done = False
                total_reward = 0.0
                winner = None

                while not done:
                    mask = info.get("action_mask", [True] * n_actions)
                    with torch.no_grad():
                        s_t = torch.tensor(
                            obs, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        action = masked_greedy_action(q(s_t)[0], mask)

                    obs, r, done, trunc, info = env.step(action)
                    total_reward += float(r)
                    done = bool(done) or bool(trunc)

                    # NEW: capture winner from env info
                    if done:
                        winner = info.get("winner_player_id", None)

                # NEW: win = winner matches the agent id (P1 in your eval env)
                wins.append(int(winner == PlayerId.P1))
                returns.append(total_reward)

        winrate = float(np.mean(wins)) if wins else 0.0
        mean_return = float(np.mean(returns)) if returns else 0.0
        return_variance = float(np.var(returns)) if returns else 0.0
        winrate_variance = float(np.var(wins)) if wins else 0.0
        results[opponent] = HistoryEntry(
            step=step,
            winrate=winrate,
            mean_return=mean_return,
            return_variance=return_variance,
            winrate_variance=winrate_variance,
        )

        # Ensure per-opponent history buckets exist
        if opponent not in opponents_history:
            opponents_history[opponent] = []

        # Update per-opponent history
        opponents_history[opponent].append(
            HistoryEntry(
                step=step,
                winrate=winrate,
                mean_return=mean_return,
                return_variance=return_variance,
                winrate_variance=results[opponent].winrate_variance,
            )
        )

    # Aggregate
    weights = normalize_mix({opp: opponents[opp].p_mix for opp in opponents})
    aggregate = aggregate_eval_score(results, weights)

    aggregate_history.append(
        HistoryEntry(
            step=step,
            winrate=aggregate.winrate,
            mean_return=aggregate.mean_return,
            return_variance=aggregate.return_variance,
            winrate_variance=aggregate.winrate_variance,
        )
    )

    if print_summary:
        print(f"Step {step} evaluation:")
        for opp in opponents:
            wr = results[opp].winrate
            mr = results[opp].mean_return
            print(f"  {opp:18s} winrate={wr:.3f}, mean_return={mr:.3f}")

    return results, aggregate
