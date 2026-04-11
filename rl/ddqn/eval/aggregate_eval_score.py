from __future__ import annotations

from rl.ddqn.model import HistoryEntry
from rl.env.config import OpponentName


def aggregate_eval_score(
    results: dict[
        OpponentName,
        HistoryEntry,
    ],
    weights: dict[OpponentName, float],
) -> HistoryEntry:
    mean_return = sum(weights[opp] * results[opp].mean_return for opp in weights)
    winrate = sum(weights[opp] * results[opp].winrate for opp in weights)

    # Mixture variance using second moments:
    # Var(X) = E[X^2] - (E[X])^2
    return_second_moment = sum(
        weights[opp] * (results[opp].return_variance + results[opp].mean_return ** 2)
        for opp in weights
    )
    return_variance = float(return_second_moment - mean_return**2)

    # For win indicator, mixture Bernoulli variance with p = weighted winrate
    winrate_variance = float(winrate * (1.0 - winrate))

    steps = results[next(iter(results))].step

    return HistoryEntry(
        step=steps,
        mean_return=float(mean_return),
        return_variance=float(max(0.0, return_variance)),
        winrate=float(winrate),
        winrate_variance=float(max(0.0, winrate_variance)),
    )
