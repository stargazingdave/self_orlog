from __future__ import annotations

from typing import Literal

from rl.env.config import OpponentName
from rl.pg.ppo.utils.dirs import build_checkpoint_dir


def write_eval_txt(
    run_name: str,
    benchmark_weights: dict[str, dict[OpponentName, float]],
    benchmark_name: str,
    checkpoint: Literal["best", "last"],
    step: int,
    aggregate: dict[str, float],
    opponent_results: dict[OpponentName, dict[str, float]],
) -> None:
    path = build_checkpoint_dir(run_name, benchmark_name, checkpoint) / "eval.txt"

    weights = benchmark_weights[benchmark_name]

    lines: list[str] = [
        f"benchmark={benchmark_name}",
        f"step={step}",
        "",
        "per_opponent:",
    ]

    for opp, weight in weights.items():
        slug = opp.value
        row = opponent_results[opp]
        lines.append(
            f"  {slug:<24} "
            f"weight={weight:.4f} "
            f"mean_return={row['mean_return']:.4f} "
            f"return_variance={row['return_variance']:.4f} "
            f"winrate={row['winrate']:.4f} "
            f"winrate_variance={row['winrate_variance']:.4f}"
        )

    lines.extend(
        [
            "",
            "aggregate:",
            (
                f"  mean_return={aggregate['mean_return']:.4f} "
                f"return_variance={aggregate['return_variance']:.4f} "
                f"winrate={aggregate['winrate']:.4f} "
                f"winrate_variance={aggregate['winrate_variance']:.4f}"
            ),
        ]
    )

    (path).write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[saved] evaluation results for checkpoint: {checkpoint} for '{benchmark_name}': "
        f"{path}"
    )
