from __future__ import annotations

from pathlib import Path

from rl.config import HistoryEntry
from rl.env.config import OpponentName


def write_eval_txt(
    base_path: Path | str,
    weights: dict[OpponentName, float],
    step: int,
    aggregate: HistoryEntry,
    opponent_results: dict[OpponentName, HistoryEntry],
) -> None:
    path = Path(base_path) / "eval.txt"

    lines: list[str] = [
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
            f"step={row.step} "
            f"mean_return={row.mean_return:.4f} "
            f"return_variance={row.return_variance:.4f} "
            f"winrate={row.winrate:.4f} "
            f"winrate_variance={row.winrate_variance:.4f}"
        )

    lines.extend(
        [
            "",
            "aggregate:",
            (
                f"  step={aggregate.step} "
                f"mean_return={aggregate.mean_return:.4f} "
                f"return_variance={aggregate.return_variance:.4f} "
                f"winrate={aggregate.winrate:.4f} "
                f"winrate_variance={aggregate.winrate_variance:.4f}"
            ),
        ]
    )

    (path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] evaluation results: " f"{path}")
