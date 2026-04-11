from __future__ import annotations

from rl.config import HistoryEntry
from rl.ddqn.utils.dirs import build_graphs_dir
from rl.env.config import OpponentName
from rl.utils.plot import plot_mean_return_graph, plot_winrate_graph


def save_all_graphs(
    run_name: str,
    all_opponents: list[OpponentName],
    opponent_history: dict[OpponentName, list[HistoryEntry]],
    history: list[HistoryEntry],
) -> None:
    base_dir = build_graphs_dir(run_name)

    # -------- opponents --------
    for opp in all_opponents:
        rows = opponent_history[opp]

        plot_mean_return_graph(
            rows=rows,
            title=f"Opponent: {opp.value}",
            out_path=base_dir / opp.value / "mean_return.png",
        )

        plot_winrate_graph(
            rows=rows,
            title=f"Opponent: {opp.value}",
            out_path=base_dir / opp.value / "winrate.png",
        )

    plot_mean_return_graph(
        rows=history,
        title=f"Run name: {run_name}",
        out_path=base_dir / run_name / "mean_return.png",
    )

    plot_winrate_graph(
        rows=history,
        title=f"Run name: {run_name}",
        out_path=base_dir / run_name / "winrate.png",
    )
