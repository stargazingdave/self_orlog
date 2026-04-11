from __future__ import annotations

from rl.config import HistoryEntry
from rl.env.config import OpponentName
from rl.pg.ppo.utils.dirs import build_graphs_dir
from rl.utils.plot import plot_mean_return_graph, plot_winrate_graph


def save_all_graphs(
    run_name: str,
    all_opponents: list[OpponentName],
    opponent_history: dict[str, list[HistoryEntry]],
    benchmarks: list,
    benchmark_history: dict[str, list[HistoryEntry]],
) -> None:
    base_dir = build_graphs_dir(run_name)

    # -------- opponents --------
    for opp in all_opponents:
        slug = opp.value
        rows = opponent_history[slug]

        plot_mean_return_graph(
            rows=rows,
            title=f"Opponent: {slug}",
            out_path=base_dir / slug / "mean_return.png",
        )

        plot_winrate_graph(
            rows=rows,
            title=f"Opponent: {slug}",
            out_path=base_dir / slug / "winrate.png",
        )

    # -------- benchmarks --------
    for bench in benchmarks:
        rows = benchmark_history[bench.name]

        plot_mean_return_graph(
            rows=rows,
            title=f"Benchmark: {bench.name}",
            out_path=base_dir / bench.name / "mean_return.png",
        )

        plot_winrate_graph(
            rows=rows,
            title=f"Benchmark: {bench.name}",
            out_path=base_dir / bench.name / "winrate.png",
        )
