from __future__ import annotations

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from rl.config import HistoryEntry


def plot_mean_return_graph(
    rows: list[HistoryEntry],
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        return

    steps = np.array([r.step for r in rows], dtype=float)
    mean_return = np.array([r.mean_return for r in rows], dtype=float)
    return_var = np.array([r.return_variance for r in rows], dtype=float)

    # Protect against tiny negative values from numeric issues
    return_std = np.sqrt(np.maximum(return_var, 0.0))

    lower = mean_return - return_std
    upper = mean_return + return_std

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, mean_return, label="mean_return")
    ax.fill_between(steps, lower, upper, alpha=0.2, label="±1 std")

    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean_return")
    ax.set_title(title + " (mean return)")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_winrate_graph(
    rows: list[HistoryEntry],
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        return

    steps = np.array([r.step for r in rows], dtype=float)
    winrate = np.array([r.winrate for r in rows], dtype=float)
    winrate_var = np.array([r.winrate_variance for r in rows], dtype=float)

    # Protect against tiny negative values from numeric issues
    winrate_std = np.sqrt(np.maximum(winrate_var, 0.0))

    lower = np.clip(winrate - winrate_std, 0.0, 1.0)
    upper = np.clip(winrate + winrate_std, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, winrate, label="winrate")
    ax.fill_between(steps, lower, upper, alpha=0.2, label="±1 std")

    ax.set_xlabel("timesteps")
    ax.set_ylabel("winrate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title + " (winrate)")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
