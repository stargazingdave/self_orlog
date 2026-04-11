from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HistoryEntry:
    step: int
    winrate: float
    mean_return: float
    return_variance: float
    winrate_variance: float
