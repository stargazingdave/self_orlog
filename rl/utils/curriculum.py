from __future__ import annotations

from dataclasses import dataclass
from rl.env.config import OpponentName
from rl.env.env import Opponent


@dataclass
class MixPoint:
    name: str
    t: int
    mix: dict[OpponentName, float]


def normalize_mix(m: dict[OpponentName, float]) -> dict[OpponentName, float]:
    s = float(sum(max(0.0, v) for v in m.values()))
    if s <= 0:
        raise ValueError("mix weights must sum to > 0")
    return {k: float(max(0.0, v)) / s for k, v in m.items()}


def lerp_mix(
    a: dict[OpponentName, float],
    b: dict[OpponentName, float],
    alpha: float,
) -> dict[OpponentName, float]:
    """
    Linear interpolation between two opponent mixes. The mixes don't need to have the same keys, and the weights don't need to be normalized (but they must be non-negative). The output will be normalized.
    """
    keys = set(a.keys()) | set(b.keys())
    out: dict[OpponentName, float] = {}
    for k in keys:
        va = float(a.get(k, 0.0))
        vb = float(b.get(k, 0.0))
        out[k] = (1.0 - alpha) * va + alpha * vb
    return normalize_mix(out)


def to_opponent_dict(mix: dict[OpponentName, float], p_action: float = 1.0):
    return {k: Opponent(p_mix=v, p_action=p_action) for k, v in mix.items()}


def to_weights_dict(mix: dict[OpponentName, Opponent]) -> dict[OpponentName, float]:
    return {k: v.p_mix for k, v in mix.items()}
