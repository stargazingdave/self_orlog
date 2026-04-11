from __future__ import annotations

import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_random_action(mask: list[bool]) -> int:
    valid = [i for i, ok in enumerate(mask) if ok]
    if not valid:
        return random.randrange(len(mask))
    return random.choice(valid)


@torch.no_grad()
def masked_greedy_action(q_values_1d: torch.Tensor, mask: list[bool]) -> int:
    """
    Selects the greedy action based on Q-values, considering a mask of valid actions.
    q_values_1d: shape (n_actions,)
    mask: list of bools of length n_actions, where True means the action is valid
    """
    q = q_values_1d.clone()
    if mask:
        m = torch.tensor(mask, dtype=torch.bool, device=q.device)
        q[~m] = -1e9
    return int(torch.argmax(q).item())


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    """
    Linearly interpolates between start and end over the given duration (in steps).
    After duration is reached, returns end.
    """
    if step >= duration:
        return end
    frac = step / float(duration)
    return start + frac * (end - start)
