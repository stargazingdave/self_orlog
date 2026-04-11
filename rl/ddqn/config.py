from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DDQNConfig:
    hidden_dim: int = 512
    n_layers: int = 3
    lr: float = 2.9380279387035334e-05
    gamma: float = 0.965459808211767
    batch_size: int = 256
    buffer_size: int = 500000
    eps_start: float = 0.9832442640800422
    eps_end: float = 0.03698712885426209
    eps_decay_steps: int = 50000
    target_update_freq: int = 10000
    learning_starts: int = 6000
    train_freq: int = 2
