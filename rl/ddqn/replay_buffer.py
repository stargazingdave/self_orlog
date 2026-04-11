from collections import deque
import numpy as np
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=int(capacity))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def add(self, s, a, r, s2, done, next_mask):
        # store masks as packed bytes to keep memory low
        # (bool list -> uint8 array)
        nm = np.asarray(next_mask, dtype=np.uint8)
        self.buf.append((s, a, r, s2, done, nm))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done, nm = zip(*batch)

        s = torch.tensor(np.array(s, dtype=np.float32), device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.array(s2, dtype=np.float32), device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        # next masks -> bool tensor (batch, n_actions)
        nm = torch.tensor(np.stack(nm, axis=0), dtype=torch.bool, device=self.device)

        return s, a, r, s2, done, nm
