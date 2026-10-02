"""Fixed token order shared by every run, whatever its batch size.

train.bin is cut into non-overlapping rows of seq_len+1 tokens and the rows are shuffled once with
a fixed seed. Step k of a run with batch B takes rows [kB, (k+1)B). Two runs with different batch
sizes therefore read the same rows in the same order, grouped differently, so after N tokens they
have seen exactly the same text.
"""
import json
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parents[1] / "data"


def meta():
    return json.loads((DATA / "meta.json").read_text())


class Rows:
    def __init__(self, split, seq_len, seed=1337):
        arr = np.memmap(DATA / f"{split}.bin", dtype=np.uint16, mode="r")
        n = len(arr) // (seq_len + 1)
        self.rows = arr[: n * (seq_len + 1)].reshape(n, seq_len + 1)
        self.order = np.random.default_rng(seed).permutation(n) if split == "train" else np.arange(n)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.order)

    def batch(self, start, size, device):
        idx = np.sort(self.order[start:start + size])
        b = torch.from_numpy(self.rows[idx].astype(np.int64))
        b = b.to(device, non_blocking=True)
        return b[:, :-1].contiguous(), b[:, 1:].contiguous()
