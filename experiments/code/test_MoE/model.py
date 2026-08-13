import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
import math
import numpy as np
import tiktoken

data = Path("toy_datasets/tiny_stories_100M.txt").read_text(encoding="utf-8")
# print(len(data))

@dataclass
class MoeConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.1
    n_experts: int = 4

    batch_size: int = 32

device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"
DATA_DIR = Path("toy_datasets/tokenized_100M")

# ______________________________________________________________________________________________________ #
# ---------------------------------------- Class Dataloaders ------------------------------------------- #
# ______________________________________________________________________________________________________ #

class TinyStoriesDataset(Dataset):
    def __init__(self, split, block_size):
        assert split in ['train', 'val']
        self.block_size = block_size
        self._data = None
        self.path = DATA_DIR / f"{split}_100M.npy"
        self._n = len(np.load(self.path, mmap_mode='r'))

    @property
    def data(self):
        if self._data is None:
            self._data = np.load(self.path, mmap_mode='r')
        return self._data

    def __len__(self):
        return self._n - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

def data_loader(split, config, num_workers, shuffle=True):
    ds = TinyStoriesDataset(split, config.block_size)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "device"),
        drop_last=True,
    )


# ______________________________________________________________________________________________________ #
# ------------------------ Implementation of Rotary Position Embedding --------------------------------- #
# ______________________________________________________________________________________________________ #

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)