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
# Step A : a function to make the cos/sin tables
def build_RoPE_cache(seq_len, head_dim, base = 10000, device = device):
    #Note that Head dim must be even for RoPE to work because frequency per eachpair = head_dim/2
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum('i , j -> i j', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin
# Step B : a helper to shuffle the vector
def rotate_half(x):
    x1,x2 = x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)   
# Step C : a function to apply the rotary embedding to a tensor
def apply_rotary_pos_emb(x, cos, sin):
    #x: [B, n_head,T, head_dim]
    #cos, sin : [T, head_dim] -> bradcast over batch and head
    T = x.shape[-2]
    cos = cos[None, None, :,:].to(dtype=x.dtype)  # [1, 1, T, head_dim]
    sin = sin[None, None, :,:].to(dtype=x.dtype)  # [1, 1, T, head_dim]
    return x*cos +rotate_half(x)*sin 


# ______________________________________________________________________________________________________ #
# ------------------------ Implementation Causal Self Attention ---------------------------------------- #
# ______________________________________________________________________________________________________ #

import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        # one projection for q, k, v (split later)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # QK-norm: normalize over head_dim, per head
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE cache, built once, moves with the model, not saved in state_dict
        cos, sin = build_RoPE_cache(config.block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.c_attn(x).split(C, dim=2)     # each [B, T, C]

            # -> [B, n_head, T, head_dim]
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # QK-norm (over head_dim) BEFORE RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)

            # RoPE on q and k, sliced to current T
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

            # causal attention (Flash kernel when available)
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )

            # -> [B, T, C]
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y
