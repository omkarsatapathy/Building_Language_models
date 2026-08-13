import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from swiglu import SwiGLUFFN

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

    batch_size: int = 32
    #MoE specific parameters
    n_experts: int = 4
    top_k: int = 2          # each token routes to 2 of the 4 experts


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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.pre_norm = nn.RMSNorm(config.n_embd)

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
        x = self.pre_norm(x)  
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



# ______________________________________________________________________________________________________ #
# ------------------------ Implementation of Mixture of Experts ---------------------------------------- #
# ______________________________________________________________________________________________________ #

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = SwiGLUFFN(config.n_embd, dropout=config.dropout)

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = nn.RMSNorm(config.n_embd)     # pre-norm lives inside, like attn
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.shared_expert = Expert(config)      # always active, no gate

    def forward(self, x):
        x = self.pre_norm(x)
        B, T, C = x.shape
        x_flat = x.view(-1, C)                                  # N tokens (N = B·T), each a C-dim vector.

        logits = self.gate(x_flat)                              # [N, n_experts]
        probs  = F.softmax(logits, dim=-1)                      # full dist over ALL experts (for P_i)
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)    # for each token, the indices of its top-k experts. e.g. row for token 5 might be [2, 0] (experts 2 and 0).
        topk_gate = F.softmax(topk_val, dim=-1)                 # — the softmax weight for each of those k choices. e.g. [0.7, 0.3]
        out = torch.zeros_like(x_flat)                          # This is where results get summed into. A token routed to 2 experts will get two additions here.

        #For Next step training, we will also add a shared expert that is always active for every token. This is to ensure that every token gets some gradient signal, even if it is not routed to any of the top-k experts. This is a common technique in MoE models to prevent dead experts and improve training stability.
        # shared = self.shared_expert(x_flat)
        # out = out + shared                                      # every token gets shared + its top-k routed

        for e in range(self.n_experts):                         # Handle one expert at a time.
            mask = (topk_idx == e)                              # produces a matrix like boolean marking where in the top-k lists expert e appears. Example with N=3, k=2:
            if mask.any():
                token_idx, slot = mask.nonzero(as_tuple=True)
                weights = topk_gate[token_idx, slot].unsqueeze(-1)  # For each selected token, fetch its gate weight for this specific expert.
                out[token_idx] += weights * self.experts[e](x_flat[token_idx])

        # ---- load-balancing auxiliary loss ----
        P = probs.mean(dim=0)                                       # [n_experts]  mean prob mass (soft, has grad)
        one_hot = F.one_hot(topk_idx, self.n_experts).sum(dim=1)    # [N, n_experts] dispatch count per token
        f = one_hot.float().mean(dim=0)                             # [n_experts]  fraction of tokens per expert
        aux_loss = self.n_experts * torch.sum(f * P)                # scalar
        return out.view(B, T, C), aux_loss

class Block(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.moe = MoE(config)

    def forward(self, x):
        x = x + self.attn(x)    # attn owns its pre-norm
        moe_out, aux = self.moe(x)
        x = x + moe_out     # moe owns its pre-norm
        return x, aux
