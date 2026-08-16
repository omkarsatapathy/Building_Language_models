"""
TPU (PyTorch/XLA) port of model.py — same model, same training recipe.
Tuned to run on a Colab FREE TPU runtime (v2-8, 8GB HBM/core, ~35GB host RAM).

    # Colab: Runtime > Change runtime type > TPU
    !git clone <this repo> && cd Building_Language_models
    !python experiments/code/test_MoE/model_on_tpu.py

What is different from model.py, and why:

  1. MoE routing is DENSE, not a gather/scatter loop.
     model.py does `mask.nonzero()` -> `x_flat[token_idx]`, whose shape depends
     on the data. XLA compiles for FIXED shapes, so that would retrace the graph
     on nearly every step (seconds-to-minutes each). Here every expert sees every
     token and the gate weights (zero for un-routed tokens) do the selection.
     Math is identical; cost is n_experts/top_k = 2x the FFN FLOPs. On the MXU a
     single batched matmul over all 4 experts beats 4 sparse gathers anyway.

  2. Experts are stacked Parameters + einsum, not a ModuleList of nn.Linear.
     One [E, N, C] x [E, C, H] dot_general instead of E separate small matmuls.

  3. No .item() in the hot loop. Each one forces a graph execution + host sync.
     Everything stays on device and is fetched once per step via add_step_closure.

  4. Multi-core (8 chips on a v2-8). batch_size is PER DEVICE, grads are
     all-reduced with xm.reduce_gradients before clipping, and grad_accum_steps
     is derived from the device count so the GLOBAL batch stays ~524k tokens
     whether you are on 1 chip or 8.

  5. RandomOffsetSampler instead of shuffle=True / DistributedSampler.
     Those build a torch.randperm over len(dataset) ~= 545M = 4.4GB of int64,
     per process. Times 8 processes that is ~35GB and the Colab TPU VM dies
     before step 0. See the class docstring.

  6. No torch.compile / no fused AdamW — both CUDA-only. XLA fuses when it
     compiles the graph.

Env knobs: BATCH_SIZE BLOCK_SIZE GRAD_ACCUM_STEPS TOKENS_PER_STEP MAX_STEPS
           WARMUP_STEPS EVAL_EVERY EVAL_STEPS NUM_WORKERS NPROCS DATA_LABEL
           DATA_DIR USE_BF16
"""

import os

# must be set before torch_xla is imported; harmless if already correct
os.environ.setdefault("PJRT_DEVICE", "TPU")

import csv
import time
import math
from pathlib import Path
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from swiglu import swiglu_hidden_dim

from dataclasses import dataclass


# nn.RMSNorm landed in torch 2.4; fail loudly now rather than mid-training
if not hasattr(nn, "RMSNorm"):
    raise RuntimeError(
        f"torch {torch.__version__} has no nn.RMSNorm (needs >= 2.4). "
        "On Colab pick a TPU runtime with a current torch_xla."
    )


@dataclass
class MoeConfig:
    # ~27M total params on 50304 vocab (weight-tied embeddings dominate the count)
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 6          # 6 blocks
    n_head: int = 8           # head_dim = 256/8 = 32 (even, RoPE-ok)
    n_embd: int = 256         # embedding = 50304*256 ~= 12.9M
    dropout: float = 0.0      # pretraining on ~545M tokens (~18 tok/param); no dropout

    # PER DEVICE micro-batch. 4 is sized for a Colab v2-8's 8GB/core:
    # the [B*T, 50304] logits + their fp32 cross-entropy upcast + grad dominate
    # everything else (~2.1GB at B=4, ~8.2GB at B=16 -> OOM on v2).
    # On a v4/v5e (16-32GB) raise to 16 or 32 via BATCH_SIZE.
    batch_size: int = 4
    n_experts: int = 4
    top_k: int = 2
    aux_weight: float = 0.01


# data lives under <repo>/Datasets/processed_dataset/tiny_stories_<LABEL>/
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_LABEL = os.getenv("DATA_LABEL", "all")
DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "Datasets" / "processed_dataset" / f"tiny_stories_{DATA_LABEL}"))

# ______________________________________________________________________________________________________ #
# ---------------------------------------- Class Dataloaders ------------------------------------------- #
# ______________________________________________________________________________________________________ #

class TinyStoriesDataset(Dataset):
    def __init__(self, split, block_size):
        assert split in ['train', 'val']
        self.block_size = block_size
        self._data = None
        self.path = DATA_DIR / f"{split}_{DATA_LABEL}.npy"
        if not self.path.exists():
            raise FileNotFoundError(
                f"missing {self.path}\n"
                f"Run prepare_data.py / data_processor.py first, or point DATA_DIR at the .npy folder."
            )
        self._n = len(np.load(self.path, mmap_mode='r'))

    @property
    def data(self):
        # opened lazily so each dataloader worker gets its own mmap handle
        if self._data is None:
            self._data = np.load(self.path, mmap_mode='r')
        return self._data

    def __len__(self):
        return self._n - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y


class RandomOffsetSampler(Sampler):
    """Draws random window start offsets, rank-disjoint, without a permutation.

    Why not shuffle=True / DistributedSampler: both materialize a
    torch.randperm(len(dataset)). len(dataset) here is ~545M, so that is a
    4.4GB int64 tensor -- PER PROCESS. With 8 XLA processes on a Colab TPU VM
    (~35GB RAM) it OOMs the host before the first step, and even on one process
    it stalls startup for minutes.

    Sampling offsets WITH replacement is the standard LM-pretraining shortcut
    (nanoGPT does exactly this) and costs ~1MB. Every rank draws from a
    different seed, so the ranks see different data.

    All ranks get the same __len__, which matters: XLA collectives deadlock if
    one process runs out of batches before the others.
    """
    def __init__(self, n, samples_per_epoch, block_size, rank, world, seed=1337, shuffle=True):
        self.n = n
        self.block_size = block_size
        self.rank = rank
        self.world = world
        self.seed = seed
        self.shuffle = shuffle
        self.num_samples = max(1, samples_per_epoch // world)   # identical on every rank
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + 1000 * self.epoch + self.rank)
            yield from torch.randint(0, self.n, (self.num_samples,), generator=g).tolist()
        else:
            # val: deterministic, non-overlapping windows, disjoint per rank
            for i in range(self.num_samples):
                yield ((self.rank + i * self.world) * self.block_size) % self.n


def data_loader(split, config, device, num_workers, shuffle=True):
    """Returns (MpDeviceLoader, sampler). Each core reads a disjoint stream."""
    ds = TinyStoriesDataset(split, config.block_size)
    sampler = RandomOffsetSampler(
        n=len(ds),
        samples_per_epoch=int(os.getenv("SAMPLES_PER_EPOCH", 1 << 20)),
        block_size=config.block_size,
        rank=xr.global_ordinal(),
        world=xr.world_size(),
        shuffle=shuffle,
    )
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    # MpDeviceLoader prefetches to the TPU and inserts a mark_step per batch
    return pl.MpDeviceLoader(loader, device), sampler


# ______________________________________________________________________________________________________ #
# ------------------------ Implementation of Rotary Position Embedding --------------------------------- #
# ______________________________________________________________________________________________________ #
# Step A : a function to make the cos/sin tables
def build_RoPE_cache(seq_len, head_dim, base=10000):
    # built on CPU; registered as a buffer so model.to(xla_device) carries it over.
    # (Building directly on the XLA device at __init__ time just adds graph nodes.)
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).type_as(inv_freq)
    freqs = torch.einsum('i , j -> i j', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

# Step B : a helper to shuffle the vector
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# Step C : a function to apply the rotary embedding to a tensor
def apply_rotary_pos_emb(x, cos, sin):
    # x: [B, n_head, T, head_dim]
    # cos, sin : [T, head_dim] -> broadcast over batch and head
    cos = cos[None, None, :, :].to(dtype=x.dtype)  # [1, 1, T, head_dim]
    sin = sin[None, None, :, :].to(dtype=x.dtype)  # [1, 1, T, head_dim]
    return x * cos + rotate_half(x) * sin


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

        # causal attention. On XLA this lowers to plain matmuls + mask (no flash
        # kernel), so it materializes a [B, n_head, T, T] score matrix -- 67MB at
        # B=4,T=1024 in bf16. Fine here; if you scale up, swap in
        # torch_xla.experimental.custom_kernel.flash_attention (Pallas).
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

class MoE(nn.Module):
    """Dense-dispatch MoE: every expert sees every token, gate weights select.

    model.py routes with nonzero()/index-select, which gives data-dependent
    shapes. XLA needs static shapes, so instead we build a dense [N, n_experts]
    gate matrix that is zero wherever a token was NOT in that expert's top-k,
    and fold it into the combine step. Identical result, static graph.

    The n_experts SwiGLU experts are stored as stacked [E, ...] Parameters so
    all of them run in one batched matmul.
    """
    def __init__(self, config):
        super().__init__()
        self.pre_norm = nn.RMSNorm(config.n_embd)     # pre-norm lives inside, like attn
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.dropout = config.dropout
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)

        E = config.n_experts
        C = config.n_embd
        H = swiglu_hidden_dim(C)                      # same sizing as swiglu.SwiGLUFFN
        self.hidden_features = H

        # stacked expert weights, LLaMA-style SwiGLU, no bias.
        # NOTE: raw Parameters, so GPTMoE._init_weights (which keys off nn.Linear)
        # does not touch them -- they are initialized here instead.
        self.w_gate_e  = nn.Parameter(torch.empty(E, C, H))
        self.w_value_e = nn.Parameter(torch.empty(E, C, H))
        self.w_out_e   = nn.Parameter(torch.empty(E, H, C))
        for w in (self.w_gate_e, self.w_value_e, self.w_out_e):
            nn.init.normal_(w, mean=0.0, std=0.02)

        # For Next step training, add a shared expert that is always active for every
        # token, so every token gets some gradient signal even if it is not routed to
        # any top-k expert. Left out here (not just unused) because a parameter that
        # never receives a grad has p.grad = None, which trips xm.reduce_gradients.
        # self.shared_expert = ...

    def forward(self, x):
        x = self.pre_norm(x)
        B, T, C = x.shape
        x_flat = x.view(-1, C)                                  # N tokens (N = B·T), each a C-dim vector.

        # routing math in fp32 -- under bf16 autocast the softmax/mean below lose
        # too much precision for a stable aux loss
        logits = self.gate(x_flat).float()                      # [N, n_experts]
        probs  = F.softmax(logits, dim=-1)                      # full dist over ALL experts (for P_i)
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)    # for each token, its top-k experts
        topk_gate = F.softmax(topk_val, dim=-1)                 # softmax weight per choice, e.g. [0.7, 0.3]

        # scatter the k gate weights back into a dense [N, n_experts] matrix.
        # Zero everywhere a token was not routed -> multiplying by it is exactly
        # the `out[token_idx] += w * expert(x[token_idx])` loop in model.py.
        dense_gate = torch.zeros_like(logits).scatter(1, topk_idx, topk_gate)

        # ---- all experts, one batched matmul each ----
        # left in whatever dtype autocast picked (bf16); no manual casting
        g = torch.einsum('nc,ech->enh', x_flat, self.w_gate_e)  # [E, N, H]
        v = torch.einsum('nc,ech->enh', x_flat, self.w_value_e) # [E, N, H]
        h = F.silu(g) * v                                       # SwiGLU
        y = torch.einsum('enh,ehc->enc', h, self.w_out_e)       # [E, N, C]
        y = F.dropout(y, p=self.dropout, training=self.training)

        # combine: weight each expert's output by that token's gate, sum over experts
        out = torch.einsum('enc,ne->nc', y, dense_gate.to(y.dtype))

        # ---- load-balancing auxiliary loss ----
        P = probs.mean(dim=0)                                       # [n_experts]  mean prob mass (soft, has grad)
        one_hot = F.one_hot(topk_idx, self.n_experts).sum(dim=1)    # [N, n_experts] dispatch count per token
        f = one_hot.float().mean(dim=0)                             # [n_experts]  fraction of tokens per expert
        aux_loss = self.n_experts * torch.sum(f * P)                # scalar
        # f[e] = fraction of tokens routed to expert e; sums to top_k. Balanced = top_k/n_experts.
        return out.view(B, T, C), aux_loss, f.detach()


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.moe = MoE(config)

    def forward(self, x):
        x = x + self.attn(x)    # attn owns its pre-norm
        moe_out, aux, f = self.moe(x)
        x = x + moe_out         # moe owns its pre-norm
        return x, aux, f


# ______________________________________________________________________________________________________ #
# ------------------------ Top-level model: GPT-MoE decoder -------------------------------------------- #
# ______________________________________________________________________________________________________ #

class GPTMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: input embedding and output projection share one weight matrix
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialize weights for linear and embedding layers
        # (MoE's stacked expert Parameters are initialized in MoE.__init__)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate):
        # all params that require gradients
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        # split: 2D+ tensors decay (matmuls + embeddings + stacked experts), 1D don't
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() <  2]

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # no fused=True here -- that is a CUDA-only kernel. XLA fuses the update
        # itself when it compiles the graph.
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),      # GPT-3 recipe
            eps=1e-8,
        )

        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        xm.master_print(f"decayed params:   {len(decay_params)} tensors, {num_decay:,} values")
        xm.master_print(f"un-decayed params:{len(nodecay_params)} tensors, {num_nodecay:,} values")

        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"seq len {T} > block_size {self.config.block_size}"

        x = self.drop(self.tok_emb(idx))          # [B, T, C]  — NO pos emb; RoPE handles position

        total_aux = 0.0
        route_sum = 0.0
        for block in self.blocks:
            x, aux, f = block(x)                  # residual/skip connections live INSIDE Block
            total_aux = total_aux + aux
            route_sum = route_sum + f             # [n_experts] routing fraction, per layer

        route_frac = route_sum / len(self.blocks) # [n_experts] avg fraction across layers

        x = self.final_norm(x)                    # final pre-LM-head norm
        logits = self.lm_head(x)                  # [B, T, vocab_size]

        return logits, total_aux, route_frac


# ______________________________________________________________________________________________________ #
# -------------------------------------- Training Loop ------------------------------------------------- #
# ______________________________________________________________________________________________________ #

# ---- training hyperparameters ----
max_lr        = 6e-4
min_lr        = max_lr * 0.1        # 10% of max, per GPT-3 recipe
warmup_steps  = int(os.getenv("WARMUP_STEPS", 200))
max_steps     = int(os.getenv("MAX_STEPS", 2000))
weight_decay  = 0.1
grad_clip     = 1.0

# global tokens per optimizer step, summed over every core. grad_accum_steps is
# derived from this at runtime so 1 chip and 8 chips train the same recipe.
TOKENS_PER_STEP = int(os.getenv("TOKENS_PER_STEP", 524288))

# bf16 is the native TPU compute dtype. Probe once: some torch_xla builds do not
# register the 'xla' autocast backend, and we would rather fall back than crash.
USE_BF16 = os.getenv("USE_BF16", "1") == "1"
if USE_BF16:
    try:
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            pass
    except Exception as e:                                  # pragma: no cover
        print(f"[warn] xla autocast unavailable ({e}); running fp32")
        USE_BF16 = False


def autocast_ctx():
    if USE_BF16:
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)
    return nullcontext()


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2) after training ends, hold at min_lr
    if step >= max_steps:
        return min_lr

    # 3) cosine decay from max_lr -> min_lr in between
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def infinite_batches(loader, sampler):
    # set_epoch reseeds the offset draw, otherwise every pass repeats the same windows
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for xb, yb in loader:
            yield xb, yb
        epoch += 1


def train_step(model, batch_iter, optimizer, grad_accum_steps, config):
    optimizer.zero_grad(set_to_none=True)

    loss_accum = 0.0
    aux_accum  = 0.0
    ce_micro, aux_micro, route_micro = [], [], []

    for micro in range(grad_accum_steps):
        xb, yb = next(batch_iter)   # MpDeviceLoader already put these on the TPU

        with autocast_ctx():
            logits, aux, route = model(xb)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                ignore_index=-1,
            )
            loss = ce_loss + config.aux_weight * aux

        loss = loss / grad_accum_steps
        loss_accum = loss_accum + loss.detach()
        aux_accum  = aux_accum + (aux.detach() / grad_accum_steps)

        # per-micro record, kept AS TENSORS. Calling .item() here would force a
        # host sync per micro-batch and serialize the whole pipeline.
        ce_micro.append(ce_loss.detach())
        aux_micro.append(aux.detach())
        route_micro.append(route)

        loss.backward()

    micro_stats = (
        torch.stack(ce_micro),      # [grad_accum]
        torch.stack(aux_micro),     # [grad_accum]
        torch.stack(route_micro),   # [grad_accum, n_experts]
    )
    return loss_accum, aux_accum, micro_stats


@torch.no_grad()
def evaluate(model, val_iter, eval_steps, config):
    model.eval()                       # dropout OFF
    loss_accum = 0.0
    for _ in range(eval_steps):
        xb, yb = next(val_iter)
        with autocast_ctx():
            logits, _, _ = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                ignore_index=-1,
            )
        loss_accum = loss_accum + loss.detach() / eval_steps
    model.train()                      # back to train mode
    # average across cores so every rank prints/saves the same number.
    # mesh_reduce is collective: EVERY rank must reach it or the run hangs.
    return xm.mesh_reduce("val_loss", loss_accum.item(), lambda vals: sum(vals) / len(vals))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    xm.master_print(f"total params: {total:,} ({total/1e6:.2f}M)")
    return total


def save_checkpoint(model, optimizer, step, val_loss, config, ckpt_dir):
    ckpt = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "val_loss":  val_loss,
        "config":    config,
    }
    path = ckpt_dir / f"moe_tpu_step{step}_with{val_loss:.4f}.pt"
    # xm.save moves tensors off the TPU and writes from the master rank only.
    # Also collective -- every rank must call it.
    xm.save(ckpt, str(path))
    xm.master_print(f"  💾 saved {path}  (val_loss {val_loss:.4f})")


# ______________________________________________________________________________________________________ #
# ------------------------ Per-core entry point (one process per TPU chip) ----------------------------- #
# ______________________________________________________________________________________________________ #

def _mp_fn(index):
    device = xm.xla_device()
    world = xr.world_size()
    is_master = xr.global_ordinal() == 0

    # identical seed on every core -> identical init, so replicas stay in sync
    torch.manual_seed(1337)

    # ---- build everything ----
    config = MoeConfig()
    # optional env overrides (used by run.sh smoke tests)
    if os.getenv("BATCH_SIZE"): config.batch_size = int(os.getenv("BATCH_SIZE"))
    if os.getenv("BLOCK_SIZE"): config.block_size = int(os.getenv("BLOCK_SIZE"))

    # keep the global batch constant regardless of how many chips we got
    per_step = config.batch_size * config.block_size * world
    grad_accum_steps = int(os.getenv("GRAD_ACCUM_STEPS", max(1, TOKENS_PER_STEP // per_step)))
    tokens = per_step * grad_accum_steps

    xm.master_print(
        f"world={world} chips | per-device batch={config.batch_size} x {config.block_size} "
        f"| grad_accum={grad_accum_steps} | {tokens:,} tokens/step | bf16={USE_BF16}"
    )

    model = GPTMoE(config).to(device)
    # no torch.compile -- XLA traces and compiles the graph on its own
    optimizer = model.configure_optimizers(weight_decay, max_lr)

    # 0 by default: a Colab TPU VM has few vCPUs, and 8 procs x N workers thrash.
    num_workers = int(os.getenv("NUM_WORKERS", 0))
    train_loader, train_sampler = data_loader("train", config, device, num_workers, shuffle=True)
    batch_iter = infinite_batches(train_loader, train_sampler)

    val_loader, val_sampler = data_loader("val", config, device, num_workers, shuffle=False)
    val_iter = infinite_batches(val_loader, val_sampler)

    eval_every = int(os.getenv("EVAL_EVERY", 250))   # steps
    eval_steps = int(os.getenv("EVAL_STEPS", 20))    # micro-batches per eval

    CKPT_DIR = REPO_ROOT / "checkpoints"
    LOG_DIR = REPO_ROOT / "logs"
    if is_master:
        CKPT_DIR.mkdir(exist_ok=True)
        LOG_DIR.mkdir(exist_ok=True)

    count_params(model)

    # ---- CSV micro-batch logger (master only) ----
    csv_path = LOG_DIR / "train_micro_log_tpu.csv"
    csv_file = csv_writer = None
    if is_master:
        CSV_FIELDS = (
            ["step", "micro", "ce_loss", "aux_loss", "total_loss"]            # micro-level
            + [f"expert{e}_frac" for e in range(config.n_experts)]            # MoE routing
            + ["step_ce_loss", "step_aux_loss"]                              # step-level means
            + ["lr", "grad_norm", "dt_ms", "tok_per_sec", "tokens"]          # step-level meta
            + ["wall_time_s"]                                                # seconds since start
        )
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

    run_start = time.time()

    def log_step(step, lr, t0, loss_accum, aux_accum, norm, ce_m, aux_m, route_m):
        """Runs on the host AFTER the graph for this step has executed.
        This is the only place tensors get pulled off the TPU."""
        dt = time.time() - t0
        tok_per_sec = tokens / dt
        step_loss = loss_accum.item()
        step_aux  = aux_accum.item()
        grad_norm = norm.item()

        if step % 10 == 0:
            xm.master_print(
                f"step {step:5d} | loss {step_loss:.4f} | aux {step_aux:.4f} | "
                f"lr {lr:.2e} | norm {grad_norm:.2f} | {dt*1000:.0f}ms | {tok_per_sec:,.0f} tok/s"
            )

        if csv_writer is None:
            return
        ce_m, aux_m, route_m = ce_m.cpu(), aux_m.cpu(), route_m.cpu()
        wall = time.time() - run_start
        for micro in range(ce_m.shape[0]):
            rec = {
                "micro":         micro,
                "ce_loss":       ce_m[micro].item(),
                "aux_loss":      aux_m[micro].item(),
                "total_loss":    (ce_m[micro] + config.aux_weight * aux_m[micro]).item(),
                "step":          step,
                "step_ce_loss":  step_loss - config.aux_weight * step_aux,
                "step_aux_loss": step_aux,
                "lr":            lr,
                "grad_norm":     grad_norm,
                "dt_ms":         dt * 1000,
                "tok_per_sec":   tok_per_sec,
                "tokens":        tokens,
                "wall_time_s":   wall,
            }
            # per-expert routing fraction (avg over layers); balanced = top_k/n_experts
            for e in range(config.n_experts):
                rec[f"expert{e}_frac"] = route_m[micro, e].item()
            csv_writer.writerow(rec)
        csv_file.flush()     # flush each step so a crash keeps your data

    # ---- the loop ----
    for step in range(max_steps):
        t0 = time.time()

        # 1) set LR for this step
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 2) accumulate grads over micro-batches
        loss_accum, aux_accum, micro_stats = train_step(
            model, batch_iter, optimizer, grad_accum_steps, config
        )

        # 3) all-reduce grads across cores, THEN clip the (global) norm to 1.0
        xm.reduce_gradients(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # 4) take the step. reduce_gradients already ran, so plain .step() here
        #    (xm.optimizer_step would reduce a second time).
        optimizer.step()

        # 5) queue the metric fetch, then cut the graph. The closure fires once
        #    this step's graph has actually run, so it never stalls the pipeline.
        xm.add_step_closure(
            log_step,
            args=(step, lr, t0, loss_accum, aux_accum, norm, *micro_stats),
        )
        xm.mark_step()

        # ---- periodic eval + checkpoint (collective: every rank runs these) ----
        if step > 0 and step % eval_every == 0:
            val_loss = evaluate(model, val_iter, eval_steps, config)
            xm.master_print(f"  >>> eval @ step {step}: val_loss {val_loss:.4f}")
            save_checkpoint(model, optimizer, step, val_loss, config, CKPT_DIR)

    xm.wait_device_ops()
    if is_master:
        csv_file.close()
        xm.master_print(f"📊 wrote micro-batch log → {csv_path}")


if __name__ == "__main__":
    # NPROCS=1 forces single-process (one chip). Leave unset to use every chip.
    #
    # Colab caveat: multiprocess spawn needs the TPU to be UNTOUCHED by the
    # parent kernel. If any earlier notebook cell did `import torch_xla` and
    # poked a device, the child processes cannot claim the chips and libtpu
    # fails with e.g. "Expected 4 worker addresses, got 1". Restart the runtime
    # and launch this as a subprocess (!python model_on_tpu.py) — or run with
    # NPROCS=1, which skips the multiprocess path entirely.
    env_nprocs = os.getenv("NPROCS")
    nprocs = int(env_nprocs) if env_nprocs else None

    try:
        xmp.spawn(_mp_fn, args=(), nprocs=nprocs)
    except RuntimeError as e:
        if env_nprocs or "TPU initialization failed" not in str(e):
            raise
        print(
            f"\n[warn] multi-chip spawn failed:\n  {e}\n"
            "[warn] falling back to a single chip. This is ~n_chips slower; see the\n"
            "       note above if you want all of them. If this fallback also fails,\n"
            "       restart the Colab runtime (the TPU is held by another process).\n"
        )
        xmp.spawn(_mp_fn, args=(), nprocs=1)
