import os
import time
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


@dataclass
class MoeConfig:
    # ~39M total params (~32.5M active) @ n_embd=320, n_layer=8, vocab 8192
    vocab_size: int = 8192    # == tinystories_bpe_8k.json n_vocab; already a multiple of 64
    block_size: int = 1024
    n_layer: int = 8          # 8 blocks
    n_head: int = 8           # head_dim = 320/8 = 40 (even, RoPE-ok)
    n_embd: int = 320         # embedding = 8192*320 ~= 2.62M
    dropout: float = 0.0      # pretraining on ~545M tokens (~18 tok/param); no dropout

    batch_size: int = 64      # H100 80GB: this model is tiny, room to spare
    n_experts: int = 4
    top_k: int = 2
    aux_weight: float = 0.01



device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"

# BPE-tokenized instruct corpus (uint16 .npy) lives directly under <repo>/Datasets/
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "Datasets"

# ______________________________________________________________________________________________________ #
# ---------------------------------------- Class Dataloaders ------------------------------------------- #
# ______________________________________________________________________________________________________ #

class TinyStoriesDataset(Dataset):
    def __init__(self, split, block_size):
        assert split in ['train', 'val']
        self.block_size = block_size
        self._data = None
        self.path = DATA_DIR / f"tinystories_instruct_{split}_uint16.npy"
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
        x = x + moe_out     # moe owns its pre-norm
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
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # all params that require gradients
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        # split: 2D+ tensors decay (matmuls + embeddings), 1D don't (biases, norms)
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() <  2]

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # fused AdamW = single CUDA kernel for the update; only on CUDA
        use_fused = (device == "cuda")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),      # GPT-3 recipe
            eps=1e-8,
            fused=use_fused,
        )

        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"decayed params:   {len(decay_params)} tensors, {num_decay:,} values")
        print(f"un-decayed params:{len(nodecay_params)} tensors, {num_nodecay:,} values")

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

# ---- training hyperparameters (single H100) ----
max_lr        = 6e-4
min_lr        = max_lr * 0.1        # 10% of max, per GPT-3 recipe
warmup_steps  = int(os.getenv("WARMUP_STEPS", 200))
max_steps     = int(os.getenv("MAX_STEPS", 2000))          # ~1.05B tokens (~2 epochs of 545M)
weight_decay  = 0.1
grad_clip     = 1.0
grad_accum_steps = int(os.getenv("GRAD_ACCUM_STEPS", 8))   # 64*1024*8 ~= 524k tokens/step

# autocast dtype: bf16 on CUDA, fp32 elsewhere (MPS bf16 support is spotty)
autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

from contextlib import nullcontext
def autocast_ctx():
    # autocast only supports cuda/cpu here; on MPS it errors, so no-op there
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
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


def infinite_batches(loader):
    while True:
        for xb, yb in loader:
            yield xb, yb

def train_step(model, batch_iter, optimizer, grad_accum_steps, config):
    optimizer.zero_grad(set_to_none=True)

    loss_accum = 0.0
    aux_accum  = 0.0
    micro_logs = []

    for micro in range(grad_accum_steps):
        xb, yb = next(batch_iter)
        xb, yb = xb.to(device), yb.to(device)

        with autocast_ctx():
            logits, aux, route = model(xb)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                ignore_index=-1,
            )
            loss = ce_loss + config.aux_weight * aux

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        aux_accum  += (aux.detach() / grad_accum_steps)

        # per-micro record (raw, unscaled numbers — most informative)
        rec = {
            "micro":      micro,
            "ce_loss":    ce_loss.item(),
            "aux_loss":   aux.item(),
            "total_loss": (ce_loss + config.aux_weight * aux).item(),
        }
        # per-expert routing fraction (avg over layers); balanced = top_k/n_experts
        for e in range(config.n_experts):
            rec[f"expert{e}_frac"] = route[e].item()
        micro_logs.append(rec)

        loss.backward()

    return loss_accum, aux_accum, micro_logs


@torch.no_grad()
def evaluate(model, val_iter, eval_steps, config):
    model.eval()                       # dropout OFF
    loss_accum = 0.0
    for _ in range(eval_steps):
        xb, yb = next(val_iter)
        xb, yb = xb.to(device), yb.to(device)
        with autocast_ctx():
            logits, _, _ = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                ignore_index=-1,
            )
        loss_accum += loss.detach() / eval_steps
    model.train()                      # back to train mode
    return loss_accum


# ---- build everything ----
config    = MoeConfig()
# optional env overrides (used by run.sh smoke to fit on MPS)
if os.getenv("BATCH_SIZE"): config.batch_size = int(os.getenv("BATCH_SIZE"))
if os.getenv("BLOCK_SIZE"): config.block_size = int(os.getenv("BLOCK_SIZE"))
model     = GPTMoE(config).to(device)
model = torch.compile(model, mode="max-autotune")

optimizer = model.configure_optimizers(weight_decay, max_lr, device)

num_workers = int(os.getenv("NUM_WORKERS", 0))
train_loader = data_loader("train", config, num_workers=num_workers, shuffle=True)
batch_iter   = infinite_batches(train_loader)

val_loader = data_loader("val", config, num_workers=num_workers, shuffle=False)
val_iter   = infinite_batches(val_loader)

eval_every = int(os.getenv("EVAL_EVERY", 250))   # steps
eval_steps = int(os.getenv("EVAL_STEPS", 20))    # micro-batches per eval

CKPT_DIR = REPO_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

def count_params(model):
    total  = sum(p.numel() for p in model.parameters())
    active = total  # note: MoE stores all experts but routes to top_k
    print(f"total params: {total:,} ({total/1e6:.2f}M)")
    return total

count_params(model)


def save_checkpoint(model, optimizer, step, val_loss):
    ckpt = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "val_loss":  val_loss,
        "config":    config,
    }
    path = CKPT_DIR / f"moe_step{step}_with{val_loss:.4f}.pt"
    torch.save(ckpt, path)
    print(f"  💾 saved {path}  (val_loss {val_loss:.4f})")


# ---- CSV micro-batch logger ----
import csv

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
csv_path = LOG_DIR / "train_micro_log.csv"

# column order for the big CSV
CSV_FIELDS = (
    ["step", "micro", "ce_loss", "aux_loss", "total_loss"]            # micro-level
    + [f"expert{e}_frac" for e in range(config.n_experts)]            # MoE routing
    + ["step_ce_loss", "step_aux_loss"]                              # step-level means
    + ["lr", "grad_norm", "dt_ms", "tok_per_sec", "tokens"]          # step-level meta
    + ["wall_time_s"]                                                # seconds since start
)

csv_file   = open(csv_path, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
csv_writer.writeheader()

run_start = time.time()


# ---- the loop ----
for step in range(max_steps):
    t0 = time.time()

    # 1) set LR for this step (from Piece 2)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # 2) accumulate grads over micro-batches (from Piece 4)
    loss_accum, aux_accum, micro_logs = train_step(
        model, batch_iter, optimizer, grad_accum_steps, config
    )

    # 3) clip global grad norm to 1.0 — tames spikes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 4) take the step
    optimizer.step()

    # 5) timing + throughput
    if device == "cuda":
        torch.cuda.synchronize()      # wait for GPU to actually finish
    dt = time.time() - t0
    tokens = config.batch_size * config.block_size * grad_accum_steps
    tok_per_sec = tokens / dt

    # 5b) write every micro-batch row, enriched with step-level fields
    for rec in micro_logs:
        rec.update({
            "step":          step,
            "step_ce_loss":  (loss_accum.item() - config.aux_weight * aux_accum.item()),
            "step_aux_loss": aux_accum.item(),
            "lr":            lr,
            "grad_norm":     norm.item(),
            "dt_ms":         dt * 1000,
            "tok_per_sec":   tok_per_sec,
            "tokens":        tokens,
            "wall_time_s":   time.time() - run_start,
        })
        csv_writer.writerow(rec)
    csv_file.flush()     # flush each step so a crash keeps your data

    # 6) log
    if step % 10 == 0:
        print(
            f"step {step:5d} | loss {loss_accum:.4f} | aux {aux_accum:.4f} | "
            f"lr {lr:.2e} | norm {norm:.2f} | {dt*1000:.0f}ms | {tok_per_sec:,.0f} tok/s"
        )
    # ---- periodic eval + checkpoint ----
    if step > 0 and step % eval_every == 0:
        val_loss = evaluate(model, val_iter, eval_steps, config)
        print(f"  >>> eval @ step {step}: val_loss {val_loss:.4f}")
        save_checkpoint(model, optimizer, step, val_loss)

csv_file.close()
print(f"📊 wrote micro-batch log → {csv_path}")

