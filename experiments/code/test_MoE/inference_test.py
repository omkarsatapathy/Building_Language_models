"""
Inference / sanity check for the trained GPT-MoE checkpoint.

Why self-contained: model.py runs its training loop at module level (no
`__main__` guard), so importing it would start training. We therefore redefine
the model here. `TinyMoeConfig` is defined in this file so the pickled config object
inside the checkpoint unpickles cleanly (it was saved from __main__).

What it does:
  1. Pull a random story from the toy TinyStories dataset.
  2. Feed the model the first 20 words as a prompt.
  3. Let the model generate the next ~50 tokens.
  4. Print, side by side:
        - the 20-word prompt
        - the MODEL's continuation
        - the REAL continuation from the dataset (ground truth)
     so you can eyeball how close the model is.

Run:  python experiments/code/test_MoE/inference_test.py
"""

import os
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from swiglu import SwiGLUFFN   # same folder as this script

torch.set_float32_matmul_precision("high")

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT_PATH = REPO_ROOT / "logs" / "30M param_moe_step16500_with1.2656_500M_sample.pt"

# a couple of candidate story files (first that exists is used)
STORY_CANDIDATES = [
    REPO_ROOT / "toy_datasets" / "tiny_stories_100M.txt",
    REPO_ROOT / "toy_datasets" / "TinyStoriesV2-GPT4-train.txt",
]

# defaults (overridable from the command line — see parse_args)
PROMPT_WORDS = 20        # feed the model this many words
MAX_NEW_TOKENS = 50      # generate this many next tokens
TEMPERATURE = 0.8        # <1.0 = more focused; set to 0 for greedy
TOP_K = 40               # sample from top-k tokens (None = full vocab)
SEED = None              # set an int for reproducibility, None = random each run


def parse_args():
    p = argparse.ArgumentParser(description="Inference / sanity check for the GPT-MoE checkpoint.")
    p.add_argument("--words", type=int, default=PROMPT_WORDS,
                   help=f"words of the story to feed as prompt (default {PROMPT_WORDS}); ignored if --prompt is given")
    p.add_argument("--new-tokens", type=int, default=MAX_NEW_TOKENS,
                   help=f"number of tokens to generate (default {MAX_NEW_TOKENS})")
    p.add_argument("--temp", type=float, default=TEMPERATURE,
                   help=f"sampling temperature, 0 = greedy (default {TEMPERATURE})")
    p.add_argument("--top-k", type=int, default=TOP_K,
                   help=f"top-k sampling; 0 = full vocab (default {TOP_K})")
    p.add_argument("--seed", type=int, default=SEED,
                   help="random seed for reproducibility (default: random each run)")
    p.add_argument("--prompt", type=str, default=None,
                   help="use YOUR text as the prompt instead of a random dataset story "
                        "(no ground-truth comparison in this mode)")
    p.add_argument("--ckpt", type=str, default=str(CKPT_PATH),
                   help="path to the .pt checkpoint")
    return p.parse_args()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class TinyMoeConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    batch_size: int = 64
    n_experts: int = 4
    top_k: int = 2
    aux_weight: float = 0.01


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def build_RoPE_cache(seq_len, head_dim, base=10000, device=device):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum('i , j -> i j', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[None, None, :, :].to(dtype=x.dtype)
    sin = sin[None, None, :, :].to(dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Attention / MoE / Block / model  (mirrors model.py)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0
        self.pre_norm = nn.RMSNorm(config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        cos, sin = build_RoPE_cache(config.block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x = self.pre_norm(x)
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = SwiGLUFFN(config.n_embd, dropout=config.dropout)

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = nn.RMSNorm(config.n_embd)
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.shared_expert = Expert(config)   # present in state_dict; unused in fwd (matches model.py)

    def forward(self, x):
        x = self.pre_norm(x)
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_gate = F.softmax(topk_val, dim=-1)
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (topk_idx == e)
            if mask.any():
                token_idx, slot = mask.nonzero(as_tuple=True)
                weights = topk_gate[token_idx, slot].unsqueeze(-1)
                out[token_idx] += weights * self.experts[e](x_flat[token_idx])
        P = probs.mean(dim=0)
        one_hot = F.one_hot(topk_idx, self.n_experts).sum(dim=1)
        f = one_hot.float().mean(dim=0)
        aux_loss = self.n_experts * torch.sum(f * P)
        return out.view(B, T, C), aux_loss, f.detach()


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.moe = MoE(config)

    def forward(self, x):
        x = x + self.attn(x)
        moe_out, aux, f = self.moe(x)
        x = x + moe_out
        return x, aux, f


class GPTMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        x = self.drop(self.tok_emb(idx))
        total_aux = 0.0
        route_sum = 0.0
        for block in self.blocks:
            x, aux, f = block(x)
            total_aux = total_aux + aux
            route_sum = route_sum + f
        route_frac = route_sum / len(self.blocks)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux, route_frac

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]                       # last position
            if temperature <= 0.0:                          # greedy
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def pick_random_story(min_words=PROMPT_WORDS + MAX_NEW_TOKENS):
    path = next((p for p in STORY_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No story file found in {STORY_CANDIDATES}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    long_enough = [s for s in stories if len(s.split()) >= min_words]
    pool = long_enough or stories
    return random.choice(pool), path.name


def main():
    args = parse_args()
    top_k = None if args.top_k in (0, None) else args.top_k

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"device: {device}")
    ckpt_path = Path(args.ckpt)
    print(f"checkpoint: {ckpt_path.name}\n")

    # ---- load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", TinyMoeConfig())
    print(f"trained step {ckpt.get('step')} | val_loss {ckpt.get('val_loss'):.4f}")

    model = GPTMoE(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    # ---- build prompt: custom text, or the first N words of a random story ----
    if args.prompt is not None:
        prompt_text = args.prompt
        truth_text = None
        src = "(custom prompt)"
    else:
        story, src = pick_random_story(min_words=args.words + args.new_tokens)
        words = story.split()
        prompt_text = " ".join(words[:args.words])
        truth_text = " ".join(words[args.words:args.words + args.new_tokens])

    prompt_ids = enc.encode(prompt_text)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ---- generate ----
    out = model.generate(x, args.new_tokens, temperature=args.temp, top_k=top_k)
    gen_ids = out[0, len(prompt_ids):].tolist()
    gen_text = enc.decode(gen_ids)

    # ---- report ----
    bar = "=" * 80
    print(f"\n{bar}\nsource: {src}\n{bar}")
    print(f"\n📥 PROMPT:\n{prompt_text}")
    print(f"\n🤖 MODEL CONTINUATION (~{args.new_tokens} tokens, temp={args.temp}, top_k={top_k}):\n{gen_text}")
    if truth_text is not None:
        print(f"\n📖 REAL CONTINUATION (ground truth from dataset):\n{truth_text}")
    print(f"\n{bar}")


if __name__ == "__main__":
    main()
