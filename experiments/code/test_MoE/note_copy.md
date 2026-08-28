# Notes — test_MoE

## Implementation status (as of current model.py)

Built and aligned:
- **RoPE** — `build_RoPE_cache`, `rotate_half`, `apply_rotary_pos_emb`. Llama-style
  (rotate_half / two-halves) convention.
- **CausalSelfAttention** — pre-norm (RMSNorm) inside the module, fused qkv projection,
  QK-norm (RMSNorm over head_dim) before RoPE, `F.scaled_dot_product_attention(is_causal=True)`,
  residual dropout.
- **MoE** — RMSNorm pre-norm inside, `gate` router, top-k routing with softmax over the
  chosen k, expert-parallel loop, scatter-add combine.
- **Block** — `x = x + attn(x)` then `x = x + moe(x)` (clean residuals; each sub-module owns
  its pre-norm, so the Block must NOT normalize again).
- **SwiGLUFFN** — in swiglu.py, standard SwiGLU (verified against paper, see below).

## Pending / TODO

- [ ] **Wire SwiGLU into Expert.** Expert still uses `nn.Sequential(Linear, GELU, Linear)`.
      Replace with `SwiGLUFFN(config.n_embd, dropout=config.dropout)` (from swiglu import SwiGLUFFN).
- [ ] **Load-balancing auxiliary loss for MoE.** Without it the router collapses onto a few
      experts (the classic MoE stability failure). MoE forward must RETURN an aux-loss term;
      Block and top-level model must accumulate it up to the training loop. This is the next step.
- [ ] **Config cleanup:** `n_experts` is declared twice in TinyMoeConfig — remove the duplicate.
- [ ] **Verify** the `device=device` default was removed from `build_RoPE_cache` signature
      (device should come from the registered buffer via `.to(device)`).
- [ ] Top-level model (token embedding, stack of Blocks, final norm, LM head, weight tying)
      not built yet.

## RoPE: attention-wiring checklist (all DONE — kept for reference)

1. [x] Mixed-precision safety: build cos/sin in float32, cast to `x.dtype` at apply time.
2. [x] Slice cache to actual T (`cos[:T]`), done in attention forward.
3. [x] Build cache ONCE as a non-persistent buffer (not every forward).
4. [ ] Drop `device=device` default arg in `build_RoPE_cache` — VERIFY.
5. [x] `assert head_dim % 2 == 0`.
6. [x] Apply RoPE to q and k only, not v.

## SwiGLU: faithfulness check vs paper

Paper read: **papers/SwiGLU.pdf** = *"Confidence-Adaptive SwiGLU for Mixture-of-Experts"*
(kappa-SwiGLU, 2026) — NOT the original Shazeer paper, but it states the standard definition.

Standard SwiGLU (paper Eq. 5):  `SwiGLU(x) = SiLU(W_g x) ⊙ (W_u x)`, `SiLU(z) = z·σ(z)`.
Our swiglu.py implements exactly this (`F.silu(gate) * value`), plus the FFN down-projection
`w_out` and the 2/3·4·d hidden-dim scaling. **Faithful — no deviation** (kappa = 1 case).

## Future experiment: kappa-SwiGLU (the paper's actual contribution)

The paper proposes a **confidence-adaptive** gate — an MoE-specific upgrade, directly relevant
to this project:

- Sharpness-adjusted SiLU: `SiLU_κ(z) = z · σ(κ·z)`. Standard SwiGLU is the fixed κ=1 case.
- κ is a **learned function of router confidence** (the router logit `s_e(x) = r_eᵀx`):
  `κ_{e,j}(x) = φ(α_{e,j}·s_e(x) + b_{e,j})`, with bounded map `φ(z) = U^{tanh(z)}` (U>1)
  so κ stays in (1/U, U) and κ=1 when the signal is 0.
- L2 regularization on α and b keeps it near standard SiLU.
- Intuition: larger κ → sharper/more selective gate; smaller κ → smoother/broadly active.
  Each expert learns whether high-confidence tokens should be gated sharper or softer.

Plumbing implication: kappa-SwiGLU couples the expert to the router — the expert forward needs
the router logit for its assigned tokens. So this requires passing routing confidence from the
MoE router into each SwiGLU expert. Run as a labeled A/B experiment AGAINST the standard-SwiGLU
baseline once the baseline + load-balancing loss are training.


while building traing loop Dont forget to add the auxilary loss to the total loss : 

# training step (sum aux over all layers)
total_aux = sum_of_layer_aux_losses
loss = ce_loss + 0.01 * total_aux


## Watch load balancing happen: per-expert token-count logging + CSV

Goal: SEE routing collapse (or balance) live, and keep a per-batch record so after
training I can compute the overall % of tokens each expert handled across the whole run.

### Step 1 — make each MoE layer return its per-expert counts
`one_hot` is already computed for the aux loss, so counts are ~free.

```python
# MoE.forward  (one_hot: [N, n_experts], 1 where a token dispatched to that expert)
counts = one_hot.sum(dim=0)          # [n_experts]  tokens routed to each expert this batch
return out.view(B, T, C), aux_loss, counts
```

```python
# Block.forward — pass counts up alongside aux
def forward(self, x):
    x = x + self.attn(x)
    moe_out, aux, counts = self.moe(x)
    x = x + moe_out
    return x, aux, counts
```

The top-level model collects `counts` from every Block into a list, then
`torch.stack(...)` -> [n_layers, n_experts] and returns it to the training loop.

### Step 2 — live console log (every 50 steps)
`max/min` ratio is the key number: 1.0 = perfectly balanced, big = collapse.

```python
layer_counts   = torch.stack(all_layer_counts)     # [n_layers, n_experts]
tok_per_expert = layer_counts.sum(dim=0).float()   # [n_experts] summed over layers
frac           = tok_per_expert / tok_per_expert.sum()

if step % 50 == 0:
    ideal     = 1.0 / config.n_experts
    imbalance = frac.max().item() / max(frac.min().item(), 1e-9)
    pretty    = "  ".join(f"E{i}:{f*100:4.1f}%" for i, f in enumerate(frac.tolist()))
    print(f"step {step:5d} | {pretty} | ideal {ideal*100:.1f}% | max/min {imbalance:4.1f}x")
```

Expect: NO aux loss -> one/two experts drift to 60-90%, max/min blows past 10x within a
few hundred steps. WITH aux loss (alpha=0.01) -> fracs hover near 1/n_experts, max/min ~1-2x.

### Step 3 — write EVERY batch to CSV (for post-training analysis)
Log RAW COUNTS (not fractions) so the overall % is correctly token-weighted at the end.
`flush()` each row so a crash still leaves usable data.

```python
import csv
# --- before the training loop ---
csv_file   = open("expert_usage_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["step"] + [f"E{i}" for i in range(config.n_experts)] + ["max_min"])

# --- inside the loop, every batch (tok_per_expert from Step 2) ---
frac      = tok_per_expert / tok_per_expert.sum()
imbalance = frac.max().item() / max(frac.min().item(), 1e-9)
csv_writer.writerow([step] + [int(c) for c in tok_per_expert.tolist()] + [round(imbalance, 3)])
csv_file.flush()

# --- after the loop ---
csv_file.close()
```

### Step 4 — after-training analysis (overall distribution across the whole run)
```python
import pandas as pd
df           = pd.read_csv("expert_usage_log.csv")
expert_cols  = [c for c in df.columns if c.startswith("E")]
totals       = df[expert_cols].sum()               # total tokens per expert over ALL batches
overall_pct  = 100 * totals / totals.sum()
print(overall_pct.round(2))                         # e.g. E0 24.9  E1 25.1  E2 25.0  E3 25.0
print("worst-batch max/min:", df["max_min"].max())  # spikes reveal transient collapse
```

Note: counts are DISPATCH SLOTS — a token in top-2 counts for both its experts, which is
exactly what load balancing cares about.
