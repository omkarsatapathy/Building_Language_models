# MoE Training Slowdown — Root-Cause Analysis & Fixes

**Model:** `experiments/code/test_MoE/model.py` (GPT-MoE, ~39.1M params, 4 experts, top-2)
**Hardware:** H200
**Run command:**

```bash
WARMUP_STEPS=150 BATCH_SIZE=320 BLOCK_SIZE=1024 GRAD_ACCUM_STEPS=1 MAX_STEPS=6653 NUM_WORKERS=8 ./run.sh full
```

**Log source:** `~/Documents/train_micro_log.csv` (771 micro rows, one micro per step since `GRAD_ACCUM_STEPS=1`)

---

## TL;DR

All three reported symptoms share **one root cause**: the Mixture-of-Experts router
sends a **different number of tokens to each expert on every step**, which produces
**dynamically-shaped matmuls**. Combined with `torch.compile(model, mode="max-autotune")`,
this forces PyTorch to **re-autotune Triton kernels and re-record CUDA graphs for each
new shape**, forever. The CUDA-graph pool grows without bound, memory fragments, and the
allocator eventually stalls for minutes.

| Symptom (user question) | Real cause |
|---|---|
| Q1 — `torch.compile` takes very long | `max-autotune` benchmarks 20–28 kernel candidates per matmul on first sight of each new shape; MoE keeps producing new shapes, so it never stops paying that cost |
| Q3 — GPU idle 5–7 min, throughput drops to ~10K tok/s | Those "idle" steps **are** recompiles + CUDA-graph re-recording, not data starvation |
| Q2 — too many `.pt` files | Unrelated config: checkpoint saving is welded to the eval cadence (`eval_every=250`) |

**Primary fix:** stop compiling one kernel per shape.

```python
# model.py — replace: model = torch.compile(model, mode="max-autotune")
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
model = torch.compile(model, dynamic=True)
```

---

## Evidence

### 1. The smoking-gun warning

```
CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size.
Recording too many CUDAGraphs may lead to extra overhead. We have observed 51 distinct sizes.
```

51 distinct sizes — and climbing. Each distinct size = one full re-autotune + one new CUDA graph.

### 2. Autotune blocks appear *mid-training*, not just at startup

Fresh `AUTOTUNE` headers show up at step 250+ with oddly specific dimensions:

```
AUTOTUNE mm(144789x320, 320x864)     <- 144789 = tokens routed to some expert this step
AUTOTUNE mm(161760x864, 864x320)     <- 161760 = tokens routed to another expert
AUTOTUNE mm(327680x320, 320x320)     <- 327680 = 320 * 1024 = full batch (attention path, static)
```

`327680 = batch_size(320) × block_size(1024)` is the fixed full-batch size (attention, LM head).
`144789`, `161760`, etc. are **per-expert token counts** — they change every step because routing changes every step.

### 3. The slowdown grows monotonically (the signature of a leak/cascade)

Per-step wall time from the CSV (`micro == 0` rows):

| step range | slow steps (>3s) | avg step time | worst step |
|---|---|---|---|
| 0–100   | 1 (warmup)  | ~0.65s after warmup | 122.4s (step 0, expected first-compile) |
| 100–200 | 1  | 0.85s | 5.0s |
| 200–300 | 4  | 1.09s | 7.2s |
| 300–400 | 8  | 1.32s | 14.1s |
| 400–500 | 24 | 2.58s | 16.7s |
| 500–600 | 32 | 3.63s | 29.0s |
| 600–700 | 45 | 6.48s | 29.4s |
| 700–772 | 40 | **15.86s** | **216.3s** |

Worst 10 steps:

```
step  759   216312 ms     1515 tok/s   <- the "5-7 minute" stall
step    0   122404 ms     2677 tok/s   <- initial compile (expected, one-time)
step  756    64088 ms     5113 tok/s
step  754    62069 ms     5279 tok/s
step  764    59677 ms     5491 tok/s
step  747    49710 ms     6592 tok/s
step  721    46829 ms     6997 tok/s
step  719    46576 ms     7035 tok/s
step  752    35001 ms     9362 tok/s
step  740    32443 ms    10100 tok/s
```

Healthy steps run ~500–680 ms (~500–680K tok/s). The slow steps are 5×–300× slower and become
**more frequent and more severe over time** — exactly what an ever-growing CUDA-graph pool +
allocator fragmentation looks like. A data-starvation problem would be flat/random, not
monotonically worsening.

### 4. Why the dataloader is *not* the culprit

- `block_size` is fixed at 1024 and `drop_last=True`, so every batch fed to the model is
  identically shaped `[320, 1024]`. The variable shapes are created **inside** the model by
  MoE routing, not by the loader.
- The slow steps correlate with new expert-token-count shapes, not with I/O timing.
- `NUM_WORKERS=8` is adequate for this throughput.

---

## Where the dynamic shapes come from

In `MoE.forward` (`model.py`):

```python
for e in range(self.n_experts):
    mask = (topk_idx == e)
    if mask.any():
        token_idx, slot = mask.nonzero(as_tuple=True)
        weights = topk_gate[token_idx, slot].unsqueeze(-1)
        out[token_idx] += weights * self.experts[e](x_flat[token_idx])
        #                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                            x_flat[token_idx] has a DIFFERENT row count every step
```

`token_idx` length = number of tokens routed to expert `e` this step. Routing is data-dependent,
so this count fluctuates (e.g. 144789 one step, 161760 the next). Each distinct count is a new
matmul shape → a new compiled program under `max-autotune`.

Total dispatch slots per step: `N × top_k = (320×1024) × 2 = 655,360`, spread over 4 experts
(~163,840 average each) but **never exactly balanced**, so the per-expert counts jitter every step.

---

## Fixes

### Q1 + Q3 — kill the recompilation cascade (same root cause)

#### Option A — Quick fix (recommended first step)

Compile shape-generic kernels instead of one-per-size. In `model.py`, replace:

```python
model = torch.compile(model, mode="max-autotune")
```

with:

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
model = torch.compile(model, dynamic=True)
```

- `dynamic=True` marks the token dimension as dynamic, so Inductor compiles **one** kernel that
  handles all row counts instead of recompiling per shape.
- Dropping `max-autotune` also slashes the initial compile time. For a 39M model the matmuls are
  tiny; the autotuned kernels are barely faster than the defaults, so you lose almost nothing.

Expected result: no more mid-training `AUTOTUNE` blocks, no 5–216s stalls, steady ~500K+ tok/s.

#### Option B — Proper fix (best long-term): fixed expert capacity

Give every expert a **fixed capacity** so its input is always the same shape. Tokens beyond
capacity are dropped (standard Switch-Transformer / GShard behavior). With static shapes,
`max-autotune` + CUDA graphs become safe and optimal.

```
capacity = ceil(capacity_factor * top_k * N / n_experts)   # e.g. capacity_factor = 1.25
```

This is a ~20-line rewrite of `MoE.forward` (build a `[n_experts, capacity, C]` dispatch tensor,
run all experts as one batched matmul, scatter back). It removes data-dependent shapes entirely.
*(Not yet applied — request this if you want the full rewrite.)*

#### Option C — If you must keep `max-autotune`

At minimum stop the CUDA-graph re-recording. Add near the other backend flags at the top of
`model.py`:

```python
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
```

This is weaker than Option A (it still re-autotunes per shape). Prefer A.

---

### Q2 — checkpoint only every 15% of the run

Currently saving is tied to eval in the training loop:

```python
if step > 0 and step % eval_every == 0:      # eval_every = 250
    val_loss = evaluate(...)
    save_checkpoint(...)                       # -> ~26 files over 6653 steps
```

Separate the two cadences. Near the other config:

```python
# save a checkpoint every 15% of the run (~6-7 files total)
ckpt_every = max(1, int(0.15 * max_steps))     # 6653 -> ~998 steps
```

Then in the loop:

```python
if step > 0 and step % eval_every == 0:
    val_loss = evaluate(model, val_iter, eval_steps, config)
    print(f"  >>> eval @ step {step}: val_loss {val_loss:.4f}")
    if step % ckpt_every == 0 or step == max_steps - 1:
        save_checkpoint(model, optimizer, step, val_loss)
```

Keep `eval_every` a divisor of `ckpt_every` (250 into ~1000) so every save has a fresh
`val_loss`. If you'd rather keep only the single **best** checkpoint, track `best_val` and
overwrite one file instead of saving per interval.

---

## Secondary observation (not a cause of the stalls)

The auxiliary load-balancing loss `aux` sits at ~16.0 for the entire run. Since
`aux = n_experts * sum(f * P) = 4 * sum(f * P)`, a value pinned near 16 means `sum(f·P) ≈ 4`.
Worth watching the `expert{0..3}_frac` columns once throughput is fixed — if one expert
dominates or routing collapses to uniform-by-construction, the MoE isn't specializing.
This does **not** affect step timing; address it after the compile fix.

---

## Recommended order of action

1. **Apply Option A** (`dynamic=True` + `cache_size_limit`). Re-run; confirm the mid-training
   `AUTOTUNE` blocks and multi-second stalls are gone.
2. **Apply the Q2 checkpoint change** to stop flooding `checkpoints/` with `.pt` files.
3. If you want peak throughput and clean CUDA-graph behavior, do **Option B** (fixed capacity)
   and restore `mode="max-autotune"`.
4. Once fast, investigate the flat `aux` / routing-fraction behavior.
