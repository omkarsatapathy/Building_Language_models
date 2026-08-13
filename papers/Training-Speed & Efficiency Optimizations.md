# Training-Speed & Efficiency Optimizations in Karpathy's "Let's Reproduce GPT-2 (124M)"
## A Technical Reference for Lecture Use

## TL;DR
- Karpathy's lecture drives a single-step training time on one NVIDIA A100 from ~1,000 ms down to ~90 ms — an ~11× speedup — through a stacked sequence of pure-efficiency changes (TF32 → BF16 autocast → torch.compile → FlashAttention → vocab padding → fused AdamW), none of which change the model's mathematical definition.
- On 8× A100 with DDP, gradient accumulation to a 524,288-token batch, AdamW (β=0.9/0.95, ε=1e-8, wd=0.1 on 2D params only), gradient clipping at 1.0, and a cosine LR schedule (max 6e-4, min 6e-5, 715 warmup steps, 19,073 steps ≈ 1 epoch over 10B FineWeb-EDU tokens), the reproduced 124M model surpasses OpenAI's GPT-2 124M on both FineWeb val loss and HellaSwag (29.9% vs 29.4%), approaching GPT-3 Small's 33.7% — using only 10B tokens vs GPT-2/GPT-3's ~100B/300B.
- The whole run costs ~90 minutes and ~$20 on an 8×A100 80GB node (Lambda, ~$14/hr); `llm.c`, Karpathy's raw C/CUDA equivalent, does the same reproduction and runs ~7% faster than optimized PyTorch nightly (with vocab padded to 50304; ~11% faster without).

## Key Findings
- The optimizations divide into two categories: (1) **pure speed** changes that leave the loss curve mathematically unchanged (precision, compilation, kernel choice, tensor shapes), and (2) **algorithmic/scaling** changes (optimizer hyperparameters, batch size, DDP) that change what is computed but follow the GPT-3 recipe.
- The single most cost-effective change is TF32 (`torch.set_float32_matmul_precision('high')`): roughly 3× (~1000 ms → ~333 ms) for one line of code, with negligible accuracy impact.
- BF16 is preferred over FP16 specifically because it keeps FP32's 8-bit exponent (dynamic range), eliminating the need for a gradient scaler.
- The counter-intuitive "make vocab bigger to go faster" (50,257 → 50,304) yields a large speedup because 50,304 = 128×393 aligns matmul tiles to power-of-2 boundaries.
- FlashAttention is *more* FLOPs but faster, because it is memory-movement-bound, not compute-bound; torch.compile cannot discover it because it requires an algorithmic rewrite (online softmax) that a compiler will not invent.

## Details

### 1. Executive Summary: The Speedup Timeline
All per-step timings below are from the lecture, measured on a single NVIDIA A100 (the lecture used an 80GB A100 via Lambda), training the 124M model at batch B=16, sequence length T=1024 (the micro-batch used during the optimization section). The chapter markers and reported timings are:

| # | Optimization | ms/step | Approx. tok/s | Cumulative speedup | Notes |
|---|---|---|---|---|---|
| 0 | FP32 baseline, GPU | ~1000 ms | ~16k | 1× | naive PyTorch, everything FP32 |
| 1 | TF32 (`set_float32_matmul_precision('high')`) | ~333 ms | ~49k | ~3× | tensor cores, FP32 I/O |
| 2 | BF16 via `torch.autocast` | ~300 ms | ~54k | ~3.3× | mixed precision, no grad scaler |
| 3 | `torch.compile` | ~130 ms | — | ~7.7× | kernel fusion, less Python overhead |
| 4 | FlashAttention (`F.scaled_dot_product_attention`) | ~96 ms | — | ~10.4× | fused attention kernel |
| 5 | "Nice numbers": vocab 50257→50304 | ~93 ms | — | ~10.7× | power-of-2 tile alignment |
| 6 | Hyperparameters + Fused AdamW | ~90 ms | — | ~11× | fused optimizer kernel |

(These chapter timings — 1000ms / 333ms / 300ms / 130ms / 96ms / 93ms / 90ms — are the exact figures Karpathy uses as his lecture section titles.) After step 6, further wall-clock gains come from scaling: gradient accumulation to a 524,288-token batch and DDP across 8 A100s, which multiplies throughput ~8×.

The TF32→BF16 step gives a smaller marginal gain than expected because after TF32 the workload becomes increasingly memory-bandwidth-bound rather than compute-bound — the matmuls are already fast, so halving precision again mostly helps data movement, not arithmetic.

### 2. Floating-Point Data Types: FP32, TF32, FP16, BF16
Bit layouts (sign / exponent / mantissa):
- **FP32 (IEEE single):** 1 / 8 / 23. Full range and precision. A100 non-tensor FP32 = 19.5 TFLOPS.
- **TF32 (TensorFloat-32):** 1 / 8 / 10 (a 19-bit internal representation). Same 8-bit exponent as FP32, mantissa truncated to 10 bits like FP16. Inputs/outputs are still FP32 in memory; the A100 tensor core internally rounds operands to TF32 for the multiply. A100 TF32 tensor core = 156 TFLOPS (312 with sparsity).
- **FP16 (IEEE half):** 1 / 5 / 10. More mantissa precision than BF16 but a much smaller exponent (max ~65,504), which causes gradient underflow/overflow — hence gradient scaling. A100 FP16 tensor core = 312 TFLOPS (624 with sparsity).
- **BF16 (bfloat16):** 1 / 8 / 7. Same dynamic range as FP32 (8-bit exponent), fewer mantissa bits. A100 BF16 tensor core = 312 TFLOPS (624 with sparsity).

**Why BF16 over FP16 for training:** BF16 preserves FP32's exponent range, so gradients (which span many orders of magnitude) don't underflow to zero. This removes the need for `torch.cuda.amp.GradScaler`, which FP16 requires to multiply the loss by a large factor before backward and unscale before the optimizer step. This is why nanoGPT's `train.py` chooses `dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16'`, and only the float16 path "will auto implement a GradScaler."

**Where autocast applies reduced precision:** Under `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`, PyTorch selectively runs matrix-multiply-heavy ops (linear layers, matmuls, convolutions) in BF16, while keeping numerically sensitive ops — reductions, softmax, layernorm, and loss — in FP32. Model weights (the master copy) remain FP32; only activations flowing through eligible ops are cast. In the build-nanogpt code, only the forward pass and loss are wrapped in autocast; `loss.backward()` and `optimizer.step()` are outside it.

### 3. torch.compile
`torch.compile` (PyTorch 2.0+, backed by TorchDynamo + TorchInductor) speeds training two ways:
- **Removing Python interpreter overhead:** Eager PyTorch dispatches each operation from Python one at a time. `torch.compile` traces the whole forward/backward graph once and runs optimized code, so the GPU isn't waiting on Python between kernels.
- **Kernel fusion → less GPU memory traffic:** Without compilation, each op launches its own kernel and round-trips intermediate tensors to HBM. Per PyTorch's own engineering blog, "PyTorch's Inductor compiler automatically groups dependent operations together into single, efficient Triton kernels. This keeps data in faster memory close to the register and cuts down on kernel overhead." The lecture's example: GELU's many element-wise operations, which would otherwise stream activations to and from HBM repeatedly, become one fused kernel. Karpathy frames the key intuition as the GPU spending most of its time moving data between HBM and the on-chip SMs, and fusion eliminating those round trips.
- **Shape specialization / static graph:** With `dynamic=False`, Inductor specializes kernels for the exact tensor shapes seen. This is why "dynamic instructions" don't need to be matched — the model's control flow is static (fixed shapes each step), so a single specialized graph works; a graph break (data-dependent control flow) would force recompilation and hurt performance.

In build-nanogpt's final `train_gpt2.py`, `use_compile` is set to `False` with the comment that "torch.compile interferes with HellaSwag eval and Generation. TODO fix" — i.e., the shape changes during eval trigger graph breaks.

### 4. FlashAttention
FlashAttention (Dao et al., 2022, arXiv:2205.14135; FlashAttention-2 in 2023, arXiv:2307.08691) is an **IO-aware, exact** attention algorithm. It is invoked in the code via `F.scaled_dot_product_attention(q, k, v, is_causal=True)`.

- **Avoids materializing the N×N attention matrix:** Standard attention computes S = QKᵀ (N×N), writes it to HBM, reads it back for softmax, writes P, reads P and V for the output. FlashAttention tiles Q, K, V into blocks that fit in SRAM and computes the output block-by-block, never storing the full N×N matrix in HBM. This reduces attention memory from O(N²) to O(N).
- **Online softmax trick:** Based on Milakov & Gimelshein, "Online normalizer calculation for softmax" (NVIDIA, 2018, arXiv:1805.02867). Normal (safe) softmax needs two passes: one to find the row max (for numerical stability), one to exponentiate and normalize. Online softmax maintains a running max and running sum of exponentials as it streams blocks, computing exact softmax in a single pass with a cross-block "fix-up" rescale. That paper reports softmax speedups up to 1.3× and softmax+top-k fused up to 5×.
- **More FLOPs, but faster:** FlashAttention actually performs *more* arithmetic (it recomputes parts of attention rather than caching them), yet runs faster because attention is **memory-bound**: the bottleneck is HBM reads/writes, not FLOPs. On an A100, HBM bandwidth is ~1.5–2 TB/s while on-chip SRAM is ~19 TB/s, so eliminating HBM round-trips dominates.
- **Why torch.compile can't find it:** torch.compile fuses adjacent operations but will not invent a new algorithm. FlashAttention requires an algorithmic restructuring (blocked computation + online softmax) that a compiler's operation-fusion pass cannot derive; it must be hand-written as a dedicated kernel.

### 5. "Nice Numbers": Vocab Padding 50257 → 50304
GPT-2's tokenizer vocab is 50,257 (50,000 BPE merges + 256 byte tokens + 1 `<|endoftext|>`), an ugly number that is not divisible by high powers of 2. CUDA matmul kernels process data in block tiles that are powers of 2; when a dimension isn't tile-aligned, the kernel falls back to a slower path with a "remainder" phase handling the leftover columns.

Padding vocab to **50,304 = 2⁷ × 393 = 128 × 393** (divisible by 2, 4, 8, 16, 32, 64, 128) makes the output projection and embedding matmuls tile-aligned. In the lecture this dropped the step from ~96 ms to ~93 ms; Karpathy separately reported it as "The most dramatic optimization to nanoGPT so far (~25% speedup) is to simply increase vocab size from 50257 to 50304 (nearest multiple of 64)... This calculates added useless dimensions but goes down a different kernel path with much higher occupancy" (Twitter/X). The extra logits are wasted computation, but the model quickly learns to drive their probabilities toward zero (they never appear as targets), so quality is unaffected. Implemented simply as `GPT(GPTConfig(vocab_size=50304))`. (Note: in the PyTorch/nanoGPT setting, padding must account for the wte↔lm_head weight tying — the padded rows should never be sampled — a subtlety Karpathy flags in the llm.c writeup.)

### 6. Optimizer & Hyperparameter Setup
The lecture borrows hyperparameters from the GPT-3 paper (Brown et al., 2020, "Language Models are Few-Shot Learners", arXiv:2005.14165, Appendix B), since the GPT-2 paper does not specify them. GPT-3 Appendix B verbatim: "we use Adam with β1=0.9, β2=0.95, and ε=10⁻⁸, we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value ... There is a linear LR warmup over the first 375 million tokens ... All models use weight decay of 0.1 to provide a small amount of regularization."

- **AdamW:** betas = (0.9, 0.95), eps = 1e-8 (from the above).
- **Gradient clipping:** global norm clipped to 1.0 via `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. Karpathy monitors this norm to detect training instability (spikes early in training).
- **Cosine LR schedule with linear warmup:** max_lr = 6e-4, min_lr = 6e-5 (10% of max), warmup_steps = 715, max_steps = 19,073. The 715-step warmup ≈ 375M tokens ÷ 524,288 tokens/step.
- **Weight decay 0.1, applied selectively:** In `configure_optimizers`, parameters are split into two groups — tensors with dim ≥ 2 (all matmul weights + embeddings) get weight_decay=0.1; tensors with dim < 2 (biases, LayerNorm scales/gains) get 0.0. Code comment: "Any parameters that is 2D will be weight decayed, otherwise no. i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't."
- **Fused AdamW:** the code detects whether the installed PyTorch AdamW supports `fused=True` (via `inspect.signature(torch.optim.AdamW).parameters`) and enables it on CUDA. The final call is `torch.optim.AdamW(optim_groups, lr=..., betas=(0.9, 0.95), eps=1e-8, fused=use_fused)`. The fused kernel performs the optimizer update for all parameters in a single fused CUDA kernel rather than looping, giving another small speedup (contributing to the ~90 ms final time).

### 7. Gradient Accumulation
To simulate GPT-3's ~0.5M-token batch on limited memory, the code sets `total_batch_size = 524288` (2¹⁹ tokens, "~0.5M, in number of tokens") and computes `grad_accum_steps = total_batch_size // (B * T * ddp_world_size)`. It runs that many micro-batch forward/backward passes, accumulating gradients, before one optimizer step.

**The mean-reduction normalization bug:** `F.cross_entropy` uses `reduction='mean'` by default, so each micro-batch's loss is already averaged over its tokens. When you call `loss.backward()` repeatedly, gradients **add**. Summing N micro-batch mean-losses gives a total that is N× too large relative to the true mean over the whole big batch — it corresponds to a SUM objective, not a MEAN. The fix is to divide each micro-batch loss by `grad_accum_steps` before backward. The code comment states it directly: "we have to scale the loss to account for gradient accumulation, because the gradients just add on each successive backward(). addition of gradients corresponds to a SUM in the objective, but instead of a SUM we want MEAN. Scale the loss here so it comes out right":
```python
loss = loss / grad_accum_steps
loss_accum += loss.detach()
loss.backward()
```
This restores the correct normalization so that gradient accumulation is mathematically equivalent to a single large batch.

### 8. Distributed Data Parallel (DDP)
Launched via `torchrun --standalone --nproc_per_node=8 train_gpt2.py`, which spawns 8 processes and sets environment variables `RANK`, `LOCAL_RANK`, `WORLD_SIZE`. The code reads these to assign each process a GPU (`cuda:{ddp_local_rank}`), designates rank 0 as `master_process` for logging/checkpointing, and initializes the NCCL backend via `init_process_group(backend='nccl')`.

- **Gradient synchronization / all-reduce:** DDP wraps the model (`DDP(model, device_ids=[ddp_local_rank])`). On each `backward()`, DDP averages gradients across all ranks via an all-reduce so every replica takes an identical optimizer step. `raw_model = model.module if ddp else model` keeps a handle to the unwrapped model.
- **no_sync vs manual toggling:** During gradient accumulation you do NOT want an all-reduce on every micro-step (only the last). PyTorch's idiomatic tool is the `no_sync()` context manager; Karpathy instead directly sets the internal flag `model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)` so the expensive all-reduce fires only on the final micro-step. An erratum notes this flag is used by both forward and backward passes, so the line was moved up accordingly.
- **Loss averaging across ranks:** For reporting, `dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)` averages the logged loss across GPUs. The same is done for validation loss (`ReduceOp.AVG`) and for HellaSwag counts (`ReduceOp.SUM`).
- The DataLoaderLite is made rank-aware: each process starts at offset `B*T*process_rank` and strides by `B*T*num_processes`, so the 8 GPUs read disjoint slices.

### 9. Hardware & Memory Hierarchy (A100)
- **Architecture:** NVIDIA A100 (Ampere, TSMC 7nm), 3rd-gen tensor cores. Peak throughput: FP32 19.5 TFLOPS; TF32 156 TFLOPS (312 sparse); BF16/FP16 312 TFLOPS (624 sparse); INT8 624 TOPS (1248 sparse). FP64 tensor core 19.5 TFLOPS.
- **Memory hierarchy:** HBM2/HBM2e global memory (40GB @ 1,555 GB/s or 80GB @ ~2,039 GB/s) is large but relatively slow; on-chip SRAM (~20MB across the chip, split into per-SM L1/shared memory + 40MB L2) is tiny but ~19 TB/s. The FlashAttention paper uses exactly this A100 example (40GB HBM @ ~1.5 TB/s vs ~20MB SRAM @ ~19 TB/s).
- **Memory-bound vs compute-bound:** A kernel is compute-bound if its arithmetic dominates, memory-bound if data movement dominates. Element-wise ops (GELU, softmax, LayerNorm) and attention are memory-bound; large matmuls are compute-bound. The lecture's optimizations mostly attack memory movement — which is why fusion (torch.compile), FlashAttention, and tile-alignment all help even though FLOPs don't drop.
- **Sparsity caveat:** the 2× "sparse" figures require a 2:4 structured-sparsity fine-tune that these dense runs do not use, so the relevant peaks are the dense numbers (e.g., 312 TFLOPS BF16).

### 10. Dataset Preparation: FineWeb-EDU
- Per Penedo et al., 2024 (arXiv:2406.17557): "FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots ... we introduce FineWeb-Edu, a 1.3-trillion token collection of educational text filtered from FineWeb." FineWeb-EDU's educational-quality classifier was trained on Llama-3-70B-Instruct annotations (retaining data scored ≥ 3), which the paper shows "dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and ARC."
- The lecture uses the **sample-10BT** subset (~10B GPT-2 tokens; sample-10BT ⊂ sample-100BT ⊂ sample-350BT). Karpathy contrasts this with GPT-2's private WebText and GPT-3's CommonCrawl mixture, arguing modern filtered data can match older results with far fewer tokens.
- **Sharding & tokenization:** `fineweb.py` tokenizes documents with tiktoken's GPT-2 BPE (`enc = tiktoken.get_encoding("gpt2")`), prepends the `<|endoftext|>` delimiter, and writes shards of 100M tokens each as `.npy` files (uint16), one val shard and the rest train, into `edu_fineweb10B/`. `load_tokens` reads a shard, converts uint16→int32→torch long (the `astype(np.int32)` line was added after the video to fix an older-PyTorch uint16→long conversion issue).

### 11. Evaluation Methodology
- **Validation loss:** every 250 steps, the model runs in eval mode over 20 val micro-batches with autocast BF16, accumulating and (under DDP) all-reducing the mean loss. This tracks the FineWeb-EDU held-out loss.
- **HellaSwag** (Zellers et al., 2019, ACL, arXiv:1905.07830): ~70,000 adversarially-filtered 4-way sentence-completion questions (the validation split used here is 10,042 examples). The GPT-3 paper (§3.1.3) describes HellaSwag as "a dataset ... where the examples were adversarially mined to be difficult for language models while remaining easy for humans (who achieve 95.6% accuracy)." Because a 124M model is too weak for a multiple-choice format, the eval uses a **completion/loss-based** protocol: each of the 4 candidate endings is appended to the shared context to form 4 sequences; the model scores each by average autoregressive cross-entropy loss over only the completion tokens (masked), and the lowest-loss ending is the prediction (`get_most_likely_row`). This "acc_norm"-style scoring gives a smooth, well-behaved metric even for small models. Examples are sharded across ranks by `i % ddp_world_size == ddp_rank`.

### 12. Final Results vs GPT-2 / GPT-3 124M
Training the 12-layer, 12-head, 768-dim, 124M model on 10B FineWeb-EDU tokens (19,073 steps at 524,288 tokens/step ≈ 1 epoch):
- The reproduced model **beats the OpenAI GPT-2 124M checkpoint on FineWeb val loss** and on HellaSwag. Karpathy's writeup: "One more point of reference is that GPT-3 in Appendix H cites HellaSwag accuracy at 33.7 for GPT-3 Small (124M) model. We get to 29.9 here, which surpasses GPT-2 (124M) at 29.4." (The build-nanogpt `hellaswag.py` logs the OpenAI GPT-2 124M completion-style baseline as acc_norm ≈ 0.2955; third-party reproductions following the repo report ~0.305 at 10B tokens.) Training ~40B tokens pushes accuracy meaningfully higher.
- Cost/time: **~90 minutes, ~$20** on an 8×A100 80GB node — Karpathy: "reproducing this model on one 8X A100 80GB SXM node takes ~90 minutes ... on Lambda this node goes for ~$14/hr, so the total cost ... is about $20." The build-nanogpt README states it more loosely as "a matter of ~1hr and ~$10."

**Caveats Karpathy raises:**
- **Not an apples-to-apples comparison:** GPT-2 trained on the never-released WebText and GPT-3 on a CommonCrawl mixture; both differ in distribution from FineWeb-EDU. In his words: "This is not the ideal metric because the data distribution of GPT-2 was different ... HellaSwag has no math/code so it slightly favors our setting (common crawl-like data)."
- **Fewer tokens via higher-quality modern data:** matching/beating GPT-2 with only 10B tokens (vs ~100B for GPT-2, 300B for GPT-3) is partly attributable to better-filtered modern data, not purely modeling: "here we trained for 10B tokens, while GPT-3 models were all trained for 300B tokens."
- **Periodicity / non-shuffled data:** DataLoaderLite streams shards in the same fixed order every epoch and does not shuffle/permute documents across epochs. Karpathy flags in the lecture that this introduces periodicity/artifacts into the loss curve and that documents ideally should be permuted each epoch; GPT-3 explicitly "sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting."
- **Loss-curve spikes:** occasional gradient-norm spikes early in training are why gradient clipping is monitored.

### 13. llm.c — Raw C/CUDA Equivalent
`karpathy/llm.c` is a from-scratch LLM training stack in C/CUDA (single-file mainline `train_gpt2.cu` (~4,000 lines for the pretraining path), plus a ~1,000-line CPU reference `train_gpt2.c` and a PyTorch reference `train_gpt2.py` that Karpathy calls "a slightly tweaked nanoGPT"). It implements the same GPT-2 training directly in CUDA kernels, avoiding Python and the PyTorch runtime entirely.
- **Speed:** Karpathy's "State of the Union [May 3, 2024]" reports: "llm.c: at ~167K tok/s ... PyTorch code as is on master runs at ~150K tok/s (i.e. we are 167/150 ~= 11% faster). If you manually pad the vocab size to 50304 ... reducing llm.c speed improvement to ~7%." He also noted it was ~46% faster than the then-current PyTorch stable 2.3.0. An earlier fp32/no-flash comparison showed "78ms/iter for llm.c and 80ms/iter for PyTorch" on an A100.
- It reproduces GPT-2 124M in ~90 min for ~$20 on an 8×A100, reaching HellaSwag 29.9% (FineWeb val-loss target ~3.28), the same result referenced above. It is the basis for the modded-nanogpt speedrun, whose target is "3.28 cross-entropy loss on the FineWeb validation set ... [which] follows Andrej Karpathy's GPT-2 replication in llm.c."

## Recommendations
1. **Teach the optimizations in the lecture's order**, framing each as either "free speed" (precision/compile/kernel/shape) or "scaling recipe" (optimizer/batch/DDP). Emphasize that steps 1–5 leave the loss curve mathematically identical — this is the key conceptual takeaway.
2. **Lead with the highest-leverage, lowest-effort change:** TF32 (one line, ~3×). Then BF16 autocast, then torch.compile. Benchmark each on your own hardware — gains vary by GPU generation (Ampere+ needed for TF32/BF16 tensor cores; pre-Ampere sees little).
3. **Demonstrate the gradient-accumulation normalization bug live** by printing the loss with and without the `/grad_accum_steps` division — it is the most common subtle error students reproduce.
4. **For a classroom reproduction on a budget:** use a single A100/H100 or even a consumer 4090; expect proportionally longer runtime (single-GPU ~4–24h vs ~90min on 8×A100). Target FineWeb val loss ≤ 3.29 / HellaSwag ≥ ~0.30 as the "matched GPT-2" checkpoint.
5. **Thresholds that would change the recommendation:** if training in FP16 (older GPU without BF16), you MUST re-enable a GradScaler; if you see graph breaks with torch.compile, isolate eval/generation from the compiled region; if the loss curve shows periodic bumps, add per-epoch document shuffling before drawing scaling conclusions.

## Caveats
- All ms/step figures are as reported in the lecture on a specific A100 and PyTorch version; exact numbers will differ across hardware, PyTorch releases (which have themselves absorbed many of these speedups), and B/T choices. Treat the multipliers as the durable lesson, not the absolute milliseconds.
- The build-nanogpt README itself does not contain the specific final val-loss / HellaSwag numbers, the 19,073-step figure, or the periodicity caveat — these come from the lecture, the code (`train_gpt2.py`), and the companion `llm.c` discussion writeup (Discussion #481). Where a number originates outside a primary Karpathy source (e.g., third-party reproductions), it is flagged as such.
- A100 sparse TFLOPS figures require structured-sparsity fine-tuning not used here; the dense numbers are the relevant ones.
- HellaSwag is now saturated by frontier models (retired from the HF Open LLM Leaderboard v2 in 2024); it remains useful here only as a smooth low-end capability signal for 124M-scale models.
- The two Karpathy repos differ: `build-nanogpt` (this lecture, FineWeb-EDU, ~1hr run) is the ad-hoc teaching repo; the older `nanoGPT` reproduces GPT-2 124M on OpenWebText over ~4 days on 8×A100 and reaches val loss ~2.85 after fine-tuning-style domain adaptation. Numbers should not be mixed between them.