# FSDP Hands-On Lab — Roadmap

**Format:** instructor-led lab. One topic at a time. You ask for the code, I hand you exactly
that step, we run it, and we do a full **tensor/shape anatomy** before moving on. Nothing is
introduced before you've seen its shape and can reason about it.

**Hardware reality:** single Apple-Silicon **MPS** device. We cannot spawn 8 real GPU workers
(NCCL is CUDA-only). So we **simulate `world_size = N` ranks inside one process** — the
collectives (all-gather, reduce-scatter) become plain tensor ops we implement by hand. The
*math and the shapes are identical* to real FSDP; only the transport is faked. That is exactly
what we want for building the mental model.

**The specimen:** a plain **FFN** (`784 → 1024 → 1024 → 10`, MNIST). We treat the whole FFN as
**one FSDP unit** — one ribbon. Your lecture notes use one ribbon *per transformer block*; a
block is just several of these ribbons side by side, so once the FFN clicks, blocks are free.

**Goal:** reach the end of **one full training step, then one full epoch**, with every quantity
in the notes (FlatParameter, shard, all-gather, discard, reduce-scatter, local step) observed as
a real, printed tensor. Only when a whole epoch runs does the picture close.

**Notation we'll keep consistent** (matches your lecture notes):
- `N` = world size (number of simulated ranks/shards). We'll start with `N = 8`.
- `P` = total parameter count of the FFN.
- `L` = ribbon length (flattened length, after padding). `shard_len = L / N`.
- `rank i` holds `FlatParameter[ i*shard_len : (i+1)*shard_len ]`, permanently.

---

## Topics

### 1. The specimen FFN — build it, count every parameter
Construct the `784→1024→1024→10` FFN as a normal `nn.Module`. **Anatomy:** list every parameter
tensor (`weight`/`bias` per layer), its shape, and its numel. Hand-verify the total `P`. This is
the "6 weight tensors of a block" idea from the notes, scaled down to an FFN. Nothing sharded yet
— we first need to *see* the full model we're about to cut up.

### 2. Flatten into the FlatParameter (the "ribbon")
Concatenate every parameter tensor end-to-end into **one 1D tensor**. **Anatomy:** confirm
`ribbon.numel() == P`, watch how a `(1024, 784)` matrix collapses into a flat run of 802816
numbers, and record the **offset table** (name, shape, start, end) — the "tiny shape metadata kept
on every GPU" from your notes. This table is the *only* thing that knows what's what; the ribbon
itself is "dumb storage."

### 3. Padding so the ribbon divides evenly by N
`L` is almost never divisible by `N`. FSDP zero-pads the ribbon tail so it splits into `N` equal
pieces. **Anatomy:** compute `pad = (-P) % N`, show `L = P + pad`, and confirm `L % N == 0`.
Understand *why* equal shards matter (one big efficient collective vs. many ragged ones).

### 4. Slice into N shards — sharding at rest
Split the padded ribbon into `N` chunks; each is one rank's permanent shard. **Anatomy:** print
`shard_len`, show that `sum(shard.numel() for shard in shards) == L`, and confirm each rank now
holds only **1/N** of the model. This is the state "before any training step runs."

### 5. Reconstruct a parameter from shards + metadata (mental unshard)
Before automating, do it by hand: given the 8 shards and the offset table, recover *one* weight
matrix (e.g. `fc1.weight`) back to its `(1024, 784)` shape. **Anatomy:** show that a single logical
tensor can span a shard boundary — its numbers may live on two different "ranks." This is the crux
of why FSDP flattens first and slices blindly.

### 6. All-gather — reconstruct the full ribbon (forward, step 1)
Implement all-gather as `torch.cat(shards)` → full ribbon, then unflatten via the offset table into
named tensors. **Anatomy:** all ranks now hold an **identical** full copy (redundant on weights),
while each rank still has **its own data batch** (different on data) — the exact split from notes
§2–3. Measure the transient memory: full ribbon exists *now* that didn't exist at rest.

### 7. Forward compute + discard
Run the FFN forward from the gathered weights on each rank's own micro-batch, produce logits and
loss, then **discard** the gathered full weights — fall back to the 1/N shard. **Anatomy:** shapes
of `x`, hidden activations, logits, and loss per rank; confirm the full ribbon is gone after
compute ("only one unit's full weights exist at any instant").

### 8. Backward — all-gather *again*, compute gradients
Because forward discarded the full weights, backward must all-gather a **second time** before it can
get gradients (notes §4). **Anatomy:** obtain the gradient w.r.t. the *full ribbon* on each rank;
confirm `grad.shape == ribbon.shape`. Note each rank's gradient is **different** (different data) —
this is what must be combined next.

### 9. Reduce-scatter — the key FSDP collective
Sum the N full gradients, then hand each rank only its **1/N slice** — never materializing the full
summed gradient as a kept quantity (notes §5). **Anatomy:** `stack → sum → split`; show
`grad_shard_i` lines up byte-for-byte with `param_shard_i`. Contrast with DDP's all-reduce (which
would keep the *whole* averaged gradient on every rank).

### 10. Local optimizer step — communication-free
Each rank updates **only its own** parameter shard using **only its own** gradient shard (notes §6).
**Anatomy:** show the update touches `shard_len` numbers per rank, zero cross-rank communication,
and that optimizer state (if we add momentum) is therefore also only 1/N per rank — the biggest
memory win.

### 11. Assemble one complete training step
Wire §6→§10 into a single `fsdp_step(batch)`: gather → forward → discard → gather → backward →
reduce-scatter → local step. **Anatomy:** trace one batch end-to-end and confirm loss is finite and
shards changed. This is the atomic unit we'll repeat.

### 12. Loop to one full epoch on MNIST
Run `fsdp_step` over the whole MNIST train loader for one epoch. **Anatomy:** watch loss fall
(~2.30 → ~0.2-ish), confirm the sharded model is genuinely learning, and sanity-check that
sharded training matches a plain single-model baseline on the same data (they should track).

### 13. Sanity check — sharded vs. unsharded equivalence
Train a normal (unsharded) FFN with the same init, data order, and LR; verify the FSDP-simulated
run produces (near-)identical loss curve. **Anatomy:** this proves "sharding is invisible to the
math" from the notes — any divergence means a bug in our gather/scatter, not in FSDP the idea.

### 14. Communication accounting — 3P vs 2P, measured
Instrument our collectives to count bytes moved: forward all-gather (1P) + backward all-gather (1P)
+ reduce-scatter (1P) = **3P**, vs DDP's single all-reduce ≈ **2P** (notes §8). **Anatomy:** print
the actual element counts from our runs and confirm the 1.5× ratio with real numbers.

### 15. Scaling knobs & bridge to real FSDP
Vary `N` (2, 4, 8, 16) and watch `shard_len` shrink while the loss curve stays identical — memory
layout changes, math doesn't. Then map every hand-written piece to its real PyTorch counterpart
(`FullyShardedDataParallel` / `fully_shard`, NCCL collectives, CUDA-stream overlap from notes §7),
and note what a real multi-GPU run would add (true parallelism, prefetch overlap) that our
single-process sim deliberately fakes.

---

## Stretch topics (optional, after the epoch runs)
- **S1. Overlap intuition (notes §7):** why prefetching the next unit's all-gather hides latency —
  we can't truly overlap in one process, but we can reason about where it would happen.
- **S2. Adam optimizer state sharding:** add momentum/variance and confirm they cost 1/N per rank.
- **S3. Multi-unit FFN:** split the FFN into 2 FSDP units (2 ribbons) to mimic "one ribbon per
  block," and watch gather/discard happen unit-by-unit rather than all at once.
- **S4. The noise-scale ceiling (notes §9):** why more ranks stops helping past `B_noise` — a
  data-parallel limit FSDP does *not* fix.

---

*Progress tracker (we tick these off as we go):*
- [ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5  [ ] 6  [ ] 7  [ ] 8
- [ ] 9  [ ] 10  [ ] 11  [ ] 12  [ ] 13  [ ] 14  [ ] 15
