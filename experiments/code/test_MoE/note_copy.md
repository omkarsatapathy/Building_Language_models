# Notes — test_MoE

## RoPE: things to handle when wiring into CausalSelfAttention

These are follow-ups from the RoPE implementation, to take care of at the **attention call** stage:

1. **Mixed-precision (bf16/fp16) safety.**
   Build the cos/sin cache in float32 (preserves angle precision), but cast to `x.dtype` at apply time. Otherwise `x(bf16) * cos(fp32)` upcasts the output to fp32 and breaks the rest of the attention path.
   ```python
   cos = cos[:T][None, None, :, :].to(dtype=x.dtype)
   sin = sin[:T][None, None, :, :].to(dtype=x.dtype)
   ```

2. **Slice the cache to the actual sequence length T.**
   Cache is built up to `block_size`, but a forward pass may have `T < block_size`, so slice `cos[:T]` / `sin[:T]`.

3. **Build the cache ONCE (GPU / hot-path).**
   Do not call `build_RoPE_cache` every forward. Build it in `__init__` and register as a non-persistent buffer so it moves with `.to(device)` and stays out of the state_dict:
   ```python
   cos, sin = build_RoPE_cache(config.block_size, head_dim)
   self.register_buffer("rope_cos", cos, persistent=False)
   self.register_buffer("rope_sin", sin, persistent=False)
   ```

4. **Drop `device=device` default arg in `build_RoPE_cache`.**
   Once the cache is a buffer, device is handled by `.to(device)`; the hardcoded global becomes redundant and risks a device mismatch.

5. **Guard: `assert head_dim % 2 == 0`** — RoPE needs even head_dim (768/8 = 96 ✓).

6. **Apply RoPE to q and k only, not v** — after splitting into heads `[B, n_head, T, head_dim]`, before computing attention scores.
