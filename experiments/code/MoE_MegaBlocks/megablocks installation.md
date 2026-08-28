# MegaBlocks + Torch 2.7.0 (cu128) — Lightning.ai H100 Setup Runbook

Target: NVIDIA H100 80GB, Lightning.ai `cloudspace` conda env, Python 3.12.
Fixes the "CUDA version mismatch" and missing-header build failures for
`megablocks[gg]==0.10.0` + `grouped_gemm==0.3.0` + `stanford-stk==0.7.1`.

## Why this is needed
Lightning's default image ships a **nightly torch** built against **CUDA 13.0**,
while `megablocks==0.10.0` hard-pins `torch>=2.7.0,<2.7.1` (built against CUDA 12.8).
The system `nvcc` (from `/usr/local/cuda`) is also 13.0, which torch's build
system rejects with a strict version-match check when compiling extensions.
Fix: install a CUDA 12.8 toolchain via conda inside the same env, and point
the build at pip-installed NVIDIA headers (torch's runtime deps) instead of
system CUDA headers.

---

## 1. Sanity check GPU + current torch
```bash
nvidia-smi
python --version && pip show torch 2>&1 | head -5
which python && which pip && nvcc --version
echo $CUDA_HOME
```

## 2. Remove the mismatched torch stack
```bash
pip uninstall -y torch torchvision torchaudio triton
```

## 3. Install the pinned torch stack (cu128, matches megablocks' requirement)
```bash
pip install torch==2.7.0 torchvision==0.22.0 triton==3.3.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch, triton; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '| triton', triton.__version__)"
```
Expect: `torch 2.7.0+cu128 | cuda 12.8 | available True | triton 3.3.0`

## 4. Pin numpy/packaging (megablocks constrains both)
```bash
pip install "numpy>=1.26,<2.1.0" "packaging>=21.3.0,<24.2"
```

## 5. Install a matching CUDA 12.8 `nvcc` via conda
The system CUDA toolkit (13.0) will NOT match torch's cu128 build — building
any extension against it fails with:
`RuntimeError: The detected CUDA version (13.0) mismatches the version that
was used to compile PyTorch (12.8)`

Pip's `nvidia-cuda-nvcc-cu12` wheel does **not** include the `nvcc` binary
itself (only `ptxas`/`nvvm`), so use conda instead — pin every package
explicitly or conda will silently pull the latest CUDA version:

```bash
conda install -y -c "nvidia/label/cuda-12.8.0" cuda-nvcc=12.8.61 cuda-cudart-dev=12.8.57
```

Point the shell at the new compiler (must come before system CUDA on PATH):
```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version   # should report release 12.8, V12.8.61
```

## 6. Add pip-installed NVIDIA headers to CPATH
Torch's headers (`ATen/cuda/CUDAContext.h`) pull in `cusparse.h`, `cublas_v2.h`,
etc. — these ship as headers inside torch's own pip dependencies
(`nvidia-cusparse-cu12`, `nvidia-cublas-cu12`, ...), not in the conda CUDA
toolkit or system CUDA. Add them all to `CPATH` in one shot:

```bash
export CPATH=$(find $CONDA_PREFIX/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name include | tr '\n' ':')$CPATH
```

## 7. Build megablocks, grouped_gemm, stanford-stk
Install one at a time — if any package in a combined `pip install` fails to
build, pip aborts the **whole transaction**, silently dropping packages that
built successfully (e.g. megablocks can build fine while grouped_gemm fails,
and pip won't install megablocks either).

```bash
pip install megablocks==0.10.0 --no-build-isolation
pip install grouped_gemm==0.3.0 --no-build-isolation
pip install stanford-stk==0.7.1 --no-build-isolation
```

## 8. Verify everything imports and the arch target is right
```bash
export TORCH_CUDA_ARCH_LIST="9.0"   # H100 = sm_90; use "8.0" for A100
export MAX_JOBS=8

python -c "import megablocks.layers.dmoe as dmoe; import megablocks.layers.arguments as args; import megablocks.layers.moe as moe; import grouped_gemm; import stk; print('ALL OK')"
```

## 9. Persist env vars (session-scoped exports vanish on new shells/restarts)
```bash
{
  echo "export CUDA_HOME=$CONDA_PREFIX"
  echo 'export PATH=$CUDA_HOME/bin:$PATH'
  echo "export CPATH=$CPATH"
  echo 'export TORCH_CUDA_ARCH_LIST="9.0"'
  echo 'export MAX_JOBS=8'
} >> ~/.bashrc
```

---

## Known code-level gotcha (not an install issue)
`megablocks.layers.moe.batched_load_balancing_loss(mb_args)` aggregates
load-balancing stats across **all** MoE layers in the model, using
`mb_args.num_layers` to know how many to expect. If `build_mb_args()` never
sets `num_layers`, it defaults to `1` and you'll hit:

```
ValueError: Expected 1 token_per_experts but found <n_layer>.
```

Fix — pass the real layer count when constructing `Arguments`:
```python
return mb_args_mod.Arguments(
    ...
    num_layers=config.n_layer,   # <-- add this
    ...
)
```

---

## Quick reference — architecture flags
| GPU  | `TORCH_CUDA_ARCH_LIST` |
|------|------------------------|
| H100 | `"9.0"`                |
| A100 | `"8.0"`                |

MegaBlocks is **CUDA-only** — no CPU fallback (`build_mb_args()` hardcodes
`device=torch.device("cuda")`, and the grouped-GEMM kernels require CUDA).