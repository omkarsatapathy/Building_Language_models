#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install all dependencies for Building_Language_models on a Linux CUDA box.
#
# Target: NVIDIA H100 80GB (SM 9.0 / Hopper), Linux x86_64, Python 3.11, CUDA 12.8
#
#   ./install.sh              -> install into the active environment
#   ./install.sh --venv       -> create ./.venv first, then install into it
#   ./install.sh --check      -> verify an existing install, change nothing
#
# Single command, start to finish. Handles the two things a bare
# `pip install -r requirements.txt` gets wrong:
#   1. --no-build-isolation, so megablocks builds against the torch we install
#      instead of trying to resolve its own torch from source and failing.
#   2. TORCH_CUDA_ARCH_LIST=9.0, so the fused CUDA kernels build for Hopper only
#      (correct arch, and far faster than fanning out over every arch).
# ---------------------------------------------------------------------------
set -euo pipefail

# Always run from the repo root, regardless of where this is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CUDA_INDEX="https://download.pytorch.org/whl/cu128"
MODE="${1:-install}"

# H100 = compute capability 9.0. Building for just this arch keeps nvcc from
# compiling every supported arch (minutes vs. tens of minutes) and is what the
# grouped-GEMM / block-sparse kernels are tuned for.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# Each nvcc job is memory-hungry; unbounded parallelism OOMs smaller nodes.
export MAX_JOBS="${MAX_JOBS:-8}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# verify: import the stack and confirm the GPU is actually usable
# ---------------------------------------------------------------------------
verify() {
  log "Verifying installation"
  python - <<'PY'
import sys

def line(label, value): print(f"  {label:<22} {value}")

import torch
line("python", sys.version.split()[0])
line("torch", torch.__version__)
line("cuda available", torch.cuda.is_available())

if torch.cuda.is_available():
    line("cuda (torch build)", torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        line(f"gpu {i}", f"{name} (sm_{cap[0]}{cap[1]}, {mem:.0f} GB)")
        if cap[0] < 8:
            print(f"  !! sm_{cap[0]}{cap[1]} < sm_80: megablocks bf16 grouped-GEMM "
                  "paths need Ampere or newer", file=sys.stderr)
    line("bf16 supported", torch.cuda.is_bf16_supported())
else:
    print("  !! No CUDA device visible. Training and the megablocks kernels "
          "both require one.", file=sys.stderr)

for mod in ("torchvision", "triton", "numpy", "tiktoken", "regex",
            "sentencepiece", "datasets", "matplotlib"):
    try:
        m = __import__(mod)
        line(mod, getattr(m, "__version__", "ok"))
    except Exception as e:
        print(f"  !! {mod}: {e}", file=sys.stderr)

# megablocks imports its compiled CUDA extension at import time, so a successful
# import is a real check that the kernels built and loaded — not just that the
# wheel is on disk.
try:
    import megablocks
    line("megablocks", getattr(megablocks, "__version__", "ok"))
    from megablocks.layers.dmoe import dMoE  # noqa: F401
    line("megablocks.dMoE", "import ok")
except Exception as e:
    print(f"  !! megablocks: {e}", file=sys.stderr)

try:
    import grouped_gemm  # noqa: F401
    line("grouped_gemm", "import ok")
except Exception as e:
    print(f"  !! grouped_gemm (fused expert GEMM): {e}", file=sys.stderr)
PY
}

if [[ "$MODE" == "--check" ]]; then
  verify
  exit 0
fi

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
[[ -f requirements.txt ]] || die "requirements.txt not found in $REPO_ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  warn "This script targets Linux + CUDA. megablocks/grouped_gemm build CUDA"
  warn "kernels and will not install on $(uname -s)."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU detected"
  nvidia-smi --query-gpu=name,memory.total,driver_version \
             --format=csv,noheader | sed 's/^/  /'
else
  warn "nvidia-smi not found — no GPU visible. The CUDA kernel builds will fail."
fi

# The kernel builds shell out to nvcc; it must exist and should match the torch
# CUDA version (12.8 here). A mismatch produces confusing link errors later.
if command -v nvcc >/dev/null 2>&1; then
  log "nvcc: $(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)"
else
  warn "nvcc not on PATH. megablocks/grouped_gemm/stanford-stk cannot compile."
  warn "Install the CUDA toolkit, or set CUDA_HOME and add \$CUDA_HOME/bin to PATH."
fi

# ---------------------------------------------------------------------------
# optional venv
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--venv" ]]; then
  if [[ ! -d .venv ]]; then
    log "Creating virtualenv at ./.venv"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Using $(python -V) at $(command -v python)"
fi

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
log "Upgrading pip toolchain"
# setuptools is capped: megablocks 0.10.0 requires setuptools < 79.0.0.
python -m pip install --upgrade "pip>=24.0" "setuptools<79.0.0" wheel

# torch must land BEFORE the kernel packages. With --no-build-isolation their
# setup.py does `import torch` at build time, so it has to already be importable.
log "Installing torch 2.7.0 (cu128) — required before the CUDA kernel builds"
python -m pip install --extra-index-url "$CUDA_INDEX" \
  torch==2.7.0 torchvision==0.22.0 triton==3.3.0

log "Building megablocks + CUDA kernels (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST, MAX_JOBS=$MAX_JOBS)"
log "This compiles fused block-sparse kernels and can take several minutes."
python -m pip install -r requirements.txt \
  --extra-index-url "$CUDA_INDEX" \
  --no-build-isolation

verify

log "Done."
if [[ "$MODE" == "--venv" ]]; then
  echo "  Activate with: source .venv/bin/activate"
fi
echo "  Smoke test:    ./run.sh smoke"
echo "  Full training: ./run.sh full"
