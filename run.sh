#!/usr/bin/env bash
# Train the GPT-MoE model.
#   ./run.sh smoke   -> quick sanity run (tiny steps, validates plumbing + CSV)
#   ./run.sh full    -> full training run (file defaults: 5000 steps)
set -euo pipefail

# Always run from the repo root (relative data paths depend on it), regardless
# of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Put the model package on the path so `from swiglu import ...` resolves.
export PYTHONPATH="experiments/code/test_MoE"
MODEL="experiments/code/test_MoE/model.py"

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    echo "🔥 smoke test (20 steps, small batch/seq to fit MPS)"
    MAX_STEPS=20 WARMUP_STEPS=5 EVAL_EVERY=10 EVAL_STEPS=5 GRAD_ACCUM_STEPS=2 \
    BATCH_SIZE=4 BLOCK_SIZE=256 \
      python "$MODEL"
    ;;
  full)
    echo "🚀 full run (file defaults)"
    python "$MODEL"
    ;;
  *)
    echo "usage: ./run.sh [smoke|full]" >&2
    exit 1
    ;;
esac
