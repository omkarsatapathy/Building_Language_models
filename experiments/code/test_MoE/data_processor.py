"""Slice a chunk of the raw TinyStories text, tokenize it with tiktoken (GPT-2)
using multithreading, and save the trimmed text + train/val token files.

Reads:
    Datasets/TinyStoriesV2-GPT4-train.txt        (produced by download_toy_dataset.py)

Writes (label = chunk size, e.g. "100M"):
    Datasets/processed_dataset/tiny_stories_100M/tiny_stories_100M.txt
    Datasets/processed_dataset/tiny_stories_100M/train_100M.npy
    Datasets/processed_dataset/tiny_stories_100M/val_100M.npy

Usage:
    python data_processor.py                     # default chunk (100M chars)
    python data_processor.py --chunk-size 500M
    python data_processor.py --chunk-size 5000000 --val-frac 0.01
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# tiktoken bootstrap: install it if the import fails, then import.
# --------------------------------------------------------------------------- #
def ensure_tiktoken():
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("[setup] tiktoken not found -> pip installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
    import tiktoken
    return tiktoken


# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "Datasets"
RAW_TRAIN = DATASET_DIR / "TinyStoriesV2-GPT4-train.txt"
PROCESSED_DIR = DATASET_DIR / "processed_dataset"

SEP = "<|endoftext|>"                 # story delimiter in the raw text
N_THREADS = os.cpu_count() or 4
CHUNKS_PER_CORE = 4                   # a few big chunks per core, NOT millions


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_chunk_size(text: str):
    """'100M' -> (100_000_000, '100M'); '5000000' -> (5000000, '5000000')."""
    text = text.strip()
    label = text.upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = label[-1]
    if suffix in multipliers:
        n_chars = int(float(label[:-1]) * multipliers[suffix])
    else:
        n_chars = int(text)
        label = text  # keep raw digits as the label
    return n_chars, label


def read_chunk(n_chars: int) -> str:
    """Read ~n_chars characters, then trim back to the last full story boundary."""
    if not RAW_TRAIN.exists():
        raise FileNotFoundError(
            f"{RAW_TRAIN} not found. Run download_toy_dataset.py first."
        )
    with RAW_TRAIN.open("r", encoding="utf-8") as f:
        chunk = f.read(n_chars)
    # cut at the last complete story so we never emit a truncated tail
    cut = chunk.rfind(SEP)
    if cut != -1:
        chunk = chunk[: cut + len(SEP)]
    return chunk


def encode_split(docs, enc):
    """Regroup docs into a few big chunks and encode them across all cores."""
    import numpy as np

    n_chunks = max(1, N_THREADS * CHUNKS_PER_CORE)
    target = max(1, sum(len(d) for d in docs) // n_chunks)
    chunks, buf, size = [], [], 0
    for d in docs:
        buf.append(d)
        size += len(d) + len(SEP)
        if size >= target:
            chunks.append(SEP.join(buf) + SEP)   # trailing SEP -> boundary token
            buf, size = [], 0
    if buf:
        chunks.append(SEP.join(buf) + SEP)

    # encode the big chunks in parallel; allowed_special turns SEP into id 50256
    batches = enc.encode_batch(
        chunks,
        allowed_special={SEP},
        num_threads=N_THREADS,
    )
    return np.concatenate([np.asarray(b, dtype=np.uint16) for b in batches])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Chunk + tokenize TinyStories.")
    parser.add_argument("--chunk-size", default="100M",
                        help="chars to take from the raw file, e.g. 100M / 1B / 5000000")
    parser.add_argument("--val-frac", type=float, default=0.005,
                        help="fraction of stories held out for validation")
    args = parser.parse_args()

    tiktoken = ensure_tiktoken()
    import numpy as np

    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab < 2**16, "gpt2 vocab must fit in uint16"

    n_chars, label = parse_chunk_size(args.chunk_size)
    print(f"Chunk: {n_chars:,} chars (label '{label}'), using {N_THREADS} cores")

    # 1) slice the raw text
    text = read_chunk(n_chars)
    stories = [s for s in text.split(SEP) if s.strip()]
    del text
    print(f"{len(stories):,} stories in chunk")

    # 2) output directory
    out_dir = PROCESSED_DIR / f"tiny_stories_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) save the trimmed text (rejoined with SEP so it round-trips)
    txt_path = out_dir / f"tiny_stories_{label}.txt"
    txt_path.write_text(SEP.join(stories) + SEP, encoding="utf-8")
    print(f"text: -> {txt_path}")

    # 4) split + tokenize + save
    n_val = max(1, int(len(stories) * args.val_frac))
    splits = {"train": stories[:-n_val], "val": stories[-n_val:]}
    for split, docs in splits.items():
        arr = encode_split(docs, enc)
        out = out_dir / f"{split}_{label}.npy"
        np.save(out, arr)
        print(f"{split}: {arr.size:,} tokens -> {out}")


if __name__ == "__main__":
    main()
