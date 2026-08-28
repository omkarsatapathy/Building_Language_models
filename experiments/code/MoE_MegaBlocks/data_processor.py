"""Tokenize TinyStories up to a desired number of tokens (not characters).

Streams the raw text story-by-story, tokenizes with tiktoken (GPT-2) using
multithreading, and stops once the target token count is reached. Saves the
consumed text plus train/val token files.

Reads:
    Datasets/TinyStoriesV2-GPT4-train.txt        (produced by download_toy_dataset.py)

Writes (label = token target, e.g. "500M"):
    Datasets/processed_dataset/tiny_stories_500M/tiny_stories_500M.txt
    Datasets/processed_dataset/tiny_stories_500M/train_500M.npy
    Datasets/processed_dataset/tiny_stories_500M/val_500M.npy

Usage:
    python data_processor.py                    # process the ENTIRE raw file (label 'all')
    python data_processor.py --tokens 500M
    python data_processor.py --tokens 5000000 --val-frac 0.01
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
READ_BLOCK_CHARS = 16_000_000        # size of each raw read from disk
BATCH_CHARS = 64_000_000             # collect ~this many chars before tokenizing


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_count(text: str):
    """'500M' -> (500_000_000, '500M'); '5000000' -> (5000000, '5000000')."""
    text = text.strip()
    label = text.upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = label[-1]
    if suffix in multipliers:
        n = int(float(label[:-1]) * multipliers[suffix])
    else:
        n = int(text)
        label = text  # keep raw digits as the label
    return n, label


def stream_stories(f):
    """Yield complete stories (SEP-delimited) from an open text file."""
    buf = ""
    while True:
        block = f.read(READ_BLOCK_CHARS)
        if not block:
            break
        buf += block
        parts = buf.split(SEP)
        buf = parts.pop()             # last piece may be an incomplete story
        for p in parts:
            if p.strip():
                yield p
    if buf.strip():
        yield buf


def encode_batch_of_stories(docs, enc):
    """Regroup docs into a few big chunks and encode them across all cores."""
    import numpy as np

    n_chunks = max(1, N_THREADS * CHUNKS_PER_CORE)
    target = max(1, sum(len(d) for d in docs) // n_chunks)
    chunks, part, size = [], [], 0
    for d in docs:
        part.append(d)
        size += len(d) + len(SEP)
        if size >= target:
            chunks.append(SEP.join(part) + SEP)   # trailing SEP -> boundary token
            part, size = [], 0
    if part:
        chunks.append(SEP.join(part) + SEP)

    batches = enc.encode_batch(chunks, allowed_special={SEP}, num_threads=N_THREADS)
    return np.concatenate([np.asarray(b, dtype=np.uint16) for b in batches])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Tokenize TinyStories to a token target.")
    parser.add_argument("--tokens", default=None,
                        help="desired number of tokens, e.g. 100M / 1B / 5000000. "
                             "If omitted, process the entire raw file.")
    parser.add_argument("--val-frac", type=float, default=0.005,
                        help="fraction of tokens held out for validation")
    args = parser.parse_args()

    tiktoken = ensure_tiktoken()
    import numpy as np

    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab < 2**16, "gpt2 vocab must fit in uint16"

    if args.tokens is None:
        target_tokens, label = None, "all"      # process the whole file
        print(f"Target: ALL tokens (label '{label}'), using {N_THREADS} cores")
    else:
        target_tokens, label = parse_count(args.tokens)
        print(f"Target: {target_tokens:,} tokens (label '{label}'), using {N_THREADS} cores")

    if not RAW_TRAIN.exists():
        raise FileNotFoundError(f"{RAW_TRAIN} not found. Run download_toy_dataset.py first.")

    out_dir = PROCESSED_DIR / f"tiny_stories_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"tiny_stories_{label}.txt"
    bin_path = out_dir / f"tokens_{label}.bin"      # temp; removed after splitting

    # Stream + tokenize batches, writing text and tokens straight to disk so we
    # never hold the whole corpus in RAM (this caused OOM kills on large runs).
    total_tokens = 0
    batch, batch_chars = [], 0

    def flush(batch):
        nonlocal total_tokens
        arr = encode_batch_of_stories(batch, enc)
        # trim the final batch so we stop at the exact token target
        if target_tokens is not None and total_tokens + arr.size > target_tokens:
            arr = arr[: target_tokens - total_tokens]
        txt_f.write(SEP.join(batch) + SEP)
        arr.tofile(tok_f)
        total_tokens += arr.size
        print(f"  ...{total_tokens:,} tokens")

    with RAW_TRAIN.open("r", encoding="utf-8") as f, \
            txt_path.open("w", encoding="utf-8") as txt_f, \
            bin_path.open("wb") as tok_f:
        for story in stream_stories(f):
            batch.append(story)
            batch_chars += len(story) + len(SEP)
            if batch_chars >= BATCH_CHARS:
                flush(batch)
                batch, batch_chars = [], 0
                if target_tokens is not None and total_tokens >= target_tokens:
                    break
        else:
            if batch:                               # ran out of file before target
                flush(batch)

    if target_tokens is not None and total_tokens < target_tokens:
        print(f"[warn] raw file exhausted at {total_tokens:,} tokens (< {target_tokens:,})")
    print(f"total: {total_tokens:,} tokens -> text {txt_path}")

    # Memory-map the token file and split train/val without loading it all.
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_val = max(1, int(tokens.size * args.val_frac))
    splits = {"train": tokens[:-n_val], "val": tokens[-n_val:]}
    for split, arr in splits.items():
        out = out_dir / f"{split}_{label}.npy"
        np.save(out, arr)                            # streams from the mmap view
        print(f"{split}: {arr.size:,} tokens -> {out}")

    del tokens                                       # release the mmap, then clean up
    bin_path.unlink()


if __name__ == "__main__":
    main()
