"""Tokenize the TinyStories text into uint16 .npy token files (train/val).

Run:  python experiments/code/prepare_data.py
Output: toy_datasets/train_100M.npy, toy_datasets/val_100M.npy
"""
import os
from pathlib import Path

import numpy as np
import tiktoken

# --- config ---
DATA_PATH = Path("toy_datasets/tiny_stories_100M.txt")
OUT_DIR = Path("toy_datasets/tokenized_100M")
VAL_FRAC = 0.005                    # last 0.5% of stories held out for validation
SEP = "<|endoftext|>"              # story delimiter in the raw text
N_THREADS = os.cpu_count()
CHUNKS_PER_CORE = 4                 # a few big chunks per core, NOT millions

enc = tiktoken.get_encoding("gpt2")

# gpt2 vocab (50257) fits in uint16, so SEP -> 50256 fits too
assert enc.n_vocab < 2**16


def encode_split(docs):
    """Regroup docs into a few big chunks and encode them across all cores."""
    # regroup docs into ~equal, large chunks so each thread gets real work
    n_chunks = max(1, N_THREADS * CHUNKS_PER_CORE)
    target = sum(len(d) for d in docs) // n_chunks
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
    # concatenate at C level rather than via a Python list
    return np.concatenate([np.asarray(b, dtype=np.uint16) for b in batches])


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    stories = [s for s in text.split(SEP) if s.strip()]
    del text
    print(f"{len(stories):,} stories, using {N_THREADS} cores")

    n_val = int(len(stories) * VAL_FRAC)
    splits = {"train": stories[:-n_val], "val": stories[-n_val:]}

    OUT_DIR.mkdir(exist_ok=True)
    for split, docs in splits.items():
        arr = encode_split(docs)
        out = OUT_DIR / f"{split}_100M.npy"
        np.save(out, arr)
        print(f"{split}: {arr.size:,} tokens -> {out}")


if __name__ == "__main__":
    main()
