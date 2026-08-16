"""Compute per-record word counts for the extracted TinyStoriesInstruct JSONL files.

Splits each file into byte-range chunks and counts words for the
"instruction" and "story" fields of every record in parallel via
multiprocessing.Pool, same approach as extract_tinystories_instruct.py.
Must be run as a script, not pasted into a notebook cell -- see the comment
in extract_tinystories_instruct.py for why (spawn start method needs a real
__main__ module to re-import `worker` in each subprocess).

Usage:
    python word_count_tinystories_instruct.py
    python word_count_tinystories_instruct.py --workers 4
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Datasets"


def iter_chunk_lines(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            f.readline()  # discard partial line, align to next full line
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield line


def worker(args):
    path, start, end = args
    instruction_wc = []
    story_wc = []
    for line in iter_chunk_lines(path, start, end):
        record = json.loads(line)
        instruction_wc.append(len(record["instruction"].split()))
        story_wc.append(len(record["story"].split()))
    return instruction_wc, story_wc


def count_words_parallel(path, n_workers):
    size = path.stat().st_size
    chunk = size // n_workers + 1
    tasks = [(path, w * chunk, min((w + 1) * chunk, size)) for w in range(n_workers)]
    tasks = [t for t in tasks if t[1] < t[2]]

    with mp.Pool(n_workers) as pool:
        results = pool.map(worker, tasks)

    instruction_wc = [c for r in results for c in r[0]]
    story_wc = [c for r in results for c in r[1]]
    return instruction_wc, story_wc


def main():
    parser = argparse.ArgumentParser(description="Compute word-count distributions for TinyStoriesInstruct JSONL files.")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                         help="number of worker processes (default: all cores)")
    args = parser.parse_args()

    import numpy as np

    out = {}
    for split, fname in [("train", "tinystories_instruct_train.jsonl"),
                          ("val", "tinystories_instruct_val.jsonl")]:
        path = DATASET_DIR / fname
        instruction_wc, story_wc = count_words_parallel(path, args.workers)
        out[f"{split}_instruction_wc"] = np.array(instruction_wc, dtype=np.int32)
        out[f"{split}_story_wc"] = np.array(story_wc, dtype=np.int32)
        print(f"{split}: {len(instruction_wc):,} stories -> {path}")

    out_path = DATASET_DIR / "tinystories_instruct_word_counts.npz"
    np.savez(out_path, **out)
    print(f"Saved word counts -> {out_path}")


if __name__ == "__main__":
    main()
