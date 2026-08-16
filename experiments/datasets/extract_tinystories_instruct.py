"""Extract (instruction, story) pairs from roneneldan/TinyStoriesInstruct.

Each raw record is a run of lines shaped like:
    Features: ...
    Words: ...
    Summary: ...
    Story: <story text, possibly multiple lines>
    <|endoftext|>

This turns each record into one JSON line: {"instruction": ..., "story": ...}.

Runs the per-split extraction across all CPU cores via multiprocessing.Pool.
Must be run as a script (not pasted into a notebook cell) -- macOS/Windows
spawn a fresh interpreter per worker and re-import __main__ to find `worker`;
inside a notebook kernel there is no real __main__ file to re-import, so the
workers crash with "Can't get attribute 'worker' on <module '__main__'>".

Usage:
    python extract_tinystories_instruct.py
    python extract_tinystories_instruct.py --workers 4
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

# The dataset is already fully cached locally (downloaded once via the
# notebook), so skip the network check load_dataset() otherwise makes on
# every call -- this is what triggers the "unauthenticated requests" warning
# and needless HF Hub traffic for a purely local job.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Datasets"

DELIM = "<|endoftext|>"
HF_DATASET = "roneneldan/TinyStoriesInstruct"


def worker(args):
    split_name, start, end, out_path = args
    ds = load_dataset(HF_DATASET, split=split_name)

    features, words, summary = None, None, None
    story_lines = []
    in_story = False
    count = 0

    with open(out_path, "w") as f:
        for i in range(start, end):
            line = ds[i]["text"]
            stripped = line.strip()

            if stripped.startswith("Features:"):
                features = stripped.replace("Features:", "").strip()
                in_story = False
            elif stripped.startswith("Words:"):
                words = stripped.replace("Words:", "").strip()
            elif stripped.startswith("Summary:"):
                summary = stripped.replace("Summary:", "").strip()
            elif stripped.startswith("Story:"):
                in_story = True
                story_lines = []
            elif stripped == DELIM:
                if features and words and summary and story_lines:
                    instruction = (
                        f"Write a story that includes the words {words}, "
                        f"features {features.lower()}, "
                        f"and follows this summary: {summary}"
                    )
                    story = "\n".join(story_lines).strip()
                    f.write(json.dumps({"instruction": instruction, "story": story}) + "\n")
                    count += 1
                features, words, summary, story_lines = None, None, None, []
                in_story = False
            elif in_story:
                story_lines.append(line)

    return count


def run_parallel(split_name, out_prefix, n_workers):
    ds = load_dataset(HF_DATASET, split=split_name)
    n = len(ds)
    chunk = n // n_workers + 1
    # small overlap so a record split exactly at a boundary isn't lost—
    # each worker starts at its boundary but only commits records whose
    # delimiter falls inside its own range, so overlap must be read-only lookahead
    tasks = []
    for w in range(n_workers):
        start = w * chunk
        end = min(start + chunk, n)
        if start >= end:
            continue
        out_path = out_prefix.with_name(out_prefix.name + f"_part{w}.jsonl")
        tasks.append((split_name, start, end, out_path))

    with mp.Pool(n_workers) as pool:
        counts = pool.map(worker, tasks)

    return sum(counts), [t[3] for t in tasks]


def merge_shards(shard_paths, final_path):
    with open(final_path, "w") as out:
        for p in shard_paths:
            with open(p) as f:
                out.write(f.read())
            p.unlink()


def main():
    parser = argparse.ArgumentParser(description="Extract TinyStoriesInstruct into instruction/story JSONL.")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                         help="number of worker processes (default: all cores)")
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    train_count, train_shards = run_parallel(
        "train", DATASET_DIR / "tinystories_instruct_train", args.workers
    )
    val_count, val_shards = run_parallel(
        "validation", DATASET_DIR / "tinystories_instruct_val", args.workers
    )

    merge_shards(train_shards, DATASET_DIR / "tinystories_instruct_train.jsonl")
    merge_shards(val_shards, DATASET_DIR / "tinystories_instruct_val.jsonl")

    print(f"Train stories: {train_count:,} -> {DATASET_DIR / 'tinystories_instruct_train.jsonl'}")
    print(f"Validation stories: {val_count:,} -> {DATASET_DIR / 'tinystories_instruct_val.jsonl'}")


if __name__ == "__main__":
    main()
