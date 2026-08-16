"""Flatten the extracted TinyStoriesInstruct JSONL files into a single text file per split.

Each JSONL record {"instruction": ..., "story": ...} becomes one line shaped like:

    <|startofinstruction|>INSTRUCTION<|endofinstruction|><|startofstory|>STORY<|endofstory|>

so a tokenizer can learn/emit the four markers as special tokens later. Records
are written back to back (one per line) with no other separator -- the markers
themselves delimit the corpus, and the story's own newlines are preserved
verbatim by escaping them, so one record always stays on one line.

Reads:
    Datasets/tinystories_instruct_train.jsonl   (produced by extract_tinystories_instruct.py)
    Datasets/tinystories_instruct_val.jsonl

Writes:
    Datasets/tinystories_instruct_train.txt
    Datasets/tinystories_instruct_val.txt

Splits each input into byte-range chunks and converts them in parallel via
multiprocessing.Pool -- several chunks per core so no core idles on a slow
chunk, which keeps every CPU pinned for the whole run. Must be run as a script,
not pasted into a notebook cell -- see the docstring of
extract_tinystories_instruct.py for why (the "spawn" start method needs a real
__main__ module to re-import `worker` in each subprocess).

Usage:
    python build_tinystories_instruct_txt.py
    python build_tinystories_instruct_txt.py --workers 4
    python build_tinystories_instruct_txt.py --keep-newlines   # raw multi-line stories
"""
import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Datasets"

INSTRUCTION_START = "<|startofinstruction|>"
INSTRUCTION_END = "<|endofinstruction|>"
STORY_START = "<|startofstory|>"
STORY_END = "<|endofstory|>"

CHUNKS_PER_CORE = 4          # a few chunks per core -> load balancing, not millions of tasks
WRITE_BUFFER_CHARS = 8_000_000   # join this much text before hitting the disk


def iter_chunk_lines(path, start, end):
    """Yield the full JSONL lines whose *start* offset falls inside [start, end)."""
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
    path, start, end, out_path, keep_newlines = args
    count = 0
    buf, buf_chars = [], 0

    with open(out_path, "w", encoding="utf-8") as out:
        for line in iter_chunk_lines(path, start, end):
            record = json.loads(line)
            instruction = record["instruction"].strip()
            story = record["story"].strip()
            if not keep_newlines:
                # keep one record on one line so the file stays line-addressable;
                # "\n" is re-expanded at tokenization time
                story = story.replace("\\", "\\\\").replace("\n", "\\n")

            buf.append(
                f"{INSTRUCTION_START}{instruction}{INSTRUCTION_END}"
                f"{STORY_START}{story}{STORY_END}\n"
            )
            buf_chars += len(buf[-1])
            count += 1

            if buf_chars >= WRITE_BUFFER_CHARS:
                out.write("".join(buf))
                buf, buf_chars = [], 0

        if buf:
            out.write("".join(buf))

    return count


def build_parallel(in_path, out_path, n_workers, keep_newlines):
    """Convert in_path -> out_path across n_workers processes, preserving record order."""
    size = in_path.stat().st_size
    n_chunks = max(1, n_workers * CHUNKS_PER_CORE)
    chunk = size // n_chunks + 1

    tasks = []
    for c in range(n_chunks):
        start = c * chunk
        end = min(start + chunk, size)
        if start >= end:
            continue
        shard = out_path.with_name(out_path.stem + f"_part{c}.txt")
        tasks.append((in_path, start, end, shard, keep_newlines))

    with mp.Pool(n_workers) as pool:
        counts = pool.map(worker, tasks)   # ordered results -> shards merge in order

    merge_shards([t[3] for t in tasks], out_path)
    return sum(counts)


def merge_shards(shard_paths, final_path):
    """Concatenate shards into final_path by streaming, then delete them."""
    with open(final_path, "wb") as out:
        for p in shard_paths:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, length=8 * 1024 * 1024)
            p.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Wrap TinyStoriesInstruct instructions/stories in special tokens and merge into one .txt per split."
    )
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="number of worker processes (default: all cores)")
    parser.add_argument("--keep-newlines", action="store_true",
                        help="write stories with their original line breaks instead of escaped \\n")
    args = parser.parse_args()

    for split, fname in [("train", "tinystories_instruct_train"),
                         ("val", "tinystories_instruct_val")]:
        in_path = DATASET_DIR / f"{fname}.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(f"{in_path} not found. Run extract_tinystories_instruct.py first.")

        out_path = DATASET_DIR / f"{fname}.txt"
        count = build_parallel(in_path, out_path, args.workers, args.keep_newlines)
        mb = out_path.stat().st_size / 1e6
        print(f"{split}: {count:,} records, {mb:,.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
