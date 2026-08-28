"""Download the TinyStories (V2, GPT-4) toy dataset and store it under Datasets/.

Files come from the HuggingFace-hosted raw text:
    TinyStoriesV2-GPT4-train.txt  (~2.2 GB)
    TinyStoriesV2-GPT4-valid.txt  (~22 MB)

Usage:
    python download_toy_dataset.py            # download both
    python download_toy_dataset.py --val-only # just the small validation file
"""
import argparse
import sys
import urllib.request
from pathlib import Path

# Store under <repo>/Datasets  (two levels up: test_MoE -> code -> experiments,
# then up to repo root). Adjust if you want a different anchor.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "Datasets"

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "val": "TinyStoriesV2-GPT4-valid.txt",
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:,.1f} / {total_mb:,.1f} MB)")
    else:
        sys.stdout.write(f"\r  {downloaded / 1024 / 1024:,.1f} MB")
    sys.stdout.flush()


def download(name: str) -> None:
    fname = FILES[name]
    url = f"{BASE_URL}/{fname}"
    dest = DATASET_DIR / fname

    if dest.exists():
        print(f"[skip] {fname} already exists ({dest.stat().st_size / 1024 / 1024:,.1f} MB)")
        return

    print(f"[download] {fname}\n  from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.rename(dest)
        print(f"\n[done] saved to {dest}")
    except Exception:
        if tmp.exists():
            tmp.unlink()  # don't leave a half file behind
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the TinyStories toy dataset.")
    parser.add_argument("--val-only", action="store_true", help="download only the validation split")
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {DATASET_DIR}")

    splits = ["val"] if args.val_only else ["train", "val"]
    for split in splits:
        download(split)


if __name__ == "__main__":
    main()
