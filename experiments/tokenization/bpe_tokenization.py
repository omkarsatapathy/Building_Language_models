"""Byte-level BPE tokenizer for the TinyStoriesInstruct corpus.

Trains a GPT-2 style byte-level BPE from scratch (no HuggingFace, no tiktoken)
and gives the transformer an `encode` / `decode` pair to train on.

Design, in one breath:
  * every token is a sequence of *bytes*, so the base vocab is the 256 byte
    values and nothing can ever be out-of-vocabulary -- no <unk> needed;
  * text is first split by a GPT-4 style regex into "pre-tokens" (words,
    number runs, punctuation runs, whitespace runs). Merges never cross a
    pre-token boundary, which is what stops the tokenizer from learning
    junk like "dog." or ".\nThe";
  * the four corpus markers plus <|endoftext|> / <|pad|> are *special* tokens:
    they are carved out before the regex ever sees them and get their own ids
    at the very top of the vocab.

Reads (produced by ../datasets/build_tinystories_instruct_txt.py):
    Datasets/tinystories_instruct_train.txt
    Datasets/tinystories_instruct_val.txt

Writes:
    experiments/tokenization/tinystories_bpe_8k.json     (the tokenizer)
    experiments/tokenization/tinystories_bpe_8k.vocab    (human-readable dump)

Parallelism: counting pre-tokens is the only heavy pass over the corpus, and
it is embarrassingly parallel, so the file is cut into byte ranges (aligned to
line boundaries -- one record per line) and handed to a multiprocessing.Pool,
several chunks per core so no core idles on a slow chunk. The merge loop
itself is inherently sequential, so it stays single-process but uses an
indexed/incremental pair counter instead of rescanning the corpus 8k times.
Must be run as a script, not pasted into a notebook cell -- see the docstring
of ../datasets/extract_tinystories_instruct.py for why.

Usage:
    python bpe_tokenization.py train                        # full corpus, 8192 vocab
    python bpe_tokenization.py train --vocab-size 10000
    python bpe_tokenization.py train --max-bytes 100M       # quick run on a slice
    python bpe_tokenization.py stats                        # reprint the report
    python bpe_tokenization.py demo --text "Once upon a time"

In the training pipeline:
    from bpe_tokenization import BPETokenizer
    tok = BPETokenizer.from_pretrained()
    ids = tok.encode(text)          # list[int]
    text = tok.decode(ids)          # str
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from collections import Counter
from heapq import heapify, heappop, heappush
from pathlib import Path


# --------------------------------------------------------------------------- #
# regex bootstrap: the stdlib `re` has no \p{L} / \p{N}, and a tokenizer that
# silently falls back to a different split pattern would be a footgun (the
# trained merges would no longer match the encoder). So: hard dependency.
# --------------------------------------------------------------------------- #
def ensure_regex():
    try:
        import regex  # noqa: F401
    except ImportError:
        print("[setup] regex not found -> pip installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "regex"])
    import regex
    return regex


regex = ensure_regex()


# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATASET_DIR = REPO_ROOT / "Datasets"

TRAIN_TXT = DATASET_DIR / "tinystories_instruct_train.txt"
VAL_TXT = DATASET_DIR / "tinystories_instruct_val.txt"

# The GPT-4 (cl100k_base) split pattern. Strictly better than the GPT-2 one for
# our corpus: it caps digit runs at 3, keeps punctuation runs together, and
# never glues a leading space onto a newline. Possessive quantifiers (`++`,
# `?+`) need the `regex` module -- they stop catastrophic backtracking on long
# punctuation runs ("!!!!!!!!!!" shows up a lot in kids' stories).
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)"""        # contractions: 's 't 'm 'd 'll 've 're
    r"""|[^\r\n\p{L}\p{N}]?+\p{L}+"""   # a word, with its leading space if any
    r"""|\p{N}{1,3}"""                  # digits, in runs of at most 3
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*"""  # punctuation run + trailing newlines
    r"""|\s*[\r\n]"""                   # a newline with its leading blanks
    r"""|\s+(?!\S)"""                   # trailing whitespace before a newline
    r"""|\s+"""                         # any other whitespace run
)

# Corpus markers written by build_tinystories_instruct_txt.py, plus the two
# tokens the transformer needs at training time. Order fixes their ids.
INSTRUCTION_START = "<|startofinstruction|>"
INSTRUCTION_END = "<|endofinstruction|>"
STORY_START = "<|startofstory|>"
STORY_END = "<|endofstory|>"

SPECIAL_TOKENS = [
    "<|endoftext|>",            # document separator / generation stop
    "<|pad|>",                  # right-padding for ragged batches
    INSTRUCTION_START,
    INSTRUCTION_END,
    STORY_START,
    STORY_END,
]

DEFAULT_VOCAB_SIZE = 8192       # 256 bytes + 7930 merges + 6 specials
TOKENIZER_PATH = HERE / "tinystories_bpe_8k.json"   # .vocab dump sits next to it

CHUNKS_PER_CORE = 4             # a few chunks per core -> load balancing
N_CORES = os.cpu_count() or 4

# Overwrite the progress line in a terminal, but keep one line per update when
# stdout is redirected to a log file (otherwise the whole run is one long line).
PROGRESS_END = "\r" if sys.stdout.isatty() else "\n"

# Split on specials without consuming them (capturing group keeps them in the
# output list). Longest-first so <|startofstory|> can't be eaten by a prefix.
_SPECIAL_SPLIT_RE = re.compile(
    "(" + "|".join(re.escape(s) for s in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")"
)
_ESCAPE_RE = re.compile(r"\\(.)")
_PAT = regex.compile(SPLIT_PATTERN)
_SPECIAL_SET = set(SPECIAL_TOKENS)


# --------------------------------------------------------------------------- #
# corpus helpers
# --------------------------------------------------------------------------- #
def unescape_story(text: str) -> str:
    r"""Undo the `\n` / `\\` escaping build_tinystories_instruct_txt.py applied.

    That script keeps one record on one line by writing stories with literal
    backslash-n. The tokenizer must see the real newline, otherwise it learns a
    "\n" two-character token that the model would emit forever.
    """
    return _ESCAPE_RE.sub(lambda m: "\n" if m.group(1) == "n" else m.group(1), text)


def iter_segments(text: str, unescape: bool = True):
    """Yield (segment, is_special) pairs, unescaping only the story bodies.

    Instructions were never escaped by the builder, so unescaping them would
    corrupt any literal backslash they happen to contain.
    """
    prev_special = None
    for seg in _SPECIAL_SPLIT_RE.split(text):
        if not seg:
            continue
        if seg in _SPECIAL_SET:
            prev_special = seg
            yield seg, True
        else:
            if unescape and prev_special == STORY_START:
                seg = unescape_story(seg)
            yield seg, False


def prepare_line(line: str, unescape: bool = True) -> str:
    """One raw corpus line -> the exact string to feed `encode()`."""
    return "".join(seg for seg, _ in iter_segments(line, unescape))


def iter_chunk_lines(path, start, end):
    """Yield the full lines whose *start* offset falls inside [start, end)."""
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            f.readline()            # discard partial line, align to next full one
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield line


def parse_size(text):
    """'100M' -> 100_000_000; '2G' -> 2_000_000_000; '5000' -> 5000."""
    if text is None:
        return None
    text = str(text).strip().upper()
    mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000, "B": 1_000_000_000}
    if text and text[-1] in mult:
        return int(float(text[:-1]) * mult[text[-1]])
    return int(text)


# --------------------------------------------------------------------------- #
# pass 1: count pre-tokens (parallel)
# --------------------------------------------------------------------------- #
def count_chunk(args):
    """Worker: pre-tokenize one byte range and return a {bytes: count} Counter.

    Returning frequencies instead of the pre-tokens themselves is what makes
    the whole thing cheap -- 1.1 GB of stories collapses to ~100k distinct
    pre-tokens, and BPE only ever needs the distinct ones plus their counts.
    """
    path, start, end, unescape = args
    counts = Counter()
    n_bytes = 0
    n_specials = 0
    findall = _PAT.findall

    for raw in iter_chunk_lines(path, start, end):
        line = raw.decode("utf-8", errors="replace")
        for seg, is_special in iter_segments(line, unescape):
            if is_special:
                n_specials += 1
                continue
            n_bytes += len(seg.encode("utf-8"))
            for piece in findall(seg):
                counts[piece.encode("utf-8")] += 1

    return counts, n_bytes, n_specials


def count_pretokens(path, workers, max_bytes=None, unescape=True):
    """Pre-tokenize `path` across `workers` processes -> merged frequency table."""
    size = path.stat().st_size
    if max_bytes is not None:
        size = min(size, max_bytes)

    n_chunks = max(1, workers * CHUNKS_PER_CORE)
    chunk = size // n_chunks + 1
    tasks = []
    for c in range(n_chunks):
        start = c * chunk
        end = min(start + chunk, size)
        if start < end:
            tasks.append((path, start, end, unescape))

    word_freqs = Counter()
    total_bytes = total_specials = 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (counts, n_bytes, n_specials) in enumerate(
            pool.imap_unordered(count_chunk, tasks), 1
        ):
            word_freqs.update(counts)
            total_bytes += n_bytes
            total_specials += n_specials
            done = i / len(tasks)
            print(
                f"  [pretokenize] {i:>3}/{len(tasks)} chunks  "
                f"{done:6.1%}  {total_bytes / 1e6:8.1f} MB  "
                f"{len(word_freqs):>7,} unique  {time.time() - t0:6.1f}s",
                end=PROGRESS_END,
                flush=True,
            )
    print()
    return word_freqs, total_bytes, total_specials, time.time() - t0


# --------------------------------------------------------------------------- #
# pass 2: learn the merges (sequential, but incremental)
# --------------------------------------------------------------------------- #
def apply_merge(word, a, b, new_id):
    """Replace every non-overlapping [a, b] in `word` with `new_id`."""
    out = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def learn_merges(word_freqs, n_merges, log_every=250):
    """Classic BPE over a pre-token frequency table.

    The naive version recounts every pair after every merge -- 8k merges x
    100k words is hopeless in Python. Instead we keep two live indexes:

        pair_counts[pair]  -> how often the pair occurs in the whole corpus
        where[pair]        -> which words contain it

    so a merge only touches the words that actually contain the merged pair,
    and a lazy max-heap hands us the winner without rescanning. Stale heap
    entries are detected by comparing against pair_counts (standard trick --
    cheaper than a decrease-key structure and about as fast).
    """
    words = [list(w) for w in word_freqs]          # each word: list of token ids
    freqs = list(word_freqs.values())

    pair_counts = Counter()
    where = {}
    for i, w in enumerate(words):
        f = freqs[i]
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            where.setdefault(pair, set()).add(i)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapify(heap)

    merges = {}                                     # (a, b) -> new id
    vocab = {i: bytes([i]) for i in range(256)}
    history = []                                    # (new_id, pair, count) per merge
    next_id = 256
    t0 = time.time()

    while len(merges) < n_merges and heap:
        neg_count, pair = heappop(heap)
        count = -neg_count
        if pair_counts.get(pair, 0) != count:
            continue                                # stale entry, a newer one exists
        if count < 2:
            break                                   # nothing repeats any more

        new_id = next_id
        next_id += 1
        a, b = pair
        merges[pair] = new_id
        vocab[new_id] = vocab[a] + vocab[b]
        history.append((new_id, pair, count))

        for i in tuple(where[pair]):
            old = words[i]
            new = apply_merge(old, a, b, new_id)
            f = freqs[i]
            old_pairs = Counter(zip(old, old[1:]))
            new_pairs = Counter(zip(new, new[1:]))
            for p in old_pairs.keys() | new_pairs.keys():
                delta = (new_pairs[p] - old_pairs[p]) * f
                if delta:
                    pair_counts[p] += delta
                    heappush(heap, (-pair_counts[p], p))
                if new_pairs[p] == 0:
                    where[p].discard(i)
                elif old_pairs[p] == 0:
                    where.setdefault(p, set()).add(i)
            words[i] = new

        where.pop(pair, None)
        pair_counts.pop(pair, None)

        n = len(merges)
        if n % log_every == 0 or n == n_merges:
            print(
                f"  [merge] {n:>5}/{n_merges}  {n / n_merges:6.1%}  "
                f"last={render(vocab[new_id]):<18} count={count:>9,}  "
                f"{time.time() - t0:6.1f}s",
                end=PROGRESS_END,
                flush=True,
            )
    print()
    return merges, vocab, history, time.time() - t0


# --------------------------------------------------------------------------- #
# the tokenizer
# --------------------------------------------------------------------------- #
def render(token_bytes):
    """Printable form of a token, GPT-2 style: Ġ = space, Ċ = newline."""
    s = token_bytes.decode("utf-8", errors="replace")
    return s.replace(" ", "Ġ").replace("\n", "Ċ").replace("\t", "ĉ").replace("\r", "Ř")


class BPETokenizer:
    """Byte-level BPE with GPT-4 style pre-tokenization and special tokens."""

    def __init__(self, merges, special_tokens, pattern=SPLIT_PATTERN, meta=None):
        self.pattern = pattern
        self.merges = merges                        # (int, int) -> int
        self.special_tokens = special_tokens        # str -> int
        self.meta = meta or {}

        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        for tok, idx in special_tokens.items():
            self.vocab[idx] = tok.encode("utf-8")

        self._compile()

    # -- pickling: compiled regexes and the cache don't travel to workers ---- #
    def _compile(self):
        self._pat = regex.compile(self.pattern)
        self._special_re = (
            re.compile(
                "("
                + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
                + ")"
            )
            if self.special_tokens
            else None
        )
        self._cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ("_pat", "_special_re", "_cache"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._compile()

    # -- properties --------------------------------------------------------- #
    @property
    def n_vocab(self):
        return len(self.vocab)

    @property
    def eot_id(self):
        return self.special_tokens["<|endoftext|>"]

    @property
    def pad_id(self):
        return self.special_tokens["<|pad|>"]

    # -- encode ------------------------------------------------------------- #
    def _encode_piece(self, piece: bytes):
        """BPE one pre-token: repeatedly apply the lowest-ranked merge.

        Merge rank == the new token's id, because ids are handed out in merge
        order, so "earliest learned merge wins" is just "smallest id wins".
        """
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        ids = list(piece)
        while len(ids) >= 2:
            best_pair, best_id = None, None
            for pair in zip(ids, ids[1:]):
                idx = self.merges.get(pair)
                if idx is not None and (best_id is None or idx < best_id):
                    best_pair, best_id = pair, idx
            if best_pair is None:
                break
            ids = apply_merge(ids, best_pair[0], best_pair[1], best_id)

        if len(piece) <= 64:                        # bound the cache on junk input
            self._cache[piece] = ids
        return ids

    def encode_ordinary(self, text: str):
        """Encode text, treating any `<|...|>` marker as literal characters."""
        ids = []
        for piece in self._pat.findall(text):
            ids.extend(self._encode_piece(piece.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special="all"):
        """Encode text -> list[int].

        allowed_special:
            "all"  (default) every special token in `text` becomes its id
            "none"            specials are encoded as ordinary characters
            set of str        only these are honoured; the rest stay literal
        """
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none":
            allowed = set()
        else:
            allowed = set(allowed_special)

        if not allowed or self._special_re is None:
            return self.encode_ordinary(text)

        ids = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            if chunk in allowed:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def encode_batch(self, texts, workers=None, allowed_special="all"):
        """Encode many texts across processes (BPE is pure Python -> GIL-bound).

        The caller must be inside `if __name__ == "__main__":` -- macOS/Windows
        spawn a fresh interpreter per worker and re-import the calling module,
        so an unguarded call re-executes the caller in every worker.
        """
        workers = workers or N_CORES
        if workers == 1 or len(texts) < 64:
            return [self.encode(t, allowed_special) for t in texts]

        n_chunks = max(1, workers * CHUNKS_PER_CORE)
        size = max(1, len(texts) // n_chunks + 1)
        batches = [texts[i:i + size] for i in range(0, len(texts), size)]
        with mp.Pool(workers, initializer=_worker_init, initargs=(self, allowed_special)) as pool:
            out = pool.map(_worker_encode, batches)
        return [ids for batch in out for ids in batch]

    # -- decode ------------------------------------------------------------- #
    def decode_bytes(self, ids) -> bytes:
        vocab = self.vocab
        return b"".join(vocab[i] for i in ids)

    def decode(self, ids) -> str:
        """ids -> text. `errors="replace"` because a model can emit a token
        sequence that cuts a multi-byte character in half."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # -- persistence -------------------------------------------------------- #
    def save(self, path=TOKENIZER_PATH, dump_vocab=True):
        path = Path(path)
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        payload = {
            "version": 1,
            "pattern": self.pattern,
            "vocab_size": self.n_vocab,
            "merges": [[a, b] for (a, b), _ in ordered],
            "special_tokens": self.special_tokens,
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

        if dump_vocab:
            lines = [f"{i}\t{render(self.vocab[i])}" for i in range(self.n_vocab)]
            path.with_suffix(".vocab").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=TOKENIZER_PATH):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): 256 + i for i, (a, b) in enumerate(payload["merges"])}
        specials = {k: int(v) for k, v in payload["special_tokens"].items()}
        return cls(merges, specials, payload["pattern"], payload.get("meta", {}))

    @classmethod
    def from_pretrained(cls, path=TOKENIZER_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Train it first: python {Path(__file__).name} train"
            )
        return cls.load(path)


# Pool globals: the tokenizer is built once per worker, not once per batch.
_WORKER_TOK = None
_WORKER_ALLOWED = "all"


def _worker_init(tokenizer, allowed_special):
    global _WORKER_TOK, _WORKER_ALLOWED
    _WORKER_TOK, _WORKER_ALLOWED = tokenizer, allowed_special


def _worker_encode(texts):
    return [_WORKER_TOK.encode(t, _WORKER_ALLOWED) for t in texts]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(corpus_path, vocab_size=DEFAULT_VOCAB_SIZE, workers=N_CORES,
          max_bytes=None, unescape=True):
    """Full training run: count pre-tokens, learn merges, build the tokenizer."""
    n_specials = len(SPECIAL_TOKENS)
    n_merges = vocab_size - 256 - n_specials
    if n_merges < 1:
        raise ValueError(f"vocab_size must exceed {256 + n_specials}")

    size = corpus_path.stat().st_size
    used = min(size, max_bytes) if max_bytes else size
    print(f"\ncorpus     : {corpus_path}")
    print(f"             {size / 1e9:.2f} GB on disk, using {used / 1e9:.2f} GB")
    print(f"vocab      : {vocab_size} = 256 bytes + {n_merges} merges + {n_specials} specials")
    print(f"workers    : {workers} processes on {N_CORES} cores\n")

    word_freqs, corpus_bytes, n_special_seen, t_count = count_pretokens(
        corpus_path, workers, max_bytes, unescape
    )
    n_pretokens = sum(word_freqs.values())
    print(f"  -> {n_pretokens:,} pre-tokens, {len(word_freqs):,} unique, "
          f"{t_count:.1f}s ({corpus_bytes / 1e6 / t_count:,.0f} MB/s)\n")

    merges, vocab, history, t_merge = learn_merges(word_freqs, n_merges)
    print(f"  -> {len(merges):,} merges in {t_merge:.1f}s "
          f"({len(merges) / max(t_merge, 1e-9):,.0f} merges/s)\n")

    specials = {tok: 256 + len(merges) + i for i, tok in enumerate(SPECIAL_TOKENS)}
    meta = {
        "corpus": str(corpus_path),
        "corpus_bytes": corpus_bytes,
        "corpus_pretokens": n_pretokens,
        "corpus_unique_pretokens": len(word_freqs),
        "corpus_special_markers": n_special_seen,
        "n_merges": len(merges),
        "seconds_pretokenize": round(t_count, 2),
        "seconds_merge": round(t_merge, 2),
        "workers": workers,
        "unescape": unescape,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # top merges, kept for the `stats` subcommand
        "first_merges": [[i, render(vocab[i]), c] for i, _, c in history[:40]],
    }
    return BPETokenizer(merges, specials, SPLIT_PATTERN, meta), word_freqs, history


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def hr(title=""):
    if title:
        print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")
    else:
        print("─" * 78)


def bar(frac, width=32):
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def evaluate(tok, path, n_records, workers):
    """Encode a slice of a split and measure how well the vocab compresses it."""
    if not path.exists():
        return None

    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines.append(prepare_line(line.rstrip("\n")))
            if len(lines) >= n_records:
                break
    if not lines:
        return None

    t0 = time.time()
    batches = tok.encode_batch(lines, workers=workers)
    dt = time.time() - t0

    n_tokens = sum(len(b) for b in batches)
    n_chars = sum(len(t) for t in lines)
    n_bytes = sum(len(t.encode("utf-8")) for t in lines)
    freq = Counter()
    for b in batches:
        freq.update(b)

    # round-trip: decode must reproduce the input exactly, character for character
    bad = next((i for i, (t, b) in enumerate(zip(lines, batches)) if tok.decode(b) != t), None)

    return {
        "records": len(lines),
        "chars": n_chars,
        "bytes": n_bytes,
        "tokens": n_tokens,
        "bytes_per_token": n_bytes / n_tokens,
        "chars_per_token": n_chars / n_tokens,
        "tokens_per_record": n_tokens / len(lines),
        "seconds": dt,
        "tokens_per_sec": n_tokens / max(dt, 1e-9),
        "coverage": len(freq) / tok.n_vocab,
        "unused": tok.n_vocab - len(freq),
        "freq": freq,
        "roundtrip_ok": bad is None,
        "roundtrip_bad_index": bad,
    }


def print_report(tok, history=None, word_freqs=None, eval_stats=None, sample_text=None,
                 out_path=TOKENIZER_PATH):
    m = tok.meta
    out_path = Path(out_path)
    hr("1. TOKENIZER")
    print(f"  file            : {out_path}")
    print(f"  vocab dump      : {out_path.with_suffix('.vocab')}")
    print(f"  vocab size      : {tok.n_vocab:,}")
    print(f"  base bytes      : 256")
    print(f"  learned merges  : {m.get('n_merges', len(tok.merges)):,}")
    print(f"  special tokens  : {len(tok.special_tokens)}")
    print(f"  fits uint16     : {'yes' if tok.n_vocab < 2 ** 16 else 'NO'}")
    print(f"  trained at      : {m.get('trained_at', '?')}")
    print(f"  split pattern   : {tok.pattern}")

    hr("2. TRAINING CORPUS")
    cb = m.get("corpus_bytes", 0)
    pt = m.get("corpus_pretokens", 0)
    print(f"  corpus          : {m.get('corpus', '?')}")
    print(f"  text bytes seen : {cb:,} ({cb / 1e9:.2f} GB, specials excluded)")
    print(f"  markers seen    : {m.get('corpus_special_markers', 0):,}")
    print(f"  pre-tokens      : {pt:,}")
    print(f"  unique          : {m.get('corpus_unique_pretokens', 0):,}")
    if pt:
        print(f"  bytes/pre-token : {cb / pt:.2f}")
    print(f"  pretokenize     : {m.get('seconds_pretokenize', 0):,.1f}s "
          f"on {m.get('workers', '?')} processes "
          f"({cb / 1e6 / max(m.get('seconds_pretokenize', 1), 1e-9):,.0f} MB/s)")
    print(f"  merge loop      : {m.get('seconds_merge', 0):,.1f}s")
    print(f"  total           : {m.get('seconds_pretokenize', 0) + m.get('seconds_merge', 0):,.1f}s")

    hr("3. SPECIAL TOKENS")
    for name, idx in sorted(tok.special_tokens.items(), key=lambda kv: kv[1]):
        print(f"  {idx:>6}  {name}")

    hr("4. TOKEN LENGTH DISTRIBUTION (bytes per token)")
    lengths = Counter(len(v) for v in tok.vocab.values())
    total = sum(lengths.values())
    for ln in sorted(lengths):
        n = lengths[ln]
        print(f"  {ln:>3} bytes  {n:>6,}  {n / total:6.2%}  {bar(n / total)}")
    avg = sum(ln * n for ln, n in lengths.items()) / total
    print(f"  mean token length: {avg:.2f} bytes")

    hr("5. FIRST 40 MERGES (the highest-frequency pairs in the corpus)")
    firsts = ([[i, render(tok.vocab[i]), c] for i, _, c in history[:40]]
              if history else m.get("first_merges", []))
    for row in range(0, min(40, len(firsts)), 4):
        cells = []
        for idx, text, count in firsts[row:row + 4]:
            cells.append(f"{idx:>5} {text:<10} {count:>9,}")
        print("  " + "  ".join(cells))

    hr("6. LONGEST TOKENS LEARNED")
    longest = sorted(tok.vocab.items(), key=lambda kv: -len(kv[1]))[:24]
    for row in range(0, len(longest), 4):
        print("  " + "  ".join(f"{i:>5} {render(v):<14}" for i, v in longest[row:row + 4]))

    if word_freqs:
        hr("7. MOST FREQUENT PRE-TOKENS IN THE CORPUS")
        top = word_freqs.most_common(24)
        for row in range(0, len(top), 4):
            print("  " + "  ".join(f"{render(w):<12}{c:>11,}" for w, c in top[row:row + 4]))

    if eval_stats:
        e = eval_stats
        hr("8. COMPRESSION ON THE VALIDATION SPLIT")
        print(f"  records encoded : {e['records']:,}")
        print(f"  characters      : {e['chars']:,}")
        print(f"  utf-8 bytes     : {e['bytes']:,}")
        print(f"  tokens          : {e['tokens']:,}")
        print(f"  bytes / token   : {e['bytes_per_token']:.3f}   <- compression ratio")
        print(f"  chars / token   : {e['chars_per_token']:.3f}")
        print(f"  tokens / record : {e['tokens_per_record']:.1f}")
        print(f"  encode speed    : {e['tokens_per_sec']:,.0f} tok/s "
              f"({e['seconds']:.1f}s, {N_CORES} processes)")
        print(f"  vocab coverage  : {e['coverage']:.2%}  ({e['unused']:,} tokens never used)")
        verdict = ("PASS - decode(encode(x)) == x for every record"
                   if e["roundtrip_ok"] else f"FAIL at record {e['roundtrip_bad_index']}")
        print(f"  round-trip      : {verdict}")

        hr("9. MOST FREQUENT TOKENS AT ENCODE TIME")
        top = e["freq"].most_common(24)
        for row in range(0, len(top), 4):
            cells = [f"{render(tok.vocab[i]):<12}{c:>9,}" for i, c in top[row:row + 4]]
            print("  " + "  ".join(cells))

    if sample_text:
        hr("10. SAMPLE ENCODE / DECODE")
        ids = tok.encode(sample_text)
        preview = sample_text if len(sample_text) <= 220 else sample_text[:220] + " ..."
        print(f"  text   ({len(sample_text)} chars): {preview!r}")
        print(f"  ids    ({len(ids)} tokens): {ids[:48]}{' ...' if len(ids) > 48 else ''}")
        print(f"  pieces : {' | '.join(render(tok.vocab[i]) for i in ids[:48])}"
              f"{' ...' if len(ids) > 48 else ''}")
        ok = tok.decode(ids) == sample_text
        print(f"  decode : {'exact round-trip' if ok else 'MISMATCH'}")
    hr()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
SAMPLE = (
    f'{INSTRUCTION_START}Write a story that includes the words quit, oak, gloomy.'
    f'{INSTRUCTION_END}{STORY_START}Sara and Ben were playing in the park.\n'
    f'"Ben, I want to quit," she said. It was 3 o\'clock and the sky looked gloomy!'
    f'{STORY_END}'
)


def cmd_train(args):
    corpus = Path(args.corpus)
    if not corpus.exists():
        raise FileNotFoundError(
            f"{corpus} not found. Run experiments/datasets/build_tinystories_instruct_txt.py first."
        )

    tok, word_freqs, history = train(
        corpus,
        vocab_size=args.vocab_size,
        workers=args.workers,
        max_bytes=parse_size(args.max_bytes),
        unescape=not args.no_unescape,
    )
    tok.save(args.out)
    print(f"saved -> {args.out}\n")

    print(f"evaluating on {args.eval_records:,} records of {VAL_TXT.name} ...")
    eval_stats = evaluate(tok, Path(args.val), args.eval_records, args.workers)
    print_report(tok, history=history, word_freqs=word_freqs,
                 eval_stats=eval_stats, sample_text=SAMPLE, out_path=args.out)


def cmd_stats(args):
    tok = BPETokenizer.from_pretrained(args.out)
    eval_stats = evaluate(tok, Path(args.val), args.eval_records, args.workers)
    print_report(tok, eval_stats=eval_stats, sample_text=SAMPLE, out_path=args.out)


def cmd_demo(args):
    tok = BPETokenizer.from_pretrained(args.out)
    text = args.text or SAMPLE
    ids = tok.encode(text)
    print(f"text   : {text!r}")
    print(f"tokens : {len(ids)}  ({len(text.encode()) / len(ids):.2f} bytes/token)")
    print(f"ids    : {ids}")
    print(f"pieces : {' | '.join(render(tok.vocab[i]) for i in ids)}")
    print(f"decode : {tok.decode(ids)!r}")
    print(f"exact  : {tok.decode(ids) == text}")


def main():
    # shared flags, declared on a parent so they work *after* the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(TOKENIZER_PATH), help="tokenizer json path")
    common.add_argument("--val", default=str(VAL_TXT), help="validation .txt for the report")
    common.add_argument("--workers", type=int, default=N_CORES, help="worker processes")
    common.add_argument("--eval-records", type=int, default=5000,
                        help="records of the val split used for the compression report")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], parents=[common])
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", parents=[common], help="train the tokenizer on the corpus")
    p_train.add_argument("--corpus", default=str(TRAIN_TXT))
    p_train.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                         help=f"total vocab incl. specials (default {DEFAULT_VOCAB_SIZE})")
    p_train.add_argument("--max-bytes", default=None,
                         help="only train on the first N bytes, e.g. 100M (default: all)")
    p_train.add_argument("--no-unescape", action="store_true",
                         help=r"keep stories' literal \n instead of real newlines")
    p_train.set_defaults(func=cmd_train)

    p_stats = sub.add_parser("stats", parents=[common],
                             help="reprint the report for a saved tokenizer")
    p_stats.set_defaults(func=cmd_stats)

    p_demo = sub.add_parser("demo", parents=[common], help="encode/decode one string")
    p_demo.add_argument("--text", default=None)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if not getattr(args, "cmd", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
