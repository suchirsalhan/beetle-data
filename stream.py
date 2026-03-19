"""
stream_tokens.py
----------------
Stream exactly 33.33M tokens from kidlm.txt using a sliding buffer.

Tokenisation strategy: whitespace-split (fast, no dependencies).
Swap in a HuggingFace / tiktoken tokeniser if you need subword tokens.

Usage
-----
# Basic — whitespace tokeniser (no deps, fastest)
python stream.py

# Explicit options
python stream.py --file kidlm.txt --target 33_330_000 --chunk 65536

# Subword tokens via tiktoken (needs: pip install tiktoken)
python stream.py --tokenizer tiktoken

# Subword tokens via HuggingFace (needs: pip install transformers)
python stream.py --tokenizer hf
"""

import argparse
import sys
import time
from pathlib import Path


# ── optional tokenisers ────────────────────────────────────────────────────────

def build_tokenizer(name: str):
    if name == "whitespace":
        # returns word count for a string, used only for counting
        return lambda text: len(text.split())

    if name == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text))
        except ImportError:
            sys.exit("Install tiktoken first:  pip install tiktoken")

    if name == "hf":
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except ImportError:
            sys.exit("Install transformers first:  pip install transformers")

    sys.exit(f"Unknown tokeniser '{name}'. Choose: whitespace | tiktoken | hf")


# ── helpers ────────────────────────────────────────────────────────────────────

def words_in_line(line: str) -> int:
    return len(line.split())


def truncate_line_to_tokens(line: str, remaining: int) -> str:
    """Return the first *remaining* whitespace tokens of *line*, preserving
    any trailing newline so the cut point is clean."""
    words = line.split(" ")
    truncated = " ".join(words[:remaining])
    # preserve newline if original line ended with one
    if line.endswith("\n"):
        truncated += "\n"
    return truncated


# ── main run ───────────────────────────────────────────────────────────────────

def run(args):
    path        = Path(args.file)
    target      = args.target
    output_path = Path(args.output)
    count_tokens = build_tokenizer(args.tokenizer)

    if not path.exists():
        sys.exit(f"File not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_passes  = 0
    total_tokens = 0
    t0 = t_last  = time.perf_counter()
    report_every = max(1, target // 100)
    tokens_since_last = 0

    print(f"Streaming {target:,} tokens from '{path}'")
    print(f"Saving output to '{output_path}'")
    print(f"Tokeniser: {args.tokenizer}  |  Looping file if needed\n")

    with output_path.open("w", encoding="utf-8") as out_fh:
        done = False
        while not done:
            file_passes += 1
            with path.open("r", encoding="utf-8", errors="replace") as in_fh:
                for line in in_fh:
                    line_tokens = count_tokens(line)

                    if total_tokens + line_tokens >= target:
                        # this line crosses the cutoff — write only what fits
                        remaining = target - total_tokens
                        out_fh.write(truncate_line_to_tokens(line, remaining))
                        total_tokens = target
                        done = True
                        break

                    out_fh.write(line)
                    total_tokens      += line_tokens
                    tokens_since_last += line_tokens

                    if total_tokens // report_every > (total_tokens - line_tokens) // report_every:
                        now     = time.perf_counter()
                        rate    = tokens_since_last / max(1e-9, now - t_last)
                        pct     = 100 * total_tokens / target
                        elapsed = now - t0
                        eta     = (target - total_tokens) / max(1, rate)
                        print(f"  {pct:6.2f}%  {total_tokens:>14,} / {target:,} tokens  "
                              f"  {rate:,.0f} tok/s  "
                              f"  elapsed {elapsed:,.1f}s  ETA {eta:,.1f}s",
                              flush=True)
                        t_last = now
                        tokens_since_last = 0

    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\nDone. {total_tokens:,} tokens written in {elapsed:.2f}s "
          f"({total_tokens/elapsed:,.0f} tok/s avg).")
    print(f"File passes: {file_passes}  |  Output: {output_path}  ({size_mb:.1f} MB)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stream 33.33M tokens preserving original text structure.")
    p.add_argument("--file",      default="kidlm.txt")
    p.add_argument("--target",    type=int, default=33_330_000)
    p.add_argument("--tokenizer", default="whitespace",
                   choices=["whitespace", "tiktoken", "hf"])
    p.add_argument("--output",    default="./stream/output.txt")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
