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
import itertools
import sys
import time
from pathlib import Path


# ── optional tokenisers ────────────────────────────────────────────────────────

def build_tokenizer(name: str):
    if name == "whitespace":
        return str.split

    if name == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return enc.encode
        except ImportError:
            sys.exit("Install tiktoken first:  pip install tiktoken")

    if name == "hf":
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            return lambda text: tok.encode(text, add_special_tokens=False)
        except ImportError:
            sys.exit("Install transformers first:  pip install transformers")

    sys.exit(f"Unknown tokeniser '{name}'. Choose: whitespace | tiktoken | hf")


# ── core streaming generator ───────────────────────────────────────────────────

def token_stream(path: Path, tokenize, chunk_bytes: int = 65_536):
    leftover = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            raw = fh.read(chunk_bytes)
            if not raw:
                break
            block = leftover + raw
            cut = block.rfind(" ")
            if cut == -1:
                leftover = block
                continue
            chunk_text = block[:cut]
            leftover   = block[cut + 1:]
            yield from tokenize(chunk_text)

        if leftover.strip():
            yield from tokenize(leftover)


def stream_n_tokens(path: Path, target: int, tokenize, chunk_bytes: int = 65_536):
    yielded = 0
    for _ in itertools.cycle([None]):
        for tok in token_stream(path, tokenize, chunk_bytes):
            yield tok
            yielded += 1
            if yielded >= target:
                return


# ── main run ───────────────────────────────────────────────────────────────────

def run(args):
    path        = Path(args.file)
    target      = args.target
    tok_fn      = build_tokenizer(args.tokenizer)
    output_path = Path(args.output)

    if not path.exists():
        sys.exit(f"File not found: {path}")

    # ensure ./stream/ directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {target:,} tokens from '{path}'")
    print(f"Saving output to '{output_path}'")
    print(f"Tokeniser: {args.tokenizer}  |  Chunk: {args.chunk:,} bytes\n")

    report_every      = max(1, target // 100)
    write_buffer      = []
    write_buffer_size = 100_000
    t0 = t_last       = time.perf_counter()
    count             = 0
    tokens_since_last = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for tok in stream_n_tokens(path, target, tok_fn, args.chunk):
            write_buffer.append(str(tok))
            count += 1
            tokens_since_last += 1

            if len(write_buffer) >= write_buffer_size:
                out_fh.write("\n".join(write_buffer) + "\n")
                write_buffer.clear()

            if count % report_every == 0 or count == target:
                now     = time.perf_counter()
                rate    = tokens_since_last / max(1e-9, now - t_last)
                pct     = 100 * count / target
                elapsed = now - t0
                eta     = (target - count) / max(1, rate)
                print(f"  {pct:6.2f}%  {count:>14,} / {target:,} tokens  "
                      f"  {rate:,.0f} tok/s  "
                      f"  elapsed {elapsed:,.1f}s  ETA {eta:,.1f}s",
                      flush=True)
                t_last = now
                tokens_since_last = 0

        if write_buffer:
            out_fh.write("\n".join(write_buffer) + "\n")

    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\nDone. Streamed {count:,} tokens in {elapsed:.2f}s "
          f"({count/elapsed:,.0f} tok/s avg).")
    print(f"Output saved to: {output_path}  ({size_mb:.1f} MB)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stream 33.33M tokens from a text file.")
    p.add_argument("--file",      default="kidlm.txt",
                   help="Path to source text file (default: kidlm.txt)")
    p.add_argument("--target",    type=int, default=33_330_000,
                   help="Number of tokens to stream (default: 33_330_000)")
    p.add_argument("--chunk",     type=int, default=65_536,
                   help="Read-buffer size in bytes (default: 65536)")
    p.add_argument("--tokenizer", default="whitespace",
                   choices=["whitespace", "tiktoken", "hf"],
                   help="Tokenisation strategy (default: whitespace)")
    p.add_argument("--output",    default="./stream/output.txt",
                   help="Path to save streamed tokens (default: ./stream/output.txt)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
