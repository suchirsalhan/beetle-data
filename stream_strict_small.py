"""
make_strict_small.py
--------------------
1. Downloads all .txt files from the BabyLM-2026-Strict dataset on HuggingFace.
2. Saves raw files into ./strict/
3. Counts words in every present file, then extracts a proportional slice from
   each so that the *total* across all files is exactly --total-words (default
   10 000 000).  Output goes to ./strict-small/

Usage
-----
# Install dependency first (if needed):
#   pip install huggingface_hub

python make_strict_small.py

# Skip re-downloading files already present in ./strict/:
python make_strict_small.py --skip-download

# Custom total (default 10_000_000):
python make_strict_small.py --total-words 10000000
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ID = "BabyLM-community/BabyLM-2026-Strict"
REPO_TYPE = "dataset"

FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "open_subtitles_cleaned_ckpt_70.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]

# ── download ───────────────────────────────────────────────────────────────────

def download_files(strict_dir: Path, skip_existing: bool):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required.\n"
            "Install it with:  pip install huggingface_hub"
        )

    strict_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading files to '{strict_dir}/'  (repo: {REPO_ID})\n")

    for filename in FILES:
        dest = strict_dir / filename
        if skip_existing and dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [skip]  {filename}  ({size_mb:.1f} MB already present)")
            continue

        print(f"  Downloading {filename} ...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=str(strict_dir),
                local_dir_use_symlinks=False,
            )
            elapsed = time.perf_counter() - t0
            size_mb = Path(local_path).stat().st_size / 1_048_576
            print(f"  done  ({size_mb:.1f} MB, {elapsed:.1f}s)")
        except Exception as exc:
            print(f"\n  WARNING: Could not download {filename}: {exc}")

    print()


# ── count words in a file (fast pass) ─────────────────────────────────────────

def count_file_words(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += len(line.split())
    return total


# ── extract exactly `target` words ────────────────────────────────────────────

def extract_words(src: Path, dst: Path, target: int) -> dict:
    """
    Stream *src*, writing lines to *dst* until exactly *target* words have been
    written.  The last partial line is trimmed to the word boundary.
    Returns a stats dict.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.perf_counter()

    with src.open("r", encoding="utf-8", errors="replace") as in_fh, \
         dst.open("w", encoding="utf-8") as out_fh:

        for line in in_fh:
            words = line.split()
            n = len(words)

            if written + n >= target:
                remaining = target - written
                out_fh.write(" ".join(words[:remaining]))
                out_fh.write("\n")
                written += remaining
                break

            out_fh.write(line)
            written += n

    elapsed = time.perf_counter() - t0
    size_mb = dst.stat().st_size / 1_048_576
    print(f"    → wrote {written:,} words  ({size_mb:.1f} MB)  in {elapsed:.1f}s")
    return {"file": src.name, "written_words": written}


# ── main ───────────────────────────────────────────────────────────────────────

def run(args):
    strict_dir = Path("./strict")
    small_dir  = Path("./strict-small")
    total_target: int = args.total_words

    # Step 1: download
    if not args.skip_download:
        download_files(strict_dir, skip_existing=False)
    else:
        print("Skipping download (--skip-download set).\n")

    # Step 2: discover which files are present
    present = [(fn, strict_dir / fn) for fn in FILES if (strict_dir / fn).exists()]
    missing = [fn for fn in FILES if not (strict_dir / fn).exists()]
    if missing:
        for fn in missing:
            print(f"  [missing]  {strict_dir / fn}  — skipping")
        print()

    # Step 3: count words in every present file
    print("Counting source words …\n")
    source_counts: dict[str, int] = {}
    for fn, path in present:
        print(f"  {fn} ...", end="", flush=True)
        wc = count_file_words(path)
        source_counts[fn] = wc
        print(f"  {wc:,}")

    grand_total_source = sum(source_counts.values())
    print(f"\n  Grand total source words: {grand_total_source:,}\n")

    # Step 4: allocate per-file targets proportionally, ensuring sum == total_target
    #   Use the "largest remainder" method so rounding never leaves us short/over.
    raw_shares = {
        fn: total_target * wc / grand_total_source
        for fn, wc in source_counts.items()
    }
    floor_shares = {fn: int(v) for fn, v in raw_shares.items()}
    remainder = total_target - sum(floor_shares.values())

    # Distribute leftover words to files with the largest fractional parts
    fractional_order = sorted(
        raw_shares.keys(),
        key=lambda fn: raw_shares[fn] - floor_shares[fn],
        reverse=True,
    )
    targets: dict[str, int] = dict(floor_shares)
    for fn in fractional_order[:remainder]:
        targets[fn] += 1

    assert sum(targets.values()) == total_target, "Allocation arithmetic error"

    # Step 5: extract
    print(
        f"Extracting proportional slices → exactly {total_target:,} words total\n"
        f"Output directory: {small_dir}/\n"
    )
    stats = []
    for fn, src_path in present:
        t = targets[fn]
        dst_path = small_dir / fn
        print(f"  {fn}  (target {t:,})")
        result = extract_words(src_path, dst_path, t)
        result["file"] = fn
        result["source_words"] = source_counts[fn]
        stats.append(result)
        print()

    # Summary
    total_written = sum(s["written_words"] for s in stats)
    print("=" * 62)
    print(f"{'File':<45} {'Source':>12} {'Written':>12}")
    print("-" * 62)
    for s in stats:
        print(f"  {s['file']:<43} {s['source_words']:>12,} {s['written_words']:>12,}")
    print("-" * 62)
    print(f"  {'TOTAL':<43} {grand_total_source:>12,} {total_written:>12,}")
    print("=" * 62)
    print(f"\nDone. {total_written:,} words written across {len(stats)} files → {small_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Download BabyLM strict dataset and extract a proportional slice "
            "from each file so the total word count equals --total-words."
        )
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; process files already in ./strict/",
    )
    p.add_argument(
        "--total-words", type=int, default=10_000_000,
        help="Exact total word count to write across all files (default: 10 000 000)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())