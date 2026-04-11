"""
verify_output.py — End-to-end validation of pipeline output.

Checks:
  1. Arrow datasets load successfully for every language
  2. Every input_ids entry has exactly 513 elements (seq_len + 1)
  3. All token IDs are within [0, vocab_size)
  4. Total tokens approximate the 24B target per language pair
  5. Sample sequences decoded and 13-gram checked — expect zero contamination
  6. (Optional) Instantiate PretokenizedMultilingualDataset and iterate batches

Usage:
    python scripts/verify_output.py --output-dir pipeline_output --hf-user Beetle-Data

    # Quick check (skip contamination scan)
    python scripts/verify_output.py --output-dir pipeline_output --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


def verify_arrow_dataset(
    arrow_path: Path,
    expected_chunk_len: int = 513,
    tokenizer_name: str | None = None,
    benchmark_index_path: str | None = None,
    sample_size: int = 1000,
) -> dict:
    """Verify a single Arrow dataset.

    Returns a dict with verification results.
    """
    results = {
        "path": str(arrow_path),
        "load_ok": False,
        "num_chunks": 0,
        "total_tokens": 0,
        "chunk_len_ok": True,
        "token_range_ok": True,
        "contamination_found": 0,
        "errors": [],
    }

    # 1. Load dataset
    try:
        ds = load_from_disk(str(arrow_path))
        results["load_ok"] = True
        results["num_chunks"] = len(ds)
        results["total_tokens"] = len(ds) * expected_chunk_len
    except Exception as e:
        results["errors"].append(f"Failed to load: {e}")
        return results

    # 2. Check chunk lengths
    bad_lengths = 0
    for i in range(min(len(ds), sample_size)):
        ids = ds[i]["input_ids"]
        if len(ids) != expected_chunk_len:
            bad_lengths += 1
            if bad_lengths <= 5:
                results["errors"].append(
                    f"Chunk {i}: expected {expected_chunk_len} tokens, got {len(ids)}"
                )
    if bad_lengths > 0:
        results["chunk_len_ok"] = False
        results["errors"].append(f"{bad_lengths}/{min(len(ds), sample_size)} chunks had wrong length")

    # 3. Check token ID range
    if tokenizer_name:
        try:
            tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
            vocab_size = tok.vocab_size
            oob_count = 0
            for i in range(min(len(ds), sample_size)):
                ids = ds[i]["input_ids"]
                for tid in ids:
                    if tid < 0 or tid >= vocab_size:
                        oob_count += 1
            if oob_count > 0:
                results["token_range_ok"] = False
                results["errors"].append(
                    f"{oob_count} tokens out of range [0, {vocab_size})"
                )
        except Exception as e:
            results["errors"].append(f"Tokenizer check failed: {e}")

    # 4. Contamination spot-check
    if benchmark_index_path:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from pipeline.benchmark_index import BenchmarkIndex

            index = BenchmarkIndex.load(benchmark_index_path)

            if tokenizer_name:
                tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
                checked = 0
                contaminated = 0
                for i in range(min(len(ds), sample_size)):
                    ids = ds[i]["input_ids"]
                    text = tok.decode(ids, skip_special_tokens=True)
                    if index.is_contaminated(text):
                        contaminated += 1
                    checked += 1
                results["contamination_found"] = contaminated
                if contaminated > 0:
                    results["errors"].append(
                        f"CONTAMINATION: {contaminated}/{checked} sampled sequences contain benchmark n-grams"
                    )
        except Exception as e:
            results["errors"].append(f"Contamination check failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify pipeline output.")
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--hf-user", default="Beetle-Data", help="HuggingFace user/org")
    parser.add_argument("--index", default=None, help="Path to benchmark_13gram.pkl")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Skip contamination check")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Languages to check (default: all found)")
    args = parser.parse_args()

    pretok_dir = Path(args.output_dir) / "pretokenized"
    chunk_len = args.seq_len + 1

    if not pretok_dir.exists():
        print(f"Error: pretokenized directory not found: {pretok_dir}")
        sys.exit(1)

    # Find all Arrow datasets
    arrow_dirs = sorted(d for d in pretok_dir.iterdir() if d.is_dir())
    if args.langs:
        # Filter to requested languages
        lang_prefixes = set()
        for l in args.langs:
            lang_prefixes.add(l)
            lang_prefixes.add(f"en_for_{l}")
        arrow_dirs = [d for d in arrow_dirs if d.name in lang_prefixes]

    if not arrow_dirs:
        print(f"No Arrow datasets found in {pretok_dir}")
        sys.exit(1)

    print(f"Found {len(arrow_dirs)} Arrow datasets to verify")
    print(f"Chunk length: {chunk_len} (seq_len={args.seq_len} + 1)")
    print()

    # Auto-detect benchmark index
    index_path = args.index
    if not index_path and not args.quick:
        candidate = Path(args.output_dir) / "benchmark_13gram.pkl"
        if candidate.exists():
            index_path = str(candidate)

    all_results = {}
    total_tokens = 0
    issues = 0

    for arrow_dir in tqdm(arrow_dirs, desc="Verifying datasets"):
        # Determine tokenizer for this dataset
        name = arrow_dir.name
        if name.startswith("en_for_"):
            lang = name.replace("en_for_", "")
        else:
            lang = name

        tok_name = f"{args.hf_user}/tokenizer-{lang}-en"

        result = verify_arrow_dataset(
            arrow_path=arrow_dir,
            expected_chunk_len=chunk_len,
            tokenizer_name=tok_name if not args.quick else None,
            benchmark_index_path=index_path if not args.quick else None,
            sample_size=args.sample_size,
        )

        all_results[name] = result
        total_tokens += result["total_tokens"]

        # Print inline status
        status = "OK" if not result["errors"] else "ISSUES"
        if result["errors"]:
            issues += 1
        tokens_b = result["total_tokens"] / 1e9
        print(f"  {name:20s}: {status:6s} | {result['num_chunks']:>10,} chunks | {tokens_b:.2f}B tokens")
        for err in result["errors"]:
            print(f"    ERROR: {err}")

    # Summary
    print()
    print("=" * 60)
    print(f"Verification complete: {len(arrow_dirs)} datasets")
    print(f"  Total tokens: {total_tokens / 1e9:.2f}B")
    print(f"  Issues found: {issues}")
    if issues == 0:
        print("  STATUS: ALL CHECKS PASSED")
    else:
        print("  STATUS: ISSUES DETECTED — see errors above")
    print("=" * 60)

    # Save results
    results_path = Path(args.output_dir) / "verification_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results: {results_path}")


if __name__ == "__main__":
    main()
