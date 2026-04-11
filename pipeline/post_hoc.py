"""
post_hoc.py — Post-hoc contamination analysis for Infinigram-style querying.

Provides two modes:
1. Index check: Given an eval string, check if its 13-grams match any benchmark
   n-grams — i.e., whether it would have caused document removal during
   decontamination.
2. Corpus scan: Given a query string and language, search the decontaminated
   Parquet shards for documents containing matching n-grams. Returns doc_ids,
   URLs, and surrounding text context.

This enables analysis of how model performance on MECO reading times and
beetle-analyze benchmarks correlates with training data presence.

Usage:
    # Check if a string would have been flagged as contaminated
    python -m pipeline.post_hoc check \\
        --index benchmark_13gram.pkl \\
        --text "The cat sat on the mat and watched the birds fly over."

    # Search the training corpus for n-gram matches
    python -m pipeline.post_hoc scan \\
        --lang pl --text "example query" \\
        --output-dir pipeline_output

    # Batch analysis from a file (one text per line)
    python -m pipeline.post_hoc batch \\
        --index benchmark_13gram.pkl \\
        --input eval_strings.txt \\
        --lang pl --output-dir pipeline_output
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from tqdm import tqdm

from .benchmark_index import BenchmarkIndex
from .utils import extract_ngrams_from_text, tokenize_for_ngrams, extract_ngrams

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("PostHoc")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1: Index Check
# ═══════════════════════════════════════════════════════════════════════════════

def check_contamination(
    text: str,
    index: BenchmarkIndex,
) -> Dict[str, Any]:
    """Check if a text string would have been flagged as contaminated.

    Returns a dict with:
        - is_contaminated: bool
        - overlapping_ngrams: list of matching n-gram tuples
        - num_ngrams_in_text: total n-grams extracted from text
        - num_overlaps: number of matches
    """
    tokens = tokenize_for_ngrams(text)
    ngrams = extract_ngrams(tokens, index.ngram_size)
    overlaps = sorted(ngrams & index.index) if index.index else []

    return {
        "is_contaminated": len(overlaps) > 0,
        "overlapping_ngrams": [" ".join(ng) for ng in overlaps],
        "num_ngrams_in_text": len(ngrams),
        "num_overlaps": len(overlaps),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 2: Corpus Scan
# ═══════════════════════════════════════════════════════════════════════════════

def scan_corpus(
    query_text: str,
    lang: str,
    output_dir: str,
    ngram_size: int = 13,
    max_results: int = 100,
    context_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Search decontaminated Parquet shards for documents containing query n-grams.

    Args:
        query_text: The eval string to search for.
        lang: Language code for the corpus to search.
        output_dir: Base pipeline output directory.
        ngram_size: N-gram size (should match index).
        max_results: Maximum number of matching documents to return.
        context_chars: Characters of context to show around matches.

    Returns:
        List of dicts with: doc_id, url, matched_ngrams, context.
    """
    # Extract query n-grams
    query_tokens = tokenize_for_ngrams(query_text)
    query_ngrams = extract_ngrams(query_tokens, ngram_size)
    if not query_ngrams:
        log.warning("Query text too short for %d-gram extraction", ngram_size)
        return []

    log.info("Searching for %d query n-grams in %s corpus", len(query_ngrams), lang)

    # Load manifest
    parquet_dir = Path(output_dir) / "decontaminated" / lang
    manifest_path = parquet_dir / f"{lang}_manifest.json"

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    results = []
    shard_files = sorted(manifest.keys())

    for shard_name in tqdm(shard_files, desc=f"Scanning {lang} shards"):
        shard_path = parquet_dir / shard_name
        if not shard_path.exists():
            continue

        table = pq.read_table(str(shard_path))
        texts = table.column("text").to_pylist()
        doc_ids = table.column("doc_id").to_pylist()
        urls = table.column("url").to_pylist()

        for text, doc_id, url in zip(texts, doc_ids, urls):
            doc_tokens = tokenize_for_ngrams(text)
            doc_ngrams = extract_ngrams(doc_tokens, ngram_size)
            overlaps = query_ngrams & doc_ngrams

            if overlaps:
                # Extract context around first match
                first_match = " ".join(sorted(overlaps)[0])
                lower_text = text.lower()
                match_pos = lower_text.find(first_match.split()[0])
                if match_pos >= 0:
                    start = max(0, match_pos - context_chars)
                    end = min(len(text), match_pos + context_chars)
                    context = text[start:end]
                else:
                    context = text[:context_chars * 2]

                results.append({
                    "doc_id": doc_id,
                    "url": url,
                    "matched_ngrams": [" ".join(ng) for ng in sorted(overlaps)],
                    "num_matches": len(overlaps),
                    "context": context,
                })

                if len(results) >= max_results:
                    return results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 3: Batch Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def batch_analysis(
    texts: List[str],
    lang: str,
    index: BenchmarkIndex,
    output_dir: str,
    scan_corpus_flag: bool = True,
) -> List[Dict[str, Any]]:
    """Run both index check and corpus scan for a list of eval strings.

    Returns a list of results, one per input text, containing:
        - text: the original text
        - index_check: result from check_contamination()
        - corpus_matches: result from scan_corpus() (if scan_corpus_flag)
    """
    results = []
    for text in tqdm(texts, desc="Batch analysis"):
        result: Dict[str, Any] = {
            "text": text[:200],  # truncate for readability
            "index_check": check_contamination(text, index),
        }
        if scan_corpus_flag:
            result["corpus_matches"] = scan_corpus(
                text, lang, output_dir, ngram_size=index.ngram_size, max_results=10
            )
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc contamination analysis for Infinigram-style querying."
    )
    subparsers = parser.add_subparsers(dest="mode", help="Analysis mode")

    # --- check ---
    check_p = subparsers.add_parser("check", help="Check if text is contaminated")
    check_p.add_argument("--index", required=True, help="Path to benchmark_13gram.pkl")
    check_p.add_argument("--text", required=True, help="Text to check")

    # --- scan ---
    scan_p = subparsers.add_parser("scan", help="Search corpus for n-gram matches")
    scan_p.add_argument("--lang", required=True, help="Language to search")
    scan_p.add_argument("--text", required=True, help="Query text")
    scan_p.add_argument("--output-dir", default="pipeline_output")
    scan_p.add_argument("--max-results", type=int, default=100)

    # --- batch ---
    batch_p = subparsers.add_parser("batch", help="Batch analysis from file")
    batch_p.add_argument("--index", required=True, help="Path to benchmark_13gram.pkl")
    batch_p.add_argument("--input", required=True, help="Input file (one text per line)")
    batch_p.add_argument("--lang", required=True, help="Language for corpus scan")
    batch_p.add_argument("--output-dir", default="pipeline_output")
    batch_p.add_argument("--output", default="post_hoc_results.json",
                         help="Output JSON file")
    batch_p.add_argument("--no-scan", action="store_true",
                         help="Skip corpus scan (index check only)")

    args = parser.parse_args()

    if args.mode == "check":
        index = BenchmarkIndex.load(args.index)
        result = check_contamination(args.text, index)
        print(json.dumps(result, indent=2))

    elif args.mode == "scan":
        results = scan_corpus(args.text, args.lang, args.output_dir,
                              max_results=args.max_results)
        print(json.dumps(results, indent=2, default=str))

    elif args.mode == "batch":
        index = BenchmarkIndex.load(args.index)
        with open(args.input) as f:
            texts = [line.strip() for line in f if line.strip()]
        results = batch_analysis(
            texts, args.lang, index, args.output_dir,
            scan_corpus_flag=not args.no_scan,
        )
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results written to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
