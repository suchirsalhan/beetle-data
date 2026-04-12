"""
stream_held_out.py — Stream held-out FineWeb-2 documents for AoA evaluation.

Reads the decontamination stats to determine how many documents were consumed
during training, then skips past those documents in the FineWeb-2 stream and
collects the next N documents as a held-out evaluation set.

The held-out documents are guaranteed to be unseen during training, preserving
the deterministic ordering of HuggingFace streaming datasets.

Usage:
    python -m pipeline.stream_held_out --lang de --output-dir pipeline_output --n-docs 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import FINEWEB_2, LANG_REGISTRY

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("HeldOut")


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_DOCS = 10_000
SHARD_SIZE = 10_000

SCHEMA = pa.schema([
    ("text", pa.utf8()),
    ("url", pa.utf8()),
    ("doc_id", pa.int64()),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Core Logic
# ═══════════════════════════════════════════════════════════════════════════════

def load_docs_streamed(output_dir: str, lang: str) -> int:
    """Read docs_streamed from the decontamination stats file.

    Args:
        output_dir: Base pipeline output directory.
        lang: 2-letter language code.

    Returns:
        Number of documents consumed during training.

    Raises:
        FileNotFoundError: If the stats file does not exist.
        KeyError: If docs_streamed is missing from the stats file.
    """
    stats_path = Path(output_dir) / "decontaminated" / lang / f"{lang}_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    docs_streamed = stats["docs_streamed"]
    log.info("Found docs_streamed=%d for %s in %s", docs_streamed, lang, stats_path)
    return docs_streamed


def stream_held_out(
    lang: str,
    output_dir: str,
    n_docs: int = DEFAULT_N_DOCS,
) -> Dict:
    """Stream held-out documents from FineWeb-2 for a single language.

    Skips past the documents used during training, then collects the next
    n_docs documents and writes them as Parquet shards.

    Args:
        lang: 2-letter language code (e.g., 'de').
        output_dir: Base pipeline output directory.
        n_docs: Number of held-out documents to collect.

    Returns:
        Dict with held-out stats.
    """
    from datasets import load_dataset

    lang_config = LANG_REGISTRY[lang]
    t0 = time.time()

    # Read how many docs were consumed during training
    skip_n = load_docs_streamed(output_dir, lang)

    # Output directory
    out_dir = Path(output_dir) / "held_out" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Streaming held-out data for %s (%s)", lang, lang_config.name)
    log.info("  Skipping first %d docs (used during training)", skip_n)
    log.info("  Collecting next %d docs", n_docs)

    # Stream from FineWeb-2
    ds = load_dataset(
        FINEWEB_2,
        name=lang_config.fw2_name,
        split="train",
        streaming=True,
    )

    # Skip past training documents
    ds = ds.skip(skip_n)

    # Collect held-out documents
    buffer: List[Dict] = []
    shard_idx = 0
    docs_collected = 0

    pbar = tqdm(total=n_docs, unit="docs", desc=f"Held-out {lang}", dynamic_ncols=True)

    for doc_id, entry in enumerate(ds):
        if docs_collected >= n_docs:
            break

        buffer.append({
            "text": entry.get("text", ""),
            "url": entry.get("url", ""),
            "doc_id": doc_id,
        })
        docs_collected += 1
        pbar.update(1)

        # Flush shard
        if len(buffer) >= SHARD_SIZE:
            _write_shard(out_dir, lang, shard_idx, buffer)
            shard_idx += 1
            buffer.clear()

    # Flush remaining
    if buffer:
        _write_shard(out_dir, lang, shard_idx, buffer)
        shard_idx += 1
        buffer.clear()

    pbar.close()

    # Clean up streaming iterator
    del ds

    wall_time = time.time() - t0

    # Build stats
    stats = {
        "lang": lang,
        "docs_skipped": skip_n,
        "docs_collected": docs_collected,
        "shards_written": shard_idx,
        "wall_time_sec": round(wall_time, 1),
    }

    # Save stats
    stats_path = out_dir / f"{lang}_held_out_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Done %s: %d held-out docs, %d shards, %.1f min",
             lang, docs_collected, shard_idx, wall_time / 60)

    return stats


def _write_shard(out_dir: Path, lang: str, shard_idx: int, buffer: List[Dict]) -> str:
    """Write a buffer of documents as a Parquet shard.

    Args:
        out_dir: Output directory for shards.
        lang: Language code (used in filename).
        shard_idx: Shard index number.
        buffer: List of document dicts with text, url, doc_id.

    Returns:
        Path to the written shard file.
    """
    fname = f"{lang}_held_out_{shard_idx:05d}.parquet"
    path = out_dir / fname

    table = pa.Table.from_pydict(
        {
            "text": [d["text"] for d in buffer],
            "url": [d["url"] for d in buffer],
            "doc_id": [d["doc_id"] for d in buffer],
        },
        schema=SCHEMA,
    )
    pq.write_table(table, str(path), compression="snappy")
    log.info("Wrote shard %s (%d docs)", fname, len(buffer))
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stream held-out FineWeb-2 documents for AoA evaluation."
    )
    parser.add_argument("--lang", type=str, required=True,
                        help="Language code to process (e.g., 'de')")
    parser.add_argument("--output-dir", type=str, default="pipeline_output",
                        help="Base output directory")
    parser.add_argument("--n-docs", type=int, default=DEFAULT_N_DOCS,
                        help="Number of held-out documents to collect (default: 10000)")
    args = parser.parse_args()

    if args.lang not in LANG_REGISTRY:
        parser.error(f"Unknown language: {args.lang}. "
                     f"Available: {', '.join(sorted(LANG_REGISTRY.keys()))}")

    stream_held_out(
        lang=args.lang,
        output_dir=args.output_dir,
        n_docs=args.n_docs,
    )


if __name__ == "__main__":
    main()
