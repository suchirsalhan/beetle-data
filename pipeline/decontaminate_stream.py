"""
decontaminate_stream.py — Stage 2: Stream from FineWeb + decontaminate + write Parquet.

Streams documents from FineWeb-Edu (English) or FineWeb-2 (non-English), applies
basic text cleaning, checks each document against the 13-gram benchmark index,
and writes clean documents to Parquet shards. Contaminated documents are discarded.

The output Parquet shards preserve document ordering for Infinigram-style post-hoc
analysis. Each shard contains columns: text, url, doc_id, word_count.

A manifest.json is written per language mapping shard filenames to doc_id ranges,
enabling efficient post-hoc lookup without scanning all shards.

Usage:
    # Single language
    python -m pipeline.decontaminate_stream --lang pl --index benchmark_13gram.pkl

    # All languages assigned to a SLURM node
    python -m pipeline.decontaminate_stream --node-id 0 --index benchmark_13gram.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, Queue, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .benchmark_index import BenchmarkIndex
from .config import (
    FINEWEB_2,
    FINEWEB_EDU,
    LANG_REGISTRY,
    PipelineConfig,
    langs_for_node,
)
from .utils import normalize_text, passes_length_filter, word_count

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("Decontaminate")


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DecontaminationStats:
    lang: str = ""
    docs_streamed: int = 0
    docs_too_short: int = 0
    docs_too_long: int = 0
    docs_contaminated: int = 0
    docs_clean: int = 0
    words_accumulated: int = 0
    shards_written: int = 0
    wall_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "docs_streamed": self.docs_streamed,
            "docs_too_short": self.docs_too_short,
            "docs_too_long": self.docs_too_long,
            "docs_contaminated": self.docs_contaminated,
            "docs_clean": self.docs_clean,
            "contamination_rate": (
                self.docs_contaminated / max(1, self.docs_streamed - self.docs_too_short - self.docs_too_long)
            ),
            "words_accumulated": self.words_accumulated,
            "shards_written": self.shards_written,
            "wall_time_sec": round(self.wall_time_sec, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Worker: Decontamination Check (runs in forked processes)
# ═══════════════════════════════════════════════════════════════════════════════

# Global in each worker process (set once via fork inheritance)
_worker_index: Optional[BenchmarkIndex] = None
_worker_min_chars: int = 200
_worker_max_chars: int = 100_000


def _init_worker(index: BenchmarkIndex, min_chars: int, max_chars: int):
    """Initializer for worker processes — receives the shared index."""
    global _worker_index, _worker_min_chars, _worker_max_chars
    _worker_index = index
    _worker_min_chars = min_chars
    _worker_max_chars = max_chars


def _process_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single document: clean, filter, decontaminate.

    Returns the cleaned doc dict if it passes all checks, else None.
    The returned dict includes a 'status' field:
      - 'clean': passed all checks
      - 'too_short' / 'too_long': length filter
      - 'contaminated': contains benchmark n-grams
    """
    text = doc.get("text", "")
    url = doc.get("url", "")

    # Normalize
    text = normalize_text(text)

    # Length filter
    n = len(text)
    if n < _worker_min_chars:
        return {"status": "too_short"}
    if n > _worker_max_chars:
        return {"status": "too_long"}

    # Decontamination check
    if _worker_index and _worker_index.is_contaminated(text):
        return {"status": "contaminated"}

    return {
        "status": "clean",
        "text": text,
        "url": url,
        "word_count": word_count(text),
    }


def _process_batch(docs: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of documents (called by pool.map)."""
    return [_process_doc(d) for d in docs]


# ═══════════════════════════════════════════════════════════════════════════════
# Parquet Shard Writer
# ═══════════════════════════════════════════════════════════════════════════════

class ShardWriter:
    """Accumulates clean documents and writes Parquet shards."""

    SCHEMA = pa.schema([
        ("text", pa.utf8()),
        ("url", pa.utf8()),
        ("doc_id", pa.int64()),
        ("word_count", pa.int32()),
    ])

    def __init__(self, output_dir: Path, lang: str, shard_size: int = 50_000):
        self.output_dir = output_dir
        self.lang = lang
        self.shard_size = shard_size
        self.buffer: List[Dict] = []
        self.shard_idx = 0
        self.next_doc_id = 0
        self.manifest: Dict[str, Tuple[int, int]] = {}

    def add(self, doc: Dict[str, Any]) -> Optional[str]:
        """Add a clean document. Returns shard path if a shard was flushed."""
        doc["doc_id"] = self.next_doc_id
        self.next_doc_id += 1
        self.buffer.append(doc)

        if len(self.buffer) >= self.shard_size:
            return self._flush()
        return None

    def finalize(self) -> Optional[str]:
        """Flush remaining docs. Returns shard path if any were written."""
        if self.buffer:
            return self._flush()
        return None

    def _flush(self) -> str:
        """Write current buffer as a Parquet shard."""
        fname = f"{self.lang}_clean_{self.shard_idx:05d}.parquet"
        path = self.output_dir / fname

        first_id = self.buffer[0]["doc_id"]
        last_id = self.buffer[-1]["doc_id"]

        table = pa.Table.from_pydict(
            {
                "text": [d["text"] for d in self.buffer],
                "url": [d["url"] for d in self.buffer],
                "doc_id": [d["doc_id"] for d in self.buffer],
                "word_count": [d["word_count"] for d in self.buffer],
            },
            schema=self.SCHEMA,
        )
        pq.write_table(table, str(path), compression="snappy")

        self.manifest[fname] = (first_id, last_id)
        self.shard_idx += 1
        self.buffer.clear()
        return str(path)

    def save_manifest(self) -> str:
        """Write manifest.json mapping shard files to doc_id ranges."""
        manifest_path = self.output_dir / f"{self.lang}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return str(manifest_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Decontamination Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def decontaminate_language(
    lang: str,
    index: BenchmarkIndex,
    cfg: PipelineConfig,
) -> DecontaminationStats:
    """Stream, clean, decontaminate, and write Parquet for one language.

    Args:
        lang: 2-letter language code (e.g., 'pl', 'en').
        index: Pre-built 13-gram benchmark index.
        cfg: Pipeline configuration.

    Returns:
        DecontaminationStats with counts and timing.
    """
    from datasets import load_dataset

    lc = LANG_REGISTRY[lang]
    stats = DecontaminationStats(lang=lang)
    t0 = time.time()

    # Output directory
    out_dir = Path(cfg.output_dir) / "decontaminated" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # Word target (with overshoot)
    target_words = int(cfg.target_words_per_lang * cfg.word_overshoot_factor)

    log.info("Starting decontamination for %s (%s)", lang, lc.name)
    log.info("  Target: ~%d words (%.0f%% overshoot)",
             target_words, (cfg.word_overshoot_factor - 1) * 100)

    # Stream dataset
    if lc.is_english:
        ds = load_dataset(FINEWEB_EDU, split="train", streaming=True)
    else:
        ds = load_dataset(FINEWEB_2, name=lc.fw2_name, split="train", streaming=True)

    # Shard writer
    writer = ShardWriter(out_dir, lang, shard_size=cfg.shard_size)

    # Progress bar
    pbar = tqdm(
        total=target_words,
        unit="words",
        desc=f"Decontaminating {lang}",
        unit_scale=True,
        dynamic_ncols=True,
    )

    # Process with multiprocessing pool
    num_workers = min(cfg.num_workers, cpu_count())
    batch: List[Dict] = []

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(index, cfg.min_doc_chars, cfg.max_doc_chars),
    ) as pool:

        for entry in ds:
            batch.append({"text": entry.get("text", ""), "url": entry.get("url", "")})

            if len(batch) >= cfg.batch_size * num_workers:
                # Split into sub-batches and distribute
                sub_batches = [
                    batch[i:i + cfg.batch_size]
                    for i in range(0, len(batch), cfg.batch_size)
                ]
                results = pool.map(_process_batch, sub_batches, chunksize=1)

                for sub_result in results:
                    for result in sub_result:
                        stats.docs_streamed += 1
                        if result is None:
                            continue
                        status = result["status"]
                        if status == "too_short":
                            stats.docs_too_short += 1
                        elif status == "too_long":
                            stats.docs_too_long += 1
                        elif status == "contaminated":
                            stats.docs_contaminated += 1
                        elif status == "clean":
                            stats.docs_clean += 1
                            stats.words_accumulated += result["word_count"]
                            writer.add(result)
                            pbar.update(result["word_count"])

                batch.clear()

                # Check if we've reached the word target
                if stats.words_accumulated >= target_words:
                    break

        # Process remaining batch
        if batch and stats.words_accumulated < target_words:
            sub_batches = [
                batch[i:i + cfg.batch_size]
                for i in range(0, len(batch), cfg.batch_size)
            ]
            results = pool.map(_process_batch, sub_batches, chunksize=1)

            for sub_result in results:
                for result in sub_result:
                    stats.docs_streamed += 1
                    if result is None:
                        continue
                    status = result["status"]
                    if status == "too_short":
                        stats.docs_too_short += 1
                    elif status == "too_long":
                        stats.docs_too_long += 1
                    elif status == "contaminated":
                        stats.docs_contaminated += 1
                    elif status == "clean":
                        stats.docs_clean += 1
                        stats.words_accumulated += result["word_count"]
                        writer.add(result)
                        pbar.update(result["word_count"])

    pbar.close()

    # Finalize
    writer.finalize()
    manifest_path = writer.save_manifest()
    stats.shards_written = writer.shard_idx
    stats.wall_time_sec = time.time() - t0

    # Save stats
    stats_path = out_dir / f"{lang}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info("Done %s: %d clean docs, %d contaminated (%.2f%%), %d shards, %.1f min",
             lang, stats.docs_clean, stats.docs_contaminated,
             stats.to_dict()["contamination_rate"] * 100,
             stats.shards_written, stats.wall_time_sec / 60)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Stream FineWeb + decontaminate + write Parquet."
    )
    parser.add_argument("--lang", type=str, default=None,
                        help="Single language code to process (e.g., 'pl')")
    parser.add_argument("--node-id", type=int, default=None,
                        help="SLURM node ID (0-3) — processes all assigned languages")
    parser.add_argument("--index", type=str, required=True,
                        help="Path to benchmark_13gram.pkl from Stage 1")
    parser.add_argument("--output-dir", type=str, default="pipeline_output",
                        help="Base output directory")
    parser.add_argument("--target-words", type=int, default=None,
                        help="Override target words per language (default: from config)")
    parser.add_argument("--num-workers", type=int, default=24,
                        help="Number of multiprocessing workers")
    parser.add_argument("--shard-size", type=int, default=50_000,
                        help="Documents per Parquet shard")
    args = parser.parse_args()

    # Determine languages to process
    if args.lang:
        languages = [args.lang]
    elif args.node_id is not None:
        languages = langs_for_node(args.node_id)
    else:
        parser.error("Must specify --lang or --node-id")
        return

    # Load index
    log.info("Loading benchmark index from %s", args.index)
    index = BenchmarkIndex.load(args.index)

    # Build config
    cfg = PipelineConfig(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
    )
    if args.target_words:
        cfg.target_words_per_lang = args.target_words

    # Process each language sequentially
    all_stats = {}
    for lang in languages:
        if lang not in LANG_REGISTRY:
            log.error("Unknown language: %s", lang)
            continue
        stats = decontaminate_language(lang, index, cfg)
        all_stats[lang] = stats.to_dict()

    # Write combined stats
    combined_path = Path(args.output_dir) / "decontamination_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    log.info("All done. Summary written to %s", combined_path)


if __name__ == "__main__":
    main()
