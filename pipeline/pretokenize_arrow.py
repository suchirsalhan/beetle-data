"""
pretokenize_arrow.py — Stage 3: Read clean Parquet, tokenize, write Arrow.

Reads decontaminated Parquet shards from Stage 2 **one shard at a time**,
tokenizes each document using the pre-trained bilingual BPE tokenizer, packs
tokens into 513-token sequences (512 + 1 for input + label), and writes
Arrow sub-datasets in batches. The sub-datasets are then concatenated into
a single HuggingFace Arrow dataset.

Memory-safe design:
  - Parquet shards are read one at a time (never all at once)
  - Chunks are flushed to disk every ARROW_FLUSH_CHUNKS chunks (~2 GB)
  - Final concatenation uses memory-mapped Arrow files (not in-RAM)

Critical: output format must match beetlelm's PretokenizedMultilingualDataset
exactly (see beetlelm/src/bilingual/data/pretokenize.py:110-133):
  - chunk_len = seq_len + 1 = 513
  - tokenizer.encode(text, add_special_tokens=False)
  - No cross-document token bleeding (buf = [] at document boundaries)
  - Dataset.from_dict({"input_ids": chunks}).save_to_disk(out_path)

Usage:
    # Single language (L1 side)
    python -m pipeline.pretokenize_arrow --lang pl --side l1

    # English side for a specific language pair
    python -m pipeline.pretokenize_arrow --lang pl --side en

    # All languages for a SLURM node (both L1 and EN sides)
    python -m pipeline.pretokenize_arrow --node-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from tqdm import tqdm

from .config import (
    LANG_REGISTRY,
    PipelineConfig,
    langs_for_node,
    tokenizer_repo,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("Pretokenize")

# Maximum chunks to hold in memory before flushing to a temporary Arrow file.
# 500K chunks * 513 ints * 8 bytes (Python int) ≈ 2 GB peak per flush.
ARROW_FLUSH_CHUNKS = 500_000


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PretokenizationStats:
    lang: str = ""
    side: str = ""
    docs_processed: int = 0
    tokens_total: int = 0
    chunks_created: int = 0
    tokens_discarded: int = 0  # remainder tokens at doc boundaries
    shards_read: int = 0
    arrow_parts_written: int = 0
    wall_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "side": self.side,
            "docs_processed": self.docs_processed,
            "tokens_total": self.tokens_total,
            "chunks_created": self.chunks_created,
            "tokens_discarded": self.tokens_discarded,
            "shards_read": self.shards_read,
            "arrow_parts_written": self.arrow_parts_written,
            "wall_time_sec": round(self.wall_time_sec, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Worker: Tokenization (runs in forked processes)
# ═══════════════════════════════════════════════════════════════════════════════

_worker_tokenizer = None
_worker_chunk_len: int = 513


def _init_tokenizer_worker(tokenizer_name: str, chunk_len: int):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer, _worker_chunk_len
    from transformers import PreTrainedTokenizerFast
    _worker_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
    _worker_chunk_len = chunk_len


def _tokenize_and_chunk(text: str) -> Tuple[List[List[int]], int, int]:
    """Tokenize a single document and pack into fixed-size chunks.

    Returns:
        (chunks, total_tokens, discarded_tokens)

    Exactly mirrors beetlelm/src/bilingual/data/pretokenize.py:114-125:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= chunk_len:
            chunks.append(buf[:chunk_len])
            buf = buf[chunk_len:]
        buf = []  # discard remainder at document boundary
    """
    ids = _worker_tokenizer.encode(text, add_special_tokens=False)
    total = len(ids)
    chunks = []

    buf = list(ids)
    while len(buf) >= _worker_chunk_len:
        chunks.append(buf[:_worker_chunk_len])
        buf = buf[_worker_chunk_len:]

    discarded = len(buf)  # remainder at doc boundary
    return chunks, total, discarded


def _process_text_batch(texts: List[str]) -> Tuple[List[List[int]], int, int]:
    """Process a batch of document texts — tokenize and chunk each one."""
    all_chunks = []
    total_tokens = 0
    total_discarded = 0

    for text in texts:
        if not text or not text.strip():
            continue
        chunks, tok_count, disc_count = _tokenize_and_chunk(text)
        all_chunks.extend(chunks)
        total_tokens += tok_count
        total_discarded += disc_count

    return all_chunks, total_tokens, total_discarded


# ═══════════════════════════════════════════════════════════════════════════════
# Arrow Dataset Writer — Incremental Flushing
# ═══════════════════════════════════════════════════════════════════════════════

class IncrementalArrowWriter:
    """Accumulates chunks in memory and flushes to disk in parts.

    Each flush produces a temporary Arrow dataset in a numbered sub-directory.
    At finalization, all parts are concatenated into the final Arrow dataset
    using datasets.concatenate_datasets (memory-mapped, not in-RAM).

    Memory budget: at most ARROW_FLUSH_CHUNKS * chunk_len * ~8 bytes in RAM
    at any time (~2 GB for 500K chunks of 513 ints).
    """

    def __init__(self, output_dir: Path, flush_threshold: int = ARROW_FLUSH_CHUNKS):
        self.output_dir = output_dir
        self.flush_threshold = flush_threshold
        self.buffer: List[List[int]] = []
        self.part_idx = 0
        self.parts_dir = output_dir / "_parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.total_chunks = 0

    def add_chunks(self, chunks: List[List[int]]) -> None:
        """Add chunks to the buffer; flush to disk if threshold reached."""
        self.buffer.extend(chunks)
        while len(self.buffer) >= self.flush_threshold:
            self._flush(self.buffer[:self.flush_threshold])
            self.buffer = self.buffer[self.flush_threshold:]

    def _flush(self, chunks: List[List[int]]) -> None:
        """Write a batch of chunks as a temporary Arrow dataset."""
        from datasets import Dataset

        part_path = self.parts_dir / f"part_{self.part_idx:04d}"
        ds = Dataset.from_dict({"input_ids": chunks})
        ds.save_to_disk(str(part_path))
        self.total_chunks += len(chunks)
        self.part_idx += 1
        log.info("  Flushed part %d: %d chunks (total: %d)",
                 self.part_idx, len(chunks), self.total_chunks)

    def finalize(self, target_chunks: Optional[int] = None) -> int:
        """Flush remaining buffer and concatenate all parts into final dataset.

        Args:
            target_chunks: If set, truncate to this many chunks total.

        Returns:
            Total number of chunks in the final dataset.
        """
        from datasets import Dataset, concatenate_datasets, load_from_disk

        # Flush any remaining buffer
        if self.buffer:
            self._flush(self.buffer)
            self.buffer.clear()

        # Load all parts (memory-mapped, not fully in RAM)
        part_dirs = sorted(self.parts_dir.glob("part_*"))

        if not part_dirs:
            # Edge case: no data
            ds = Dataset.from_dict({"input_ids": []})
            ds.save_to_disk(str(self.output_dir))
            return 0

        if len(part_dirs) == 1:
            # Single part — just move it
            part_ds = load_from_disk(str(part_dirs[0]))
            if target_chunks and len(part_ds) > target_chunks:
                part_ds = part_ds.select(range(target_chunks))
            part_ds.save_to_disk(str(self.output_dir))
            final_count = len(part_ds)
        else:
            # Concatenate parts (memory-mapped)
            parts = [load_from_disk(str(p)) for p in part_dirs]
            combined = concatenate_datasets(parts)

            if target_chunks and len(combined) > target_chunks:
                combined = combined.select(range(target_chunks))

            combined.save_to_disk(str(self.output_dir))
            final_count = len(combined)

        # Clean up temp parts
        shutil.rmtree(self.parts_dir, ignore_errors=True)

        log.info("  Final dataset: %d chunks saved to %s", final_count, self.output_dir)
        return final_count


# ═══════════════════════════════════════════════════════════════════════════════
# Parquet Shard Iterator (memory-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def _iter_parquet_shards(parquet_dir: Path):
    """Yield (shard_path, texts) for each Parquet shard, one at a time.

    Only one shard's text data is in memory at any point.
    Each shard is ~50K docs * ~5 KB avg = ~250 MB in RAM.
    """
    shard_files = sorted(parquet_dir.glob("*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No Parquet shards found in {parquet_dir}")

    log.info("Found %d Parquet shards in %s", len(shard_files), parquet_dir)

    for shard_path in shard_files:
        table = pq.read_table(str(shard_path), columns=["text"])
        texts = table.column("text").to_pylist()
        del table  # release Arrow memory immediately
        yield shard_path, texts


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pretokenization Pipeline (memory-safe streaming)
# ═══════════════════════════════════════════════════════════════════════════════

def pretokenize_language(
    lang: str,
    side: str,
    cfg: PipelineConfig,
    target_tokens: Optional[int] = None,
) -> PretokenizationStats:
    """Pretokenize decontaminated data for one language/side pair.

    Memory-safe: reads one Parquet shard at a time, flushes Arrow chunks
    to disk incrementally (never holds more than ~2 GB of chunks in RAM).

    Args:
        lang: 2-letter language code (e.g., 'pl').
        side: 'l1' for the non-English language, 'en' for the English side.
        cfg: Pipeline configuration.
        target_tokens: Max tokens to produce (None = all available).

    Returns:
        PretokenizationStats with counts and timing.
    """
    stats = PretokenizationStats(lang=lang, side=side)
    t0 = time.time()

    # Determine source and output paths
    if side == "l1":
        source_lang = lang
        output_name = lang
    else:
        source_lang = "en"
        output_name = f"en_for_{lang}"

    parquet_dir = Path(cfg.output_dir) / "decontaminated" / source_lang
    arrow_dir = Path(cfg.output_dir) / "pretokenized" / output_name

    # Use the bilingual tokenizer for this language pair
    tok_repo = tokenizer_repo(lang, cfg.hf_user)

    log.info("Pretokenizing %s (side=%s)", lang, side)
    log.info("  Source: %s", parquet_dir)
    log.info("  Tokenizer: %s", tok_repo)
    log.info("  Output: %s", arrow_dir)
    log.info("  Chunk length: %d (seq_len=%d + 1)", cfg.chunk_len, cfg.seq_len)

    # Compute token target based on bilingual ratio.
    # total_pair_tokens: the clean token budget for this bilingual pair.
    # Read from cfg.target_tokens_per_lang (set in config.py TARGET_TOKENS_PER_LANG = 24B).
    # Change TARGET_TOKENS_PER_LANG in config.py to adjust — no other code needs updating.
    if target_tokens is None:
        total_pair_tokens = getattr(cfg, "target_tokens_per_lang", 24_000_000_000)
        if side == "l1":
            target_tokens = int(total_pair_tokens * cfg.l1_ratio)
        else:
            target_tokens = int(total_pair_tokens * cfg.en_ratio)

    target_chunks = target_tokens // cfg.chunk_len
    log.info("  Target: %d tokens (%d chunks)", target_tokens, target_chunks)

    # Incremental Arrow writer (flushes to disk every 500K chunks ≈ 2 GB)
    writer = IncrementalArrowWriter(arrow_dir)

    # Process Parquet shards one at a time
    num_workers = min(cfg.num_workers, cpu_count())
    batch_size = cfg.batch_size
    reached_target = False

    pbar = tqdm(
        total=target_tokens,
        unit="tok",
        desc=f"Tokenizing {output_name}",
        unit_scale=True,
        dynamic_ncols=True,
    )

    with Pool(
        processes=num_workers,
        initializer=_init_tokenizer_worker,
        initargs=(tok_repo, cfg.chunk_len),
    ) as pool:

        for shard_path, texts in _iter_parquet_shards(parquet_dir):
            if reached_target:
                break

            stats.shards_read += 1

            # Split this shard's texts into batches for parallel tokenization
            text_batches = [
                texts[i:i + batch_size]
                for i in range(0, len(texts), batch_size)
            ]
            del texts  # free shard text memory before processing

            for result in pool.imap(_process_text_batch, text_batches, chunksize=1):
                chunks, tok_count, disc_count = result

                writer.add_chunks(chunks)
                stats.tokens_total += tok_count
                stats.tokens_discarded += disc_count
                stats.docs_processed += batch_size
                pbar.update(tok_count)

                # Check if we've accumulated enough chunks
                if writer.total_chunks + len(writer.buffer) >= target_chunks:
                    reached_target = True
                    break

    pbar.close()

    # Finalize: flush remaining buffer, concatenate parts, truncate to target
    final_count = writer.finalize(target_chunks=target_chunks)
    stats.chunks_created = final_count
    stats.arrow_parts_written = writer.part_idx
    stats.wall_time_sec = time.time() - t0

    # Save stats
    stats_path = arrow_dir / "pretok_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info("Done %s/%s: %d chunks (%d tokens), %d shards read, %.1f min",
             lang, side, stats.chunks_created,
             stats.chunks_created * cfg.chunk_len,
             stats.shards_read, stats.wall_time_sec / 60)

    return stats


def upload_to_hf_and_cleanup(
    arrow_dir: Path,
    repo_id: str,
    cfg: PipelineConfig,
) -> bool:
    """Upload an Arrow dataset to HuggingFace and optionally delete local copy.

    Args:
        arrow_dir: Local path to the Arrow dataset.
        repo_id: HuggingFace repo ID (e.g., "Beetle-Data/pl-24B").
        cfg: Pipeline configuration.

    Returns:
        True if upload succeeded, False otherwise.
    """
    if not cfg.upload_to_hf:
        return True

    from huggingface_hub import HfApi, create_repo

    try:
        api = HfApi(token=cfg.hf_token)

        # Create repo if it doesn't exist
        create_repo(repo_id, repo_type="dataset", exist_ok=True,
                    token=cfg.hf_token, private=False)

        # Upload entire Arrow directory
        log.info("Uploading %s to %s ...", arrow_dir, repo_id)
        api.upload_folder(
            folder_path=str(arrow_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Pretokenized data ({arrow_dir.name})",
        )
        log.info("Upload complete: %s", repo_id)

        # Delete local copy to free disk
        if cfg.cleanup_after_upload:
            shutil.rmtree(arrow_dir, ignore_errors=True)
            log.info("Deleted local copy: %s", arrow_dir)

        return True

    except Exception as e:
        log.error("Upload failed for %s: %s", repo_id, e)
        log.error("Local files preserved at %s", arrow_dir)
        return False


def cleanup_stage2_parquet(lang: str, cfg: PipelineConfig) -> None:
    """Delete decontaminated Parquet shards for a language after pretokenization.

    Only deletes non-English languages. English Parquet is kept until all
    19 language pairs have been pretokenized.
    """
    if not cfg.cleanup_stage2_after_pretok:
        return
    if lang == "en":
        log.info("Keeping EN Parquet (needed for all bilingual pairs)")
        return

    parquet_dir = Path(cfg.output_dir) / "decontaminated" / lang
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir, ignore_errors=True)
        log.info("Cleaned up Stage 2 Parquet for %s: %s", lang, parquet_dir)


def pretokenize_pair(
    lang: str,
    cfg: PipelineConfig,
) -> Tuple[PretokenizationStats, PretokenizationStats]:
    """Pretokenize both L1 and EN sides, upload to HF, clean up local files.

    Storage-optimized flow:
      1. Pretokenize L1 → upload to HF → delete local Arrow
      2. Pretokenize EN → upload to HF → delete local Arrow
      3. Delete L1 Parquet (EN Parquet kept for other pairs)
    """
    # L1 side
    l1_stats = pretokenize_language(lang, "l1", cfg)
    l1_arrow_dir = Path(cfg.output_dir) / "pretokenized" / lang
    l1_repo = cfg.hf_dataset_repo(lang, "l1")
    upload_to_hf_and_cleanup(l1_arrow_dir, l1_repo, cfg)

    # EN side
    en_stats = pretokenize_language(lang, "en", cfg)
    en_arrow_dir = Path(cfg.output_dir) / "pretokenized" / f"en_for_{lang}"
    en_repo = cfg.hf_dataset_repo(lang, "en")
    upload_to_hf_and_cleanup(en_arrow_dir, en_repo, cfg)

    # Clean up L1 Parquet (no longer needed)
    cleanup_stage2_parquet(lang, cfg)

    return l1_stats, en_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum-Mode Pretokenization (BeetleStream v2)
# ═══════════════════════════════════════════════════════════════════════════════

def _iter_indexed_shards(indexed_dir: Path, lang: str):
    """Yield (text, quality, difficulty, topic_id) from indexed Parquet shards.

    Reads from Hive-partitioned structure:
      indexed/lang={lang}/topic={id}/shard_*.parquet
    """
    lang_dir = indexed_dir / f"lang={lang}"
    if not lang_dir.exists():
        raise FileNotFoundError(f"No indexed data for lang={lang} at {lang_dir}")

    for topic_dir in sorted(lang_dir.iterdir()):
        if not topic_dir.is_dir() or not topic_dir.name.startswith("topic="):
            continue
        for shard_path in sorted(topic_dir.glob("*.parquet")):
            table = pq.read_table(str(shard_path))
            for i in range(table.num_rows):
                yield {
                    "text": table.column("text")[i].as_py(),
                    "quality": float(table.column("quality")[i].as_py()),
                    "difficulty": int(table.column("difficulty")[i].as_py()),
                    "topic_id": int(table.column("topic_id")[i].as_py()),
                }
            del table


def pretokenize_curriculum(
    lang: str,
    side: str,
    cfg: "PipelineConfig",
    target_tokens: Optional[int] = None,
) -> PretokenizationStats:
    """Pretokenize indexed shards for curriculum mode.

    Like pretokenize_language() but reads from indexed shards and produces
    Arrow datasets with extra columns: quality, difficulty, topic_id.

    Each 513-token chunk inherits quality/difficulty/topic_id from its
    source document.

    Args:
        lang: 2-letter language code.
        side: 'l1' or 'en'.
        cfg: Pipeline configuration.
        target_tokens: Max tokens to produce.
    """
    stats = PretokenizationStats(lang=lang, side=side)
    t0 = time.time()

    if side == "l1":
        source_lang = lang
        output_name = f"{lang}-curriculum"
    else:
        source_lang = "en"
        output_name = f"en-for-{lang}-curriculum"

    indexed_dir = Path(cfg.output_dir) / "indexed"
    arrow_dir = Path(cfg.output_dir) / "pretokenized" / output_name
    tok_repo = tokenizer_repo(lang, cfg.hf_user)

    log.info("Pretokenizing (curriculum) %s (side=%s)", lang, side)
    log.info("  Source: %s/lang=%s", indexed_dir, source_lang)
    log.info("  Output: %s", arrow_dir)

    if target_tokens is None:
        total_pair_tokens = 24_000_000_000
        if side == "l1":
            target_tokens = int(total_pair_tokens * cfg.l1_ratio)
        else:
            target_tokens = int(total_pair_tokens * cfg.en_ratio)

    target_chunks = target_tokens // cfg.chunk_len
    log.info("  Target: %d tokens (%d chunks)", target_tokens, target_chunks)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_repo)
    log.info("  Tokenizer loaded: %s (vocab=%d)", tok_repo, len(tokenizer))

    # Accumulate chunks with metadata
    all_input_ids = []
    all_quality = []
    all_difficulty = []
    all_topic_id = []

    buf = []  # token buffer for current document
    current_meta = {"quality": 0.0, "difficulty": 2, "topic_id": 0}

    for doc in _iter_indexed_shards(indexed_dir, source_lang):
        text = doc["text"]
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue

        current_meta = {
            "quality": doc["quality"],
            "difficulty": doc["difficulty"],
            "topic_id": doc["topic_id"],
        }

        # Pack into chunks (no cross-document bleeding)
        for tok in tokens:
            buf.append(tok)
            if len(buf) == cfg.chunk_len:
                all_input_ids.append(buf)
                all_quality.append(current_meta["quality"])
                all_difficulty.append(current_meta["difficulty"])
                all_topic_id.append(current_meta["topic_id"])
                buf = []

                stats.tokens_total += cfg.chunk_len
                if len(all_input_ids) >= target_chunks:
                    break

        # Discard remainder at document boundary (no cross-doc bleeding)
        if buf:
            stats.tokens_discarded += len(buf)
        buf = []
        stats.docs_processed += 1

        if len(all_input_ids) >= target_chunks:
            break

    # Truncate to target
    if len(all_input_ids) > target_chunks:
        all_input_ids = all_input_ids[:target_chunks]
        all_quality = all_quality[:target_chunks]
        all_difficulty = all_difficulty[:target_chunks]
        all_topic_id = all_topic_id[:target_chunks]

    stats.chunks_created = len(all_input_ids)

    # Write Arrow dataset with curriculum metadata columns
    from datasets import Dataset as HFDataset

    arrow_dir.mkdir(parents=True, exist_ok=True)

    ds = HFDataset.from_dict({
        "input_ids": all_input_ids,
        "quality": all_quality,
        "difficulty": all_difficulty,
        "topic_id": all_topic_id,
    })
    ds.save_to_disk(str(arrow_dir))

    stats.wall_time_sec = time.time() - t0
    stats_path = arrow_dir / "pretok_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info("Done %s/%s (curriculum): %d chunks (%d tokens), %d docs, %.1f min",
             lang, side, stats.chunks_created,
             stats.chunks_created * cfg.chunk_len,
             stats.docs_processed, stats.wall_time_sec / 60)

    return stats


def pretokenize_pair_curriculum(
    lang: str,
    cfg: "PipelineConfig",
) -> Tuple[PretokenizationStats, PretokenizationStats]:
    """Pretokenize both L1 and EN from indexed shards for curriculum mode."""
    l1_stats = pretokenize_curriculum(lang, "l1", cfg)
    en_stats = pretokenize_curriculum(lang, "en", cfg)

    # Upload curriculum datasets to HF
    l1_arrow_dir = Path(cfg.output_dir) / "pretokenized" / f"{lang}-curriculum"
    en_arrow_dir = Path(cfg.output_dir) / "pretokenized" / f"en-for-{lang}-curriculum"
    l1_repo = f"{cfg.hf_user}/{lang}-curriculum-{cfg.hf_dataset_suffix}"
    en_repo = f"{cfg.hf_user}/en-for-{lang}-curriculum-{cfg.hf_dataset_suffix}"

    upload_to_hf_and_cleanup(l1_arrow_dir, l1_repo, cfg)
    upload_to_hf_and_cleanup(en_arrow_dir, en_repo, cfg)

    return l1_stats, en_stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global ARROW_FLUSH_CHUNKS

    parser = argparse.ArgumentParser(
        description="Stage 3: Read clean Parquet, tokenize, write Arrow."
    )
    parser.add_argument("--lang", type=str, default=None,
                        help="Single language code to process (e.g., 'pl')")
    parser.add_argument("--side", type=str, default=None, choices=["l1", "en", "both"],
                        help="Which side to tokenize (default: both)")
    parser.add_argument("--node-id", type=int, default=None,
                        help="SLURM node ID (0-3) — processes all assigned languages")
    parser.add_argument("--output-dir", type=str, default="pipeline_output",
                        help="Base output directory (same as Stage 2)")
    parser.add_argument("--hf-user", type=str, default="Beetle-Data",
                        help="HuggingFace user/org for tokenizer repos")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="Override total target tokens per side")
    parser.add_argument("--num-workers", type=int, default=24,
                        help="Number of multiprocessing workers")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length (default: 512, chunk = seq_len + 1)")
    parser.add_argument("--flush-chunks", type=int, default=ARROW_FLUSH_CHUNKS,
                        help="Chunks to buffer before flushing to disk (default: 500K)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HuggingFace upload (keep local files)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep local files after upload")
    parser.add_argument("--dataset-suffix", type=str, default="24B",
                        help="HF dataset repo suffix (default: 24B)")
    args = parser.parse_args()

    # Update global flush threshold
    ARROW_FLUSH_CHUNKS = args.flush_chunks

    # Determine languages
    if args.lang:
        languages = [args.lang]
    elif args.node_id is not None:
        languages = langs_for_node(args.node_id)
    else:
        parser.error("Must specify --lang or --node-id")
        return

    cfg = PipelineConfig(
        output_dir=args.output_dir,
        hf_user=args.hf_user,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        chunk_len=args.seq_len + 1,
        upload_to_hf=not args.no_upload,
        cleanup_after_upload=not args.no_cleanup,
        cleanup_stage2_after_pretok=not args.no_cleanup,
        hf_dataset_suffix=args.dataset_suffix,
    )

    all_stats = {}
    for lang in languages:
        if lang not in LANG_REGISTRY:
            log.error("Unknown language: %s", lang)
            continue

        if lang == "en":
            log.info("Skipping 'en' — it is processed as part of each bilingual pair")
            continue

        side = args.side or "both"
        if side == "both":
            l1_s, en_s = pretokenize_pair(lang, cfg)
            all_stats[f"{lang}_l1"] = l1_s.to_dict()
            all_stats[f"{lang}_en"] = en_s.to_dict()
        elif side == "l1":
            s = pretokenize_language(lang, "l1", cfg,
                                     target_tokens=args.target_tokens)
            all_stats[f"{lang}_l1"] = s.to_dict()
        else:
            s = pretokenize_language(lang, "en", cfg,
                                     target_tokens=args.target_tokens)
            all_stats[f"{lang}_en"] = s.to_dict()

    # Write combined stats
    combined_path = Path(args.output_dir) / "pretokenization_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    log.info("All done. Summary written to %s", combined_path)


if __name__ == "__main__":
    main()
