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
    STREAM_TOKENS_PER_LANG,
    TARGET_TOKENS_PER_LANG,
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
    words_accumulated: int = 0   # clean words written to Parquet
    shards_written: int = 0
    wall_time_sec: float = 0.0
    stream_token_target: int = STREAM_TOKENS_PER_LANG
    train_token_target: int = TARGET_TOKENS_PER_LANG

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
            "stream_token_target": self.stream_token_target,
            "train_token_target": self.train_token_target,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Worker: Decontamination Check (runs in forked processes)
# ═══════════════════════════════════════════════════════════════════════════════

# Global in each worker process (set once via fork inheritance)
_worker_index: Optional[BenchmarkIndex] = None
_worker_min_chars: int = 200
_worker_max_chars: int = 100_000
_worker_text_field: str = "text"


def _init_worker(index: BenchmarkIndex, min_chars: int, max_chars: int,
                 text_field: str = "text"):
    """Initializer for worker processes — receives the shared index."""
    global _worker_index, _worker_min_chars, _worker_max_chars, _worker_text_field
    _worker_index = index
    _worker_min_chars = min_chars
    _worker_max_chars = max_chars
    _worker_text_field = text_field


def _process_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single document: clean, filter, decontaminate.

    Returns the cleaned doc dict if it passes all checks, else None.
    The returned dict includes a 'status' field:
      - 'clean': passed all checks
      - 'too_short' / 'too_long': length filter
      - 'contaminated': contains benchmark n-grams

    Note: _worker_text_field controls which dict key holds the document text.
    Set via _init_worker to support datasets that use fields other than "text".
    """
    text = doc.get(_worker_text_field, "") or doc.get("text", "")
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
        # Keep manifest on disk in sync with shards so a crash between shard
        # flushes still leaves a consistent snapshot for resume.
        self.save_manifest()
        return str(path)

    def save_manifest(self) -> str:
        """Write manifest.json mapping shard files to doc_id ranges."""
        manifest_path = self.output_dir / f"{self.lang}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return str(manifest_path)

    def save_checkpoint(self, stats: "DecontaminationStats") -> str:
        """Persist resume state after a successful batch cycle.

        Written atomically (write-to-tmp + rename) so a crash mid-write
        can't leave a truncated JSON file. Resume uses stats.docs_streamed
        as the ds.skip(...) offset.
        """
        cp_path = self.output_dir / f"{self.lang}_checkpoint.json"
        tmp_path = cp_path.with_suffix(".json.tmp")
        payload = {
            "shard_idx": self.shard_idx,
            "next_doc_id": self.next_doc_id,
            "stats": stats.to_dict(),
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, cp_path)
        return str(cp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Optional lightweight memory logger (gated on BEETLE_MEMLOG=1)
# ═══════════════════════════════════════════════════════════════════════════════

def _start_rss_logger(out_dir: Path, lang: str, interval: float = 30.0):
    """Start a daemon thread that samples parent+worker RSS to a TSV.

    No-op unless the env var BEETLE_MEMLOG=1 is set. Safe to call from a
    short-lived process: the thread is daemonic, so it dies with the process.
    Silently skips if psutil is not installed.
    """
    if os.environ.get("BEETLE_MEMLOG") != "1":
        return None
    try:
        import psutil  # type: ignore
    except ImportError:
        log.warning("BEETLE_MEMLOG=1 set but psutil not installed; skipping")
        return None

    import threading

    memlog_path = out_dir / f"{lang}_memlog.tsv"
    parent_pid = os.getpid()
    header_written = memlog_path.exists()

    def _run():
        nonlocal header_written
        try:
            parent = psutil.Process(parent_pid)
        except psutil.Error:
            return
        while True:
            try:
                procs = [parent] + parent.children(recursive=False)
                ts = time.time()
                with open(memlog_path, "a") as f:
                    if not header_written:
                        f.write(
                            "ts\tpid\trole\trss_mb\tvms_mb\tnum_fds\tnum_threads\n"
                        )
                        header_written = True
                    for i, p in enumerate(procs):
                        try:
                            mi = p.memory_info()
                            fds = p.num_fds() if hasattr(p, "num_fds") else -1
                            nth = p.num_threads()
                            role = "parent" if i == 0 else "worker"
                            f.write(
                                f"{ts:.0f}\t{p.pid}\t{role}\t"
                                f"{mi.rss / 1e6:.1f}\t{mi.vms / 1e6:.1f}\t"
                                f"{fds}\t{nth}\n"
                            )
                        except psutil.Error:
                            continue
            except Exception:
                # Never let the logger crash the pipeline
                pass
            time.sleep(interval)

    t = threading.Thread(target=_run, name="beetle-memlog", daemon=True)
    t.start()
    log.info("RSS logger active: %s (every %.0fs)", memlog_path, interval)
    return t


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
    stats = DecontaminationStats(
        lang=lang,
        stream_token_target=cfg.stream_words_per_lang,  # store word count for reporting
        train_token_target=getattr(cfg, "target_tokens_per_lang", TARGET_TOKENS_PER_LANG),
    )
    t0 = time.time()

    # Output directory
    out_dir = Path(cfg.output_dir) / "decontaminated" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional per-PID RSS logger (no-op unless BEETLE_MEMLOG=1)
    _start_rss_logger(out_dir, lang)

    # Word target: stream until this many CLEAN words are accumulated.
    # Derived from STREAM_TOKENS_PER_LANG ÷ TOKENS_TO_WORDS_RATIO (≈ 21.5B words = 28B tokens).
    target_words = cfg.stream_words_per_lang

    log.info("Starting decontamination for %s (%s)", lang, lc.name)
    log.info("  Stream target: %d clean words (~%dB tokens)",
             target_words, STREAM_TOKENS_PER_LANG // 1_000_000_000)
    log.info("  Training target: %dB tokens (applied at pretokenization)",
             getattr(cfg, "target_tokens_per_lang", TARGET_TOKENS_PER_LANG) // 1_000_000_000)
    log.info("  Source dataset: %s",
             cfg.training_dataset_en if lc.is_english else cfg.training_dataset_l1)

    # Shard writer
    writer = ShardWriter(out_dir, lang, shard_size=cfg.shard_size)

    # Resume from checkpoint if one exists from a previous (crashed) run.
    # stats.docs_streamed doubles as the ds.skip(...) offset on resume.
    checkpoint_path = out_dir / f"{lang}_checkpoint.json"
    manifest_path = out_dir / f"{lang}_manifest.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                cp = json.load(f)
            for k, v in cp.get("stats", {}).items():
                if hasattr(stats, k):
                    setattr(stats, k, v)
            writer.shard_idx = int(cp.get("shard_idx", 0))
            writer.next_doc_id = int(cp.get("next_doc_id", 0))
            if manifest_path.exists():
                with open(manifest_path) as f:
                    writer.manifest = {
                        k: tuple(v) for k, v in json.load(f).items()
                    }
            log.info("Resuming %s from %d streamed docs, shard %d (accumulated %d words)",
                     lang, stats.docs_streamed, writer.shard_idx, stats.words_accumulated)
        except Exception as e:
            log.warning("Failed to load checkpoint for %s (%s); starting fresh",
                        lang, e)
            stats = DecontaminationStats(
                lang=lang,
                stream_token_target=cfg.stream_words_per_lang,
                train_token_target=getattr(cfg, "target_tokens_per_lang", TARGET_TOKENS_PER_LANG),
            )
            writer = ShardWriter(out_dir, lang, shard_size=cfg.shard_size)

    skip_count = stats.docs_streamed

    # Stream dataset (configurable source — change TRAINING_DATASET_* in config.py)
    def _open_stream(skip: int):
        if lc.is_english:
            d = load_dataset(cfg.training_dataset_en, split="train", streaming=True)
        else:
            d = load_dataset(cfg.training_dataset_l1, name=lc.fw2_name,
                             split="train", streaming=True)
        return d.skip(skip) if skip else d

    ds = _open_stream(skip_count)

    # Progress bar — seed with any already-accumulated words so tqdm reflects
    # resumed progress rather than restarting at 0.
    pbar = tqdm(
        total=target_words,
        initial=stats.words_accumulated,
        unit="words",
        desc=f"Decontaminating {lang}",
        unit_scale=True,
        dynamic_ncols=True,
    )

    # Process with multiprocessing pool
    num_workers = min(cfg.num_workers, cpu_count())
    batch: List[Dict] = []

    text_field = getattr(cfg, "training_text_field", "text")

    # Periodic HF upload cadence (in shards). Every SHARDS_PER_UPLOAD new
    # shards we push everything in out_dir to the raw repo so a crash never
    # loses already-committed work. upload_large_folder is resumable — files
    # that already match the remote are skipped.
    SHARDS_PER_UPLOAD = int(os.environ.get("BEETLE_SHARDS_PER_UPLOAD", "20"))
    upload_enabled = (
        getattr(cfg, "upload_raw_parquet", False) and cfg.upload_to_hf
    )
    last_uploaded_shard_idx = writer.shard_idx

    def _consume_batch(batch_docs: List[Dict], pool) -> None:
        """Dispatch a batch to workers and fold results into stats + writer."""
        sub_batches = [
            batch_docs[i:i + cfg.batch_size]
            for i in range(0, len(batch_docs), cfg.batch_size)
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

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(index, cfg.min_doc_chars, cfg.max_doc_chars, text_field),
        maxtasksperchild=200,
    ) as pool:

        # Fail-fast: on any stream error we flush buffered docs, commit a
        # checkpoint + manifest, optionally push partial shards to HF, then
        # raise. The outer supervisor (launch_full_pipeline.sh retry loop)
        # re-invokes Stage 2 in a fresh process, which resumes cleanly via
        # the checkpoint-loading block above — this is the only reliable way
        # to release leaked aiohttp/httpx connectors and `datasets` state
        # that accumulate when retrying in-place inside one long-lived
        # process (see plan: typed-discovering-swing.md).
        try:
            for entry in ds:
                batch.append({
                    text_field: entry.get(text_field, "") or entry.get("text", ""),
                    "url": entry.get("url", ""),
                })

                if len(batch) >= cfg.batch_size * num_workers:
                    _consume_batch(batch, pool)
                    batch.clear()

                    # Persist resume state after every full batch cycle.
                    writer.save_checkpoint(stats)

                    # Periodic HF upload: every SHARDS_PER_UPLOAD new shards,
                    # push the out_dir so committed work survives a crash.
                    if (
                        upload_enabled
                        and writer.shard_idx - last_uploaded_shard_idx
                        >= SHARDS_PER_UPLOAD
                    ):
                        try:
                            _upload_raw_parquet(out_dir, lang, cfg)
                            last_uploaded_shard_idx = writer.shard_idx
                        except Exception as up_e:
                            log.warning(
                                "Periodic upload for %s failed (will retry "
                                "next cycle): %s", lang, up_e,
                            )

                    # Check if we've reached the word target
                    if stats.words_accumulated >= target_words:
                        break
        except BaseException as stream_exc:
            # Ensure buffered docs become a real shard and manifest on disk
            # before the process exits, so the resume path finds them.
            log.error(
                "Stream error for %s at %d streamed docs (%s: %s) — "
                "flushing %d buffered docs and exiting for supervisor restart",
                lang, stats.docs_streamed, type(stream_exc).__name__,
                stream_exc, len(batch),
            )
            try:
                if batch:
                    _consume_batch(batch, pool)
                    batch.clear()
                writer.finalize()
                writer.save_manifest()
                writer.save_checkpoint(stats)
                if upload_enabled and writer.shard_idx > last_uploaded_shard_idx:
                    try:
                        _upload_raw_parquet(out_dir, lang, cfg)
                    except Exception as up_e:
                        log.warning(
                            "Crash-path upload for %s failed: %s", lang, up_e,
                        )
            except Exception as flush_e:
                log.error("Crash-path flush for %s failed: %s", lang, flush_e)
            raise

        # Process remaining batch
        if batch and stats.words_accumulated < target_words:
            _consume_batch(batch, pool)
            batch.clear()
            writer.save_checkpoint(stats)

    pbar.close()

    # Clean up the streaming dataset iterator to avoid orphaned httpx/aiohttp
    # background threads that cause PyGILState_Release crashes during shutdown.
    del ds

    # Finalize
    writer.finalize()
    manifest_path = writer.save_manifest()
    stats.shards_written = writer.shard_idx
    stats.wall_time_sec = time.time() - t0

    # Save stats
    stats_path = out_dir / f"{lang}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Successful completion → remove checkpoint so the next run starts fresh
    # instead of spuriously resuming.
    try:
        (out_dir / f"{lang}_checkpoint.json").unlink(missing_ok=True)
    except OSError:
        pass

    log.info("Done %s: %d clean docs, %d contaminated (%.2f%%), %d shards, %.1f min",
             lang, stats.docs_clean, stats.docs_contaminated,
             stats.to_dict()["contamination_rate"] * 100,
             stats.shards_written, stats.wall_time_sec / 60)

    # Optional: upload raw decontaminated Parquet to HuggingFace.
    # Required for curriculum mode Stages D+3 when running on separate nodes without
    # a shared filesystem. Enable via cfg.upload_raw_parquet = True (or
    # --curriculum-prep flag in launch_full_pipeline.sh).
    if getattr(cfg, "upload_raw_parquet", False) and cfg.upload_to_hf:
        _upload_raw_parquet(out_dir, lang, cfg)

    return stats


def _upload_raw_parquet(out_dir: "Path", lang: str, cfg: "PipelineConfig") -> None:
    """Upload raw decontaminated Parquet shards to HuggingFace in one batched commit.

    Uploads to: {hf_user}/{lang}-raw-{hf_dataset_suffix}
    This allows curriculum Stage D nodes to download raw shards without
    needing a shared filesystem with the decontamination nodes.

    Uses `upload_large_folder` (huggingface_hub >= 0.26) so the entire folder is
    pushed in a small number of batched commits (not one commit per shard),
    staying well under HF's 128-commits-per-hour per-repo rate limit.
    Falls back to `upload_folder` on older hub versions.
    """
    from huggingface_hub import HfApi

    raw_repo = cfg.hf_raw_parquet_repo(lang)
    api = HfApi(token=cfg.hf_token or None)

    # Ensure the repo exists
    try:
        api.create_repo(raw_repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        log.warning("Could not create repo %s: %s", raw_repo, e)
        return

    parquet_files = sorted(out_dir.glob("*.parquet"))
    allow_patterns = [
        "*.parquet",
        f"{lang}_manifest.json",
        f"{lang}_stats.json",
    ]

    log.info(
        "Uploading %d raw Parquet shards to %s via upload_large_folder ...",
        len(parquet_files), raw_repo,
    )

    try:
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(
                folder_path=str(out_dir),
                repo_id=raw_repo,
                repo_type="dataset",
                allow_patterns=allow_patterns,
            )
        else:
            # Fallback for older huggingface_hub — still batches into one commit
            log.info(
                "  upload_large_folder unavailable; falling back to upload_folder"
            )
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=raw_repo,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                commit_message=f"Upload raw {lang} Parquet shards ({len(parquet_files)} files)",
            )
    except Exception as e:
        log.warning("Batched upload failed for %s: %s", raw_repo, e)
        return

    log.info("Raw Parquet upload complete → %s", raw_repo)


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
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of multiprocessing workers "
                             "(default: auto-detect = min(cpu_count - 4, 64))")
    parser.add_argument("--shard-size", type=int, default=50_000,
                        help="Documents per Parquet shard")
    parser.add_argument("--upload-raw-parquet", action="store_true",
                        help="Upload raw decontaminated Parquet to HF after writing "
                             "(required for modular curriculum D+3 runs)")
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

    # Build config — only pass num_workers when explicitly set, else use auto-detect
    cfg_kwargs = dict(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        upload_raw_parquet=args.upload_raw_parquet,
    )
    if args.num_workers is not None:
        cfg_kwargs["num_workers"] = args.num_workers
    cfg = PipelineConfig(**cfg_kwargs)
    if args.target_words:
        cfg.stream_words_per_lang = args.target_words

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
