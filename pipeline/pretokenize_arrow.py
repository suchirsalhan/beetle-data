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
    TOKENIZER_VOCAB_SIZE,
    TOKENIZER_TRAINING_SENTENCES,
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
# Resumable per-shard checkpointing + idempotent HF push
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mirrors the Stage 2 contract in pipeline/decontaminate_stream.py:
#   * Atomic JSON checkpoint written after each shard / part.
#   * Per-part HF upload via HfApi.create_commit (idempotent: list_repo_files
#     filters anything already on the remote).
#   * Finalization writes a sentinel marker file (`data/_finalized.json`) on the
#     remote so a relaunch on a completed repo no-ops.

def _pretok_checkpoint_path(arrow_dir: Path, output_name: str) -> Path:
    return arrow_dir.parent / f"pretok_checkpoint_{output_name}.json"


def _load_pretok_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to load pretok checkpoint %s: %s", path, e)
        return {}


def _save_pretok_checkpoint_atomic(path: Path, ckpt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)


def _list_remote_files(repo_id: str, cfg: "PipelineConfig") -> set:
    """List files on a HF dataset repo. Returns empty set if repo absent."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi(token=cfg.hf_token or None)
    try:
        return set(api.list_repo_files(repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return set()
    except Exception as e:
        log.warning("list_repo_files failed for %s: %s", repo_id, e)
        return set()


def _upload_pretok_part(
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    cfg: "PipelineConfig",
    include_readme: bool,
) -> bool:
    """Upload one parquet part to the HF dataset repo as an atomic commit.

    On the first commit (`include_readme=True`), also publishes a minimal
    README so `load_dataset(repo_id, streaming=True)` discovers the parts.
    """
    from huggingface_hub import HfApi, CommitOperationAdd

    api = HfApi(token=cfg.hf_token or None)
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        log.warning("Could not create repo %s: %s", repo_id, e)
        return False

    ops = [CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=str(local_path),
    )]
    if include_readme:
        readme = (
            "---\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files: 'data/*.parquet'\n"
            "---\n"
            f"# {repo_id}\n\n"
            "Pretokenized chunks (`input_ids` = 513-token packed sequences,\n"
            "no cross-document bleeding). Sharded incrementally; the marker\n"
            "`data/_finalized.json` is committed once all parts are uploaded.\n"
        )
        ops.append(CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme.encode("utf-8"),
        ))

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Add {path_in_repo}",
        )
    except Exception as e:
        log.warning("Upload commit failed for %s/%s: %s", repo_id, path_in_repo, e)
        return False
    return True


def _finalize_pretok_repo(
    repo_id: str,
    cfg: "PipelineConfig",
    stats_dict: dict,
) -> bool:
    """Write `data/_finalized.json` to signal completion. Idempotent."""
    from huggingface_hub import HfApi, CommitOperationAdd

    if not cfg.upload_to_hf:
        return True

    api = HfApi(token=cfg.hf_token or None)
    existing = _list_remote_files(repo_id, cfg)
    if "data/_finalized.json" in existing:
        log.info("Repo %s already finalized — skipping marker", repo_id)
        return True

    marker = json.dumps(stats_dict, indent=2).encode("utf-8")
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[CommitOperationAdd(
                path_in_repo="data/_finalized.json",
                path_or_fileobj=marker,
            )],
            commit_message="Finalize: all parts uploaded",
        )
    except Exception as e:
        log.warning("Finalize commit failed for %s: %s", repo_id, e)
        return False
    log.info("Finalized repo %s", repo_id)
    return True


def ensure_tokenizer(tok_repo: str, lang: str, hf_user: str) -> None:
    """Check if the tokenizer exists on HuggingFace; train and push if missing.

    Must be called on the main process BEFORE spawning the multiprocessing Pool,
    since worker processes cannot train tokenizers.
    """
    from transformers import PreTrainedTokenizerFast

    try:
        PreTrainedTokenizerFast.from_pretrained(tok_repo)
        log.info("Tokenizer found: %s", tok_repo)
    except OSError:
        log.warning(
            "Tokenizer %s not found on HuggingFace. Training automatically...",
            tok_repo,
        )
        import importlib.util
        tok_path = Path(__file__).resolve().parents[1] / "tok" / "multi-train-tok.py"
        spec = importlib.util.spec_from_file_location("multi_train_tok", tok_path)
        multi_train_tok = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(multi_train_tok)
        multi_train_tok.train_and_push(
            lang=lang,
            hf_user=hf_user,
            vocab_size=TOKENIZER_VOCAB_SIZE,
            n_sentences=TOKENIZER_TRAINING_SENTENCES,
        )
        log.info("Tokenizer trained and pushed: %s", tok_repo)


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


class ResumablePartWriter:
    """Buffer chunks; flush each input shard's chunks as a parquet part and push to HF.

    Part files are named `data/{shard_stem}-{sub_idx:02d}.parquet` so that
    re-processing an interrupted shard overwrites the previous attempt's
    parts deterministically (HF `create_commit` replaces files with the
    same `path_in_repo`). This avoids duplicate data when the wallclock
    kills the job between flushing and `mark_shard_done`.

    Per-shard sequence (driven by ``pretokenize_language``):
      1. ``start_shard(basename)`` — sets the active shard stem and resets
          the per-shard sub_idx to 0.
      2. ``add_chunks(...)`` repeatedly as worker results arrive. If the
          buffer crosses ``flush_threshold`` mid-shard, an interim part is
          flushed (rare with default settings: shard ≪ threshold).
      3. ``flush_now()`` — emits a final per-shard part for any buffered
          tail. One part per shard with default settings.
      4. ``mark_shard_done(basename)`` — appends to ``parquet_shards_done``.

    Resume contract:
      * On startup, any leftover staging files are uploaded before resuming
        (handles a crash between local write and remote commit).
      * ``parquet_shards_done`` lets ``pretokenize_language`` skip already-
        completed input shards.
      * ``chunks_committed`` / ``target_chunks`` enable a clean stop when
        the per-side token budget is reached.

    If ``cfg.upload_to_hf`` is False, this writer falls back to local-only
    behavior: parquet parts are kept in ``<arrow_dir>/data/`` so downstream
    code can still load them with
    ``load_dataset("parquet", data_files=...)``.
    """

    def __init__(
        self,
        arrow_dir: Path,
        repo_id: str,
        output_name: str,
        cfg: "PipelineConfig",
        flush_threshold: int,
        target_chunks: int,
    ):
        self.arrow_dir = Path(arrow_dir)
        self.repo_id = repo_id
        self.output_name = output_name
        self.cfg = cfg
        self.flush_threshold = max(1, int(flush_threshold))
        self.target_chunks = int(target_chunks)
        self.upload_enabled = bool(getattr(cfg, "upload_to_hf", True))

        # Staging dir holds parquet files between local write and HF upload.
        # When upload is disabled, parts land directly in `data/` for local use.
        if self.upload_enabled:
            self.local_dir = self.arrow_dir / "_staging"
        else:
            self.local_dir = self.arrow_dir / "data"
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = _pretok_checkpoint_path(self.arrow_dir, output_name)
        ckpt = _load_pretok_checkpoint(self.checkpoint_path)
        self.parquet_shards_done: List[str] = list(ckpt.get("parquet_shards_done", []))
        self.parts_uploaded: List[str] = list(ckpt.get("parts_uploaded", []))
        self.chunks_committed: int = int(ckpt.get("chunks_committed", 0))
        # Mirror stats so resume preserves cumulative counters.
        self.tokens_total: int = int(ckpt.get("tokens_total", 0))
        self.docs_processed: int = int(ckpt.get("docs_processed", 0))
        self.tokens_discarded: int = int(ckpt.get("tokens_discarded", 0))
        self.shards_read: int = int(ckpt.get("shards_read", 0))

        self.buffer: List[List[int]] = []
        # Per-shard cursor (set by start_shard()).
        self._current_shard_stem: Optional[str] = None
        self._current_sub_idx: int = 0

        if self.upload_enabled:
            self._drain_staging_on_start()

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        ckpt = {
            "parquet_shards_done": self.parquet_shards_done,
            "parts_uploaded": self.parts_uploaded,
            "chunks_committed": self.chunks_committed,
            "target_chunks": self.target_chunks,
            "tokens_total": self.tokens_total,
            "docs_processed": self.docs_processed,
            "tokens_discarded": self.tokens_discarded,
            "shards_read": self.shards_read,
        }
        _save_pretok_checkpoint_atomic(self.checkpoint_path, ckpt)

    # ------------------------------------------------------------------
    # Crash-recovery: upload anything left in staging before resuming
    # ------------------------------------------------------------------

    def _drain_staging_on_start(self) -> None:
        leftover = sorted(self.local_dir.glob("*.parquet"))
        if not leftover:
            return
        log.info("Found %d staged parts in %s — uploading before resuming",
                 len(leftover), self.local_dir)
        remote = _list_remote_files(self.repo_id, self.cfg)
        for staged in leftover:
            path_in_repo = f"data/{staged.name}"
            include_readme = (
                not any(f.startswith("README") for f in remote)
                and not self.parts_uploaded
            )
            ok = _upload_pretok_part(
                staged, path_in_repo, self.repo_id, self.cfg, include_readme,
            )
            if not ok:
                raise RuntimeError(
                    f"Failed to upload staged part {staged} to {self.repo_id} — "
                    f"local file kept for next restart"
                )
            remote.add(path_in_repo)
            if path_in_repo not in self.parts_uploaded:
                self.parts_uploaded.append(path_in_repo)
            try:
                staged.unlink()
            except OSError:
                pass
        self._save_checkpoint()

    # ------------------------------------------------------------------
    # Public API used by pretokenize_language
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self.target_chunks > 0 and self.chunks_committed >= self.target_chunks

    def shard_is_done(self, shard_basename: str) -> bool:
        return shard_basename in self.parquet_shards_done

    def start_shard(self, shard_basename: str) -> None:
        """Begin a new input shard. Resets the per-shard sub-index to 0.

        Subsequent flushes write `data/{shard_stem}-{sub_idx:02d}.parquet`
        with sub_idx counting from zero. Re-processing the same shard on
        resume produces deterministic file names and overwrites the
        previous attempt's parts (HF `create_commit` replaces by path).
        """
        # Strip the .parquet extension once; tolerate any other suffix.
        self._current_shard_stem = Path(shard_basename).stem
        self._current_sub_idx = 0

    def add_chunks(self, chunks: List[List[int]]) -> bool:
        """Add chunks to the buffer; auto-flush only if the soft threshold is
        exceeded mid-shard (rare).

        Returns True if the per-side target was reached (caller should stop).
        """
        if not chunks:
            return self.is_done()

        # Truncate to remaining budget so we never overshoot target_chunks.
        already = self.chunks_committed + len(self.buffer)
        remaining = self.target_chunks - already
        if remaining <= 0:
            return True
        if len(chunks) > remaining:
            chunks = chunks[:remaining]

        self.buffer.extend(chunks)
        # Soft fallback: only kicks in if a single input shard yields more
        # chunks than the buffer cap. With default config (shard=50K docs,
        # threshold=500K chunks) this branch never executes.
        while len(self.buffer) >= self.flush_threshold:
            head = self.buffer[: self.flush_threshold]
            self.buffer = self.buffer[self.flush_threshold:]
            self._flush(head)
            if self.is_done():
                self.buffer.clear()
                return True
        return self.is_done()

    def flush_now(self) -> None:
        """Emit a part for whatever is in the buffer, even if below threshold."""
        if self.buffer:
            head = self.buffer
            self.buffer = []
            self._flush(head)

    def mark_shard_done(self, shard_basename: str) -> None:
        if shard_basename not in self.parquet_shards_done:
            self.parquet_shards_done.append(shard_basename)
            self._save_checkpoint()

    def update_stats(self, *, tokens: int = 0, docs: int = 0,
                     discarded: int = 0, shards: int = 0) -> None:
        self.tokens_total += int(tokens)
        self.docs_processed += int(docs)
        self.tokens_discarded += int(discarded)
        self.shards_read += int(shards)

    # ------------------------------------------------------------------
    # Internal flush
    # ------------------------------------------------------------------

    def _flush(self, chunks: List[List[int]]) -> None:
        if not chunks:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq_writer

        if self._current_shard_stem is None:
            # Caller forgot to call start_shard(); fall back to a generic name.
            stem = f"orphan-{len(self.parts_uploaded):05d}"
        else:
            stem = self._current_shard_stem
        part_name = f"{stem}-{self._current_sub_idx:02d}.parquet"
        local_path = self.local_dir / part_name
        path_in_repo = f"data/{part_name}"

        table = pa.Table.from_pydict({"input_ids": chunks})
        pq_writer.write_table(table, str(local_path))

        if self.upload_enabled:
            include_readme = not self.parts_uploaded
            ok = _upload_pretok_part(
                local_path, path_in_repo, self.repo_id, self.cfg, include_readme,
            )
            if not ok:
                raise RuntimeError(
                    f"Upload failed for {path_in_repo}; staged at {local_path} — "
                    f"will be retried on next supervisor restart"
                )

        if path_in_repo not in self.parts_uploaded:
            self.parts_uploaded.append(path_in_repo)
        self.chunks_committed += len(chunks)
        self._current_sub_idx += 1
        self._save_checkpoint()

        if self.upload_enabled:
            try:
                local_path.unlink()
            except OSError:
                pass
            log.info(
                "  Pushed %s: %d chunks (committed %d / %d)",
                path_in_repo, len(chunks),
                self.chunks_committed, self.target_chunks,
            )
        else:
            log.info(
                "  Wrote %s: %d chunks (committed %d / %d)",
                local_path, len(chunks),
                self.chunks_committed, self.target_chunks,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Parquet Shard Iterator (memory-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def _iter_parquet_shards(parquet_dir: Path, skip_basenames: Optional[set] = None):
    """Yield (shard_path, texts) for each Parquet shard, one at a time.

    Only one shard's text data is in memory at any point.
    Each shard is ~50K docs * ~5 KB avg = ~250 MB in RAM.

    Shards whose filename is in `skip_basenames` are filtered out before
    being read, so resume after a wallclock kill never re-reads a shard
    whose chunks are already on the destination HF repo.
    """
    shard_files = sorted(parquet_dir.glob("*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No Parquet shards found in {parquet_dir}")

    skip = skip_basenames or set()
    n_total = len(shard_files)
    shard_files = [p for p in shard_files if p.name not in skip]
    if skip:
        log.info(
            "Found %d Parquet shards in %s (%d skipped via checkpoint)",
            len(shard_files), parquet_dir, n_total - len(shard_files),
        )
    else:
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

    # Auto-detect: train tokenizer if not found on HuggingFace
    ensure_tokenizer(tok_repo, lang, cfg.hf_user)

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

    # Resumable per-shard writer with periodic HF push (idempotent on restart).
    # Each input Parquet shard, once fully tokenized, contributes one or more
    # parquet "parts" pushed atomically to the destination HF dataset repo;
    # the local checkpoint records done-shards + uploaded parts so a wallclock
    # kill + re-launch picks up where it left off.
    repo_id = cfg.hf_dataset_repo(lang, side)
    writer = ResumablePartWriter(
        arrow_dir=arrow_dir,
        repo_id=repo_id,
        output_name=output_name,
        cfg=cfg,
        flush_threshold=ARROW_FLUSH_CHUNKS,
        target_chunks=target_chunks,
    )

    # Seed stats from any prior progress so resume reports cumulative counters.
    stats.tokens_total = writer.tokens_total
    stats.docs_processed = writer.docs_processed
    stats.tokens_discarded = writer.tokens_discarded
    stats.shards_read = writer.shards_read

    if writer.is_done():
        log.info(
            "Pretokenization for %s/%s already complete per checkpoint "
            "(%d / %d chunks committed) — skipping to finalize",
            lang, side, writer.chunks_committed, target_chunks,
        )
        stats.chunks_created = writer.chunks_committed
        stats.arrow_parts_written = writer.next_part_idx
        stats.wall_time_sec = time.time() - t0
        _finalize_pretok_repo(repo_id, cfg, stats.to_dict())
        return stats

    # Process Parquet shards one at a time
    num_workers = min(cfg.num_workers, cpu_count())
    batch_size = cfg.batch_size
    reached_target = False

    pbar = tqdm(
        total=target_tokens,
        initial=writer.tokens_total,
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

        skip = set(writer.parquet_shards_done)
        for shard_path, texts in _iter_parquet_shards(parquet_dir, skip_basenames=skip):
            if reached_target:
                break

            shard_basename = shard_path.name
            writer.start_shard(shard_basename)
            stats.shards_read += 1

            # Split this shard's texts into batches for parallel tokenization
            text_batches = [
                texts[i:i + batch_size]
                for i in range(0, len(texts), batch_size)
            ]
            del texts  # free shard text memory before processing

            shard_tokens = 0
            shard_discarded = 0
            shard_docs = 0
            for result in pool.imap(_process_text_batch, text_batches, chunksize=1):
                chunks, tok_count, disc_count = result
                shard_tokens += tok_count
                shard_discarded += disc_count
                shard_docs += batch_size

                stats.tokens_total += tok_count
                stats.tokens_discarded += disc_count
                stats.docs_processed += batch_size
                pbar.update(tok_count)

                if writer.add_chunks(chunks):
                    reached_target = True
                    break

            # Persist any partial buffer for this shard so the part contains
            # only data from completed shards (clean per-shard resume).
            writer.flush_now()
            writer.update_stats(
                tokens=shard_tokens,
                docs=shard_docs,
                discarded=shard_discarded,
                shards=1,
            )
            writer.mark_shard_done(shard_basename)

    pbar.close()

    # Final flush + push any remaining buffered chunks (no-op if reached_target
    # already drained the buffer).
    writer.flush_now()

    stats.chunks_created = writer.chunks_committed
    stats.arrow_parts_written = writer.next_part_idx
    stats.wall_time_sec = time.time() - t0

    # Save stats locally (matches old behavior; useful even when uploads happen).
    arrow_dir.mkdir(parents=True, exist_ok=True)
    stats_path = arrow_dir / "pretok_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Publish the finalize marker on the HF repo so a re-launch detects
    # completion and short-circuits.
    _finalize_pretok_repo(repo_id, cfg, stats.to_dict())

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

    from datasets import load_from_disk

    try:
        log.info("Uploading %s to %s ...", arrow_dir, repo_id)
        ds = load_from_disk(str(arrow_dir))
        ds.push_to_hub(repo_id, token=cfg.hf_token, private=False)
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
    """Pretokenize both L1 and EN sides; resume-safe across wallclock kills.

    Each call to ``pretokenize_language`` streams its parts to the
    destination HF repo as they are produced (see ``ResumablePartWriter``)
    and writes ``data/_finalized.json`` once the per-side target is reached,
    so there is no separate end-of-stage upload step. Re-running this
    function on a partially-completed pair resumes exactly where the last
    invocation stopped.
    """
    # L1 side — incremental push to cfg.hf_dataset_repo(lang, 'l1')
    l1_stats = pretokenize_language(lang, "l1", cfg)
    l1_arrow_dir = Path(cfg.output_dir) / "pretokenized" / lang
    if cfg.cleanup_after_upload and cfg.upload_to_hf:
        shutil.rmtree(l1_arrow_dir, ignore_errors=True)

    # EN side — incremental push to cfg.hf_dataset_repo(lang, 'en')
    en_stats = pretokenize_language(lang, "en", cfg)
    en_arrow_dir = Path(cfg.output_dir) / "pretokenized" / f"en_for_{lang}"
    if cfg.cleanup_after_upload and cfg.upload_to_hf:
        shutil.rmtree(en_arrow_dir, ignore_errors=True)

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

    # Auto-detect: train tokenizer if not found on HuggingFace
    ensure_tokenizer(tok_repo, lang, cfg.hf_user)

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

    # Disable HF tokenizer internal thread pool before any multiprocessing Pool
    # spawns workers — prevents the "current process just got forked" warnings
    # and rank-style deadlocks when running with a high NUM_WORKERS.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of multiprocessing workers "
                             "(default: auto-detect = min(cpu_count - 4, 64))")
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

    # Only pass num_workers when explicitly set, else use PipelineConfig auto-detect
    cfg_kwargs = dict(
        output_dir=args.output_dir,
        hf_user=args.hf_user,
        seq_len=args.seq_len,
        chunk_len=args.seq_len + 1,
        upload_to_hf=not args.no_upload,
        cleanup_after_upload=not args.no_cleanup,
        cleanup_stage2_after_pretok=not args.no_cleanup,
        hf_dataset_suffix=args.dataset_suffix,
    )
    if args.num_workers is not None:
        cfg_kwargs["num_workers"] = args.num_workers
    cfg = PipelineConfig(**cfg_kwargs)

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
