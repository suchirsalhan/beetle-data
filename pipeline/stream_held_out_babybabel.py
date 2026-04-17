"""stream_held_out_babybabel.py — Held-out BabyBabel corpus for AoA evaluation.

For BabyBabel bilingual training we consume ``Beetle-Data/BabyBabel-{lang}-50M``.
The held-out complement is any document that lives in
``Beetle-Data/BabyBabel-{lang}-100M`` but NOT in ``-50M``.

Because the two HF datasets are curated independently (not guaranteed prefix /
suffix), we dedup by document-text hash: build a hash set from every doc in
``-50M``, then stream ``-100M`` and emit any doc whose hash is absent.

Output schema matches ``stream_held_out.py`` exactly so the downstream
`beetle-analyze/aoa/prepare_eval_data.py` loader can treat BabyBabel and
FineWeb-2 held-out shards identically.

Usage:
    python -m pipeline.stream_held_out_babybabel --lang deu
    python -m pipeline.stream_held_out_babybabel --langs deu nld zho
    python -m pipeline.stream_held_out_babybabel --lang deu --output-dir pipeline_output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("HeldOutBabyBabel")


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BABYBABEL_ORG = "Beetle-Data"
TRAIN_TARGET = "50M"
FULL_TARGET = "100M"

SHARD_SIZE = 10_000

SCHEMA = pa.schema([
    ("text", pa.utf8()),
    ("url", pa.utf8()),
    ("doc_id", pa.int64()),
])


def babybabel_repo(lang: str, target: str) -> str:
    return f"{BABYBABEL_ORG}/BabyBabel-{lang}-{target}"


def _text_hash(text: str) -> str:
    """Stable 128-bit hash of a document's full text."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _detect_text_column(ds) -> str:
    for col in ds.column_names:
        try:
            if ds.features[col].dtype == "string":
                return col
        except (AttributeError, KeyError):
            continue
    return "text"


# ═══════════════════════════════════════════════════════════════════════════════
# Core Logic
# ═══════════════════════════════════════════════════════════════════════════════

def build_train_hashes(lang: str) -> Set[str]:
    """Return the set of document-text hashes present in BabyBabel-{lang}-50M."""
    from datasets import load_dataset

    train_repo = babybabel_repo(lang, TRAIN_TARGET)
    log.info("Loading training manifest %s to build dedup hash set", train_repo)
    ds = load_dataset(train_repo, split="train")
    text_col = _detect_text_column(ds)

    hashes: Set[str] = set()
    for entry in tqdm(ds, desc=f"Hash {lang}/{TRAIN_TARGET}", unit="docs",
                      dynamic_ncols=True):
        text = entry.get(text_col, "") or ""
        if isinstance(text, str) and text.strip():
            hashes.add(_text_hash(text.strip()))

    log.info("  %s/%s: %d unique doc hashes (from %d rows)",
             lang, TRAIN_TARGET, len(hashes), len(ds))
    return hashes


def stream_held_out_babybabel(
    lang: str,
    output_dir: str,
    max_docs: int | None = None,
) -> Dict:
    """Stream documents from BabyBabel-{lang}-100M that are NOT in -50M.

    Args:
        lang: ISO-3 language code (e.g., "deu", "nld", "zho").
        output_dir: Base pipeline output directory.
        max_docs: Optional cap on held-out docs collected (useful for smoke tests).

    Returns:
        Stats dict.
    """
    from datasets import load_dataset

    t0 = time.time()

    # Step 1: dedup hash set from training data
    train_hashes = build_train_hashes(lang)

    # Step 2: stream 100M and emit non-overlapping docs
    full_repo = babybabel_repo(lang, FULL_TARGET)
    log.info("Streaming held-out from %s (skipping docs present in %s)",
             full_repo, babybabel_repo(lang, TRAIN_TARGET))

    ds_full = load_dataset(full_repo, split="train")
    text_col = _detect_text_column(ds_full)

    out_dir = Path(output_dir) / "held_out_babybabel" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer: List[Dict] = []
    shard_idx = 0
    docs_seen = 0
    docs_in_training = 0
    docs_held_out = 0

    pbar = tqdm(total=len(ds_full), unit="docs", desc=f"Held-out {lang}",
                dynamic_ncols=True)
    for entry in ds_full:
        docs_seen += 1
        text = entry.get(text_col, "") or ""
        if isinstance(text, str):
            text = text.strip()
        pbar.update(1)
        if not text:
            continue

        h = _text_hash(text)
        if h in train_hashes:
            docs_in_training += 1
            continue

        buffer.append({
            "text": text,
            "url": entry.get("url", "") or "",
            "doc_id": docs_held_out,
        })
        docs_held_out += 1

        if len(buffer) >= SHARD_SIZE:
            _write_shard(out_dir, lang, shard_idx, buffer)
            shard_idx += 1
            buffer.clear()

        if max_docs is not None and docs_held_out >= max_docs:
            break
    pbar.close()

    if buffer:
        _write_shard(out_dir, lang, shard_idx, buffer)
        shard_idx += 1
        buffer.clear()

    wall_time = time.time() - t0

    stats = {
        "lang": lang,
        "source_full": full_repo,
        "source_train": babybabel_repo(lang, TRAIN_TARGET),
        "docs_in_training_set": len(train_hashes),
        "docs_in_full_corpus": docs_seen,
        "docs_overlapping_training": docs_in_training,
        "docs_held_out": docs_held_out,
        "shards_written": shard_idx,
        "shard_size": SHARD_SIZE,
        "wall_time_sec": round(wall_time, 1),
    }

    stats_path = out_dir / f"{lang}_held_out_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Done %s: %d held-out docs (%d overlap / %d total), %d shards, %.1f min",
             lang, docs_held_out, docs_in_training, docs_seen,
             shard_idx, wall_time / 60)

    return stats


def _write_shard(out_dir: Path, lang: str, shard_idx: int, buffer: List[Dict]) -> str:
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
    log.info("  wrote shard %s (%d docs)", fname, len(buffer))
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stream held-out BabyBabel docs (100M ∖ 50M) for AoA eval."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", type=str, help="Single ISO-3 language code")
    group.add_argument("--langs", nargs="+", help="Multiple language codes")
    parser.add_argument("--output-dir", type=str, default="pipeline_output",
                        help="Base output directory (default: pipeline_output)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap on held-out docs (useful for smoke tests)")
    args = parser.parse_args()

    langs = [args.lang] if args.lang else args.langs
    for lang in langs:
        stream_held_out_babybabel(lang=lang, output_dir=args.output_dir,
                                  max_docs=args.max_docs)


if __name__ == "__main__":
    main()
