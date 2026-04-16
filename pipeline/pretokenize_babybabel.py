"""
pretokenize_babybabel.py — Pretokenize BabyBabel human-scale data for BeetleLM.

Reads raw text from BabyLM-community HuggingFace datasets (9 languages),
tokenizes with bilingual tokenizers from Beetle-HumanScale, packs into
513-token Arrow sequences, and pushes to Beetle-Data on HuggingFace Hub.

Output format matches beetlelm/src/bilingual/data/pretokenize.py exactly:
  - chunk_len = 513  (seq_len=512 + 1 label token)
  - tokenizer.encode(text, add_special_tokens=False)
  - No cross-document token bleeding (buf = [] at document boundaries)
  - Dataset.from_dict({"input_ids": chunks}).save_to_disk(out_path)

Naming convention: Beetle-Data/BabyBabel-{lang}-{target}-{tok_l1}-{tok_l2}
  e.g. Beetle-Data/BabyBabel-nld-50M-eng-nld  (Dutch text, eng-nld tokenizer)

Since tokenizers are bilingual, the same text tokenized with different pair
tokenizers produces different input_ids. Each dataset name encodes the
tokenizer used.

Usage:
    # Single pair
    python -m pipeline.pretokenize_babybabel --pair eng nld

    # 3 pilot pairs (nld-eng, zho-eng, zho-nld)
    python -m pipeline.pretokenize_babybabel --pilot

    # All 36 pairs (72 Arrow datasets)
    python -m pipeline.pretokenize_babybabel --all

    # Skip HuggingFace upload
    python -m pipeline.pretokenize_babybabel --all --no-upload

    # Custom token target
    python -m pipeline.pretokenize_babybabel --pilot --target 100M
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("PretokBabyBabel")


# =============================================================================
# Constants
# =============================================================================

HF_ORG_DATA = "Beetle-Data"
HF_ORG_TOK = "Beetle-HumanScale"

CHUNK_LEN = 513  # seq_len(512) + 1 label token

# BabyBabel raw text sources (BabyLM community datasets)
BABYBABEL_SOURCES: Dict[str, str] = {
    "zho": "BabyLM-community/babylm-zho",
    "fas": "BabyLM-community/babylm-fas",
    "eng": "BabyLM-community/babylm-eng",
    "nld": "BabyLM-community/babylm-nld",
    "bul": "BabyLM-community/babylm-bul",
    "fra": "BabyLM-community/babylm-fra",
    "ind": "BabyLM-community/babylm-ind",
    "deu": "BabyLM-community/babylm-deu",
    "ukr": "BabyLM-community/babylm-ukr",
}

# Token targets (name → token count)
TARGETS: Dict[str, int] = {
    "10M": 10_000_000,
    "20M": 20_000_000,
    "30M": 30_000_000,
    "33M": 33_000_000,
    "40M": 40_000_000,
    "50M": 50_000_000,
    "60M": 60_000_000,
    "66M": 66_000_000,
    "70M": 70_000_000,
    "80M": 80_000_000,
    "100M": 100_000_000,
}

# All 36 bilingual tokenizer pairs (mirrored from beetlelm's TOKENIZER_PAIRS
# in configs/generate_human_scale_bilingual.py). Sorted alphabetically per pair
# to match the tokenizer naming convention: bpe-humanscale-{l1}-{l2} with l1<l2.
TOKENIZER_PAIRS: List[Tuple[str, str]] = [
    ("bul", "deu"), ("bul", "fra"), ("bul", "ind"), ("bul", "ukr"),
    ("deu", "ukr"),
    ("eng", "bul"), ("eng", "deu"), ("eng", "fra"), ("eng", "ind"),
    ("eng", "nld"), ("eng", "ukr"),
    ("fas", "bul"), ("fas", "deu"), ("fas", "eng"), ("fas", "fra"),
    ("fas", "ind"), ("fas", "nld"), ("fas", "ukr"),
    ("fra", "deu"), ("fra", "ind"), ("fra", "ukr"),
    ("ind", "deu"), ("ind", "ukr"),
    ("nld", "bul"), ("nld", "deu"), ("nld", "fra"), ("nld", "ind"),
    ("nld", "ukr"),
    ("zho", "bul"), ("zho", "deu"), ("zho", "eng"), ("zho", "fra"),
    ("zho", "fas"), ("zho", "ind"), ("zho", "nld"), ("zho", "ukr"),
]

# Pilot pairs for Phase 3a sweep
PILOT_PAIRS: List[Tuple[str, str]] = [
    ("nld", "eng"), ("zho", "eng"), ("zho", "nld"),
]


def tokenizer_repo(l1: str, l2: str) -> str:
    """HuggingFace repo ID for a bilingual tokenizer (alphabetical order)."""
    a, b = sorted([l1, l2])
    return f"{HF_ORG_TOK}/bpe-humanscale-{a}-{b}"


def dataset_repo(lang: str, l1: str, l2: str, target: str = "50M") -> str:
    """HuggingFace repo ID for a pretokenized BabyBabel Arrow dataset."""
    tok_pair = "-".join(sorted([l1, l2]))
    return f"{HF_ORG_DATA}/BabyBabel-{lang}-{target}-{tok_pair}"


# =============================================================================
# Core pretokenization
# =============================================================================

def pretokenize_one(
    lang: str,
    l1: str,
    l2: str,
    target: str = "50M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
    cleanup_local: bool = False,
) -> dict:
    """Pretokenize one language's BabyBabel text with a bilingual tokenizer.

    Exactly mirrors beetlelm/src/bilingual/data/pretokenize.py:116-131:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= chunk_len:
            chunks.append(buf[:chunk_len])
            buf = buf[chunk_len:]
        buf = []  # discard remainder at document boundary

    Args:
        lang: Language to pretokenize (e.g., "nld").
        l1: First language of the tokenizer pair (e.g., "eng").
        l2: Second language of the tokenizer pair (e.g., "nld").
        target: Token target name (e.g., "50M").
        output_dir: Local output directory for Arrow datasets.
        upload: Whether to push to HuggingFace Hub.
        cleanup_local: Whether to delete local files after upload.

    Returns:
        Dict with pretokenization statistics.
    """
    from datasets import Dataset, load_dataset
    from transformers import PreTrainedTokenizerFast

    tok_repo_id = tokenizer_repo(l1, l2)
    ds_repo_id = dataset_repo(lang, l1, l2, target)
    source_repo = BABYBABEL_SOURCES[lang]

    tok_pair = "-".join(sorted([l1, l2]))
    arrow_dir = Path(output_dir) / f"{lang}-{target}-{tok_pair}"

    log.info("=" * 70)
    log.info("Pretokenizing %s with tokenizer %s", lang, tok_repo_id)
    log.info("  Source:    %s", source_repo)
    log.info("  Output:    %s", arrow_dir)
    log.info("  HF repo:   %s", ds_repo_id)
    log.info("  Chunk len: %d (seq_len=512 + 1)", CHUNK_LEN)

    t0 = time.time()

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_repo_id)
    log.info("  Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)

    # Load BabyBabel dataset (small enough to fit in memory)
    ds = load_dataset(source_repo, split="train")
    log.info("  Dataset loaded: %d documents", len(ds))

    # Auto-detect text column
    text_col = None
    for col in ds.column_names:
        if ds.features[col].dtype == "string":
            text_col = col
            break
    if text_col is None:
        text_col = "text"
    log.info("  Text column: %s", text_col)

    # Tokenize and pack into fixed-length chunks
    target_tokens = TARGETS.get(target)
    target_chunks = target_tokens // CHUNK_LEN if target_tokens else None

    chunks: List[List[int]] = []
    total_tokens = 0
    total_discarded = 0

    for entry in ds:
        text = entry.get(text_col, "")
        if isinstance(text, str):
            text = text.strip()
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(ids)

        buf = list(ids)
        while len(buf) >= CHUNK_LEN:
            chunks.append(buf[:CHUNK_LEN])
            buf = buf[CHUNK_LEN:]
        # Discard remainder at document boundary (no cross-doc bleeding)
        total_discarded += len(buf)

        # Stop if we've hit the token target
        if target_chunks and len(chunks) >= target_chunks:
            chunks = chunks[:target_chunks]
            break

    wall_time = time.time() - t0

    log.info("  Tokenized: %d tokens, %d chunks, %d discarded, %.1fs",
             total_tokens, len(chunks), total_discarded, wall_time)

    # Save Arrow dataset
    arrow_dir.mkdir(parents=True, exist_ok=True)
    arrow_ds = Dataset.from_dict({"input_ids": chunks})
    arrow_ds.save_to_disk(str(arrow_dir))
    log.info("  Saved Arrow to %s", arrow_dir)

    # Save stats
    stats = {
        "lang": lang,
        "tokenizer": tok_repo_id,
        "target": target,
        "docs_processed": len(ds),
        "tokens_total": total_tokens,
        "chunks_created": len(chunks),
        "tokens_discarded": total_discarded,
        "wall_time_sec": round(wall_time, 1),
    }
    with open(arrow_dir / "pretok_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Upload to HuggingFace
    if upload:
        _upload_to_hf(arrow_dir, ds_repo_id, cleanup_local)

    return stats


def _upload_to_hf(arrow_dir: Path, repo_id: str, cleanup: bool = False) -> bool:
    """Upload Arrow dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    try:
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        create_repo(repo_id, repo_type="dataset", exist_ok=True,
                    token=token, private=False)

        log.info("  Uploading to %s ...", repo_id)
        api.upload_folder(
            folder_path=str(arrow_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Pretokenized BabyBabel ({arrow_dir.name})",
        )
        log.info("  Upload complete: %s", repo_id)

        if cleanup:
            shutil.rmtree(arrow_dir, ignore_errors=True)
            log.info("  Deleted local copy: %s", arrow_dir)

        return True
    except Exception as e:
        log.error("  Upload failed for %s: %s", repo_id, e)
        log.error("  Local files preserved at %s", arrow_dir)
        return False


# =============================================================================
# Pair-level entrypoint
# =============================================================================

def pretokenize_pair(
    l1: str,
    l2: str,
    target: str = "50M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
    cleanup_local: bool = False,
) -> List[dict]:
    """Pretokenize both languages of a bilingual pair.

    For pair (eng, nld), produces:
      - Beetle-Data/BabyBabel-eng-50M-eng-nld  (English text, eng-nld tokenizer)
      - Beetle-Data/BabyBabel-nld-50M-eng-nld  (Dutch text, eng-nld tokenizer)
    """
    results = []
    for lang in sorted(set([l1, l2])):
        stats = pretokenize_one(
            lang=lang, l1=l1, l2=l2, target=target,
            output_dir=output_dir, upload=upload,
            cleanup_local=cleanup_local,
        )
        results.append(stats)
    return results


# =============================================================================
# Batch entrypoints
# =============================================================================

def _unique_pairs(pair_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate pairs by tokenizer (sorted alphabetically)."""
    seen = set()
    unique = []
    for l1, l2 in pair_list:
        key = tuple(sorted([l1, l2]))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return sorted(unique)


def pretokenize_pilot(target: str = "50M", output_dir: str = "./pipeline_output/babybabel",
                      upload: bool = True) -> List[dict]:
    """Pretokenize 3 pilot pairs: nld-eng, zho-eng, zho-nld."""
    pairs = _unique_pairs(PILOT_PAIRS)
    log.info("Pilot mode: %d unique pairs → %d Arrow datasets",
             len(pairs), len(pairs) * 2)
    results = []
    for l1, l2 in pairs:
        results.extend(pretokenize_pair(l1, l2, target, output_dir, upload))
    return results


def pretokenize_all(target: str = "50M", output_dir: str = "./pipeline_output/babybabel",
                    upload: bool = True) -> List[dict]:
    """Pretokenize all 36 pairs (72 Arrow datasets)."""
    pairs = _unique_pairs(TOKENIZER_PAIRS)
    log.info("Full mode: %d unique pairs → %d Arrow datasets",
             len(pairs), len(pairs) * 2)
    results = []
    for l1, l2 in pairs:
        results.extend(pretokenize_pair(l1, l2, target, output_dir, upload))
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize BabyBabel human-scale data for BeetleLM",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", nargs=2, metavar=("L1", "L2"),
                       help="Pretokenize a single language pair (e.g., --pair eng nld)")
    group.add_argument("--pilot", action="store_true",
                       help="Pretokenize 3 pilot pairs: nld-eng, zho-eng, zho-nld")
    group.add_argument("--all", action="store_true",
                       help="Pretokenize all 36 bilingual pairs (72 Arrow datasets)")

    parser.add_argument("--target", default="50M", choices=list(TARGETS.keys()),
                        help="Token target per language (default: 50M)")
    parser.add_argument("--output-dir", default="./pipeline_output/babybabel",
                        help="Local output directory (default: ./pipeline_output/babybabel)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading to HuggingFace Hub")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete local Arrow files after successful upload")

    args = parser.parse_args()
    upload = not args.no_upload

    t0 = time.time()

    if args.pair:
        l1, l2 = args.pair
        if l1 not in BABYBABEL_SOURCES:
            parser.error(f"Unknown language: {l1}. Available: {list(BABYBABEL_SOURCES.keys())}")
        if l2 not in BABYBABEL_SOURCES:
            parser.error(f"Unknown language: {l2}. Available: {list(BABYBABEL_SOURCES.keys())}")
        results = pretokenize_pair(l1, l2, args.target, args.output_dir, upload, args.cleanup)
    elif args.pilot:
        results = pretokenize_pilot(args.target, args.output_dir, upload)
    else:
        results = pretokenize_all(args.target, args.output_dir, upload)

    total_time = time.time() - t0
    total_chunks = sum(r["chunks_created"] for r in results)
    total_tokens = sum(r["tokens_total"] for r in results)

    log.info("=" * 70)
    log.info("DONE: %d datasets, %d chunks, %d tokens, %.1f min",
             len(results), total_chunks, total_tokens, total_time / 60)
    for r in results:
        log.info("  %s [%s]: %d chunks", r["lang"], r["tokenizer"], r["chunks_created"])


if __name__ == "__main__":
    main()
