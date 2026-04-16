"""
pretokenize_babybabel.py — Pretokenize BabyBabel human-scale data for BeetleLM.

Reads raw text from Beetle-Data HuggingFace datasets (9 languages × 11 targets),
tokenizes with mono/bilingual/trilingual tokenizers from Beetle-HumanScale,
packs into 513-token Arrow sequences, and pushes to Beetle-HumanScale on
HuggingFace Hub.

Output format matches beetlelm/src/bilingual/data/pretokenize.py exactly:
  - chunk_len = 513  (seq_len=512 + 1 label token)
  - tokenizer.encode(text, add_special_tokens=False)
  - No cross-document token bleeding (buf = [] at document boundaries)
  - Dataset.from_dict({"input_ids": chunks}).save_to_disk(out_path)

Naming conventions (FineWeb-aligned, pushed to Beetle-HumanScale):
  Monolingual:  Beetle-HumanScale/{lang}-{target}
                e.g. Beetle-HumanScale/nld-100M
  Bilingual L1: Beetle-HumanScale/{l1}-{target}-{tok_pair}
                e.g. Beetle-HumanScale/nld-50M-eng-nld
  Bilingual L2: Beetle-HumanScale/{l2}-for-{l1}-{target}-{tok_pair}
                e.g. Beetle-HumanScale/eng-for-nld-50M-eng-nld
  Trilingual:   Beetle-HumanScale/{lang}-{target}-{tok_triple}
                e.g. Beetle-HumanScale/nld-33M-eng-nld-zho

Usage:
    # Monolingual
    python -m pipeline.pretokenize_babybabel --mono --lang nld
    python -m pipeline.pretokenize_babybabel --mono --lang nld --target 100M

    # Bilingual pair (default 50M per side)
    python -m pipeline.pretokenize_babybabel --pair eng nld
    python -m pipeline.pretokenize_babybabel --pair eng nld --l1-target 80M --l2-target 20M

    # Trilingual (default 33M per language)
    python -m pipeline.pretokenize_babybabel --triple eng nld zho

    # Pilot (3 MECO pairs: nld-eng, zho-eng, deu-eng)
    python -m pipeline.pretokenize_babybabel --pilot

    # All 36 bilingual pairs
    python -m pipeline.pretokenize_babybabel --all

    # Skip HuggingFace upload
    python -m pipeline.pretokenize_babybabel --all --no-upload
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

HF_ORG = "Beetle-HumanScale"

CHUNK_LEN = 513  # seq_len(512) + 1 label token

# BabyBabel raw text sources (Beetle-Data HuggingFace org, per-target datasets)
BABYBABEL_SOURCE_ORG = "Beetle-Data"
BABYBABEL_LANGS = {"zho", "fas", "eng", "nld", "bul", "fra", "ind", "deu", "ukr"}


def babybabel_source_repo(lang: str, target: str) -> str:
    """HF repo for BabyBabel raw text at a given target size.

    Example: Beetle-Data/BabyBabel-ukr-100M
    """
    return f"{BABYBABEL_SOURCE_ORG}/BabyBabel-{lang}-{target}"

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

# All 36 bilingual tokenizer pairs (sorted alphabetically per pair)
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

# Pilot pairs — MECO languages paired with English
PILOT_PAIRS: List[Tuple[str, str]] = [
    ("nld", "eng"), ("zho", "eng"), ("deu", "eng"),
]


# =============================================================================
# Repo naming conventions
# =============================================================================

def mono_tokenizer_repo(lang: str) -> str:
    """HF repo for a monolingual tokenizer."""
    return f"{HF_ORG}/bpe-humanscale-{lang}"


def bilingual_tokenizer_repo(l1: str, l2: str) -> str:
    """HF repo for a bilingual tokenizer (alphabetical order)."""
    a, b = sorted([l1, l2])
    return f"{HF_ORG}/bpe-humanscale-{a}-{b}"


def trilingual_tokenizer_repo(l1: str, l2: str, l3: str) -> str:
    """HF repo for a trilingual tokenizer (alphabetical order)."""
    a, b, c = sorted([l1, l2, l3])
    return f"{HF_ORG}/bpe-humanscale-{a}-{b}-{c}"


def mono_dataset_repo(lang: str, target: str = "100M") -> str:
    """HF repo for a monolingual pretokenized dataset.

    Example: Beetle-HumanScale/nld-100M
    """
    return f"{HF_ORG}/{lang}-{target}"


def l1_dataset_repo(l1: str, l2: str, target: str = "50M") -> str:
    """HF repo for the L1 side of a bilingual pretokenized dataset.

    Example: Beetle-HumanScale/nld-50M-eng-nld
    """
    tok_pair = "-".join(sorted([l1, l2]))
    return f"{HF_ORG}/{l1}-{target}-{tok_pair}"


def l2_dataset_repo(l1: str, l2: str, target: str = "50M") -> str:
    """HF repo for the L2 side of a bilingual pretokenized dataset.

    Example: Beetle-HumanScale/eng-for-nld-50M-eng-nld
    """
    tok_pair = "-".join(sorted([l1, l2]))
    return f"{HF_ORG}/{l2}-for-{l1}-{target}-{tok_pair}"


def tri_dataset_repo(lang: str, l1: str, l2: str, l3: str, target: str = "33M") -> str:
    """HF repo for one language side of a trilingual pretokenized dataset.

    Example: Beetle-HumanScale/nld-33M-eng-nld-zho
    """
    tok_triple = "-".join(sorted([l1, l2, l3]))
    return f"{HF_ORG}/{lang}-{target}-{tok_triple}"


# =============================================================================
# Tokenizer auto-detection
# =============================================================================

def ensure_mono_tokenizer(lang: str) -> None:
    """Check if monolingual tokenizer exists on HF; train and push if missing."""
    from transformers import PreTrainedTokenizerFast

    repo = mono_tokenizer_repo(lang)
    try:
        PreTrainedTokenizerFast.from_pretrained(repo)
        log.info("Tokenizer found: %s", repo)
    except OSError:
        log.warning("Tokenizer %s not found. Training automatically...", repo)
        _train_babybabel_tokenizer(
            langs=[lang], hf_repo=repo,
            label=f"monolingual {lang}",
        )
        log.info("Tokenizer trained and pushed: %s", repo)


def ensure_bilingual_tokenizer(l1: str, l2: str) -> None:
    """Check if bilingual tokenizer exists on HF; train and push if missing."""
    from transformers import PreTrainedTokenizerFast

    repo = bilingual_tokenizer_repo(l1, l2)
    try:
        PreTrainedTokenizerFast.from_pretrained(repo)
        log.info("Tokenizer found: %s", repo)
    except OSError:
        log.warning("Tokenizer %s not found. Training automatically...", repo)
        _train_babybabel_tokenizer(
            langs=sorted([l1, l2]), hf_repo=repo,
            label=f"bilingual {l1}-{l2}",
        )
        log.info("Tokenizer trained and pushed: %s", repo)


def ensure_trilingual_tokenizer(l1: str, l2: str, l3: str) -> None:
    """Check if trilingual tokenizer exists on HF; train and push if missing."""
    from transformers import PreTrainedTokenizerFast

    repo = trilingual_tokenizer_repo(l1, l2, l3)
    try:
        PreTrainedTokenizerFast.from_pretrained(repo)
        log.info("Tokenizer found: %s", repo)
    except OSError:
        log.warning("Tokenizer %s not found. Training automatically...", repo)
        _train_babybabel_tokenizer(
            langs=sorted([l1, l2, l3]), hf_repo=repo,
            label=f"trilingual {l1}-{l2}-{l3}",
        )
        log.info("Tokenizer trained and pushed: %s", repo)


def _train_babybabel_tokenizer(
    langs: List[str],
    hf_repo: str,
    label: str,
    vocab_size: int = 50_000,
) -> None:
    """Train a BPE tokenizer on BabyBabel data and push to HF Hub.

    Streams text from Beetle-Data datasets for the specified languages,
    allocating training data equally across languages.
    """
    import importlib.util

    tok_path = Path(__file__).resolve().parents[1] / "tok" / "multi-train-tok.py"
    spec = importlib.util.spec_from_file_location("multi_train_tok", tok_path)
    multi_train_tok = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multi_train_tok)

    multi_train_tok.train_babybabel_tokenizer(
        langs=langs,
        hf_repo=hf_repo,
        vocab_size=vocab_size,
    )


# =============================================================================
# Core pretokenization
# =============================================================================

def pretokenize_one(
    lang: str,
    tokenizer_repo_id: str,
    dataset_repo_id: str,
    target: str = "50M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
    cleanup_local: bool = False,
) -> dict:
    """Pretokenize one language's BabyBabel text with a given tokenizer.

    Exactly mirrors beetlelm/src/bilingual/data/pretokenize.py:116-131:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= chunk_len:
            chunks.append(buf[:chunk_len])
            buf = buf[chunk_len:]
        buf = []  # discard remainder at document boundary

    Args:
        lang: Language to pretokenize (e.g., "nld").
        tokenizer_repo_id: HF tokenizer repo (e.g., "Beetle-HumanScale/bpe-humanscale-eng-nld").
        dataset_repo_id: HF dataset repo to push to (e.g., "Beetle-HumanScale/nld-50M-eng-nld").
        target: Token target name (e.g., "50M").
        output_dir: Local output directory for Arrow datasets.
        upload: Whether to push to HuggingFace Hub.
        cleanup_local: Whether to delete local files after upload.

    Returns:
        Dict with pretokenization statistics.
    """
    from datasets import Dataset, load_dataset
    from transformers import PreTrainedTokenizerFast

    source_repo = babybabel_source_repo(lang, target)

    # Use the dataset repo name (without org prefix) as local dir name
    local_name = dataset_repo_id.split("/")[-1] if "/" in dataset_repo_id else dataset_repo_id
    arrow_dir = Path(output_dir) / local_name

    log.info("=" * 70)
    log.info("Pretokenizing %s → %s", lang, dataset_repo_id)
    log.info("  Source:     %s", source_repo)
    log.info("  Tokenizer:  %s", tokenizer_repo_id)
    log.info("  Output:     %s", arrow_dir)
    log.info("  Chunk len:  %d (seq_len=512 + 1)", CHUNK_LEN)

    t0 = time.time()

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_repo_id)
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
        "tokenizer": tokenizer_repo_id,
        "dataset_repo": dataset_repo_id,
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
        _upload_to_hf(arrow_dir, dataset_repo_id, cleanup_local)

    return stats


# =============================================================================
# Monolingual
# =============================================================================

def pretokenize_mono(
    lang: str,
    target: str = "100M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
    cleanup_local: bool = False,
) -> dict:
    """Pretokenize a single language with its monolingual tokenizer.

    Output: Beetle-HumanScale/{lang}-{target} (e.g., Beetle-HumanScale/nld-100M)
    """
    ensure_mono_tokenizer(lang)
    tok_repo = mono_tokenizer_repo(lang)
    ds_repo = mono_dataset_repo(lang, target)

    return pretokenize_one(
        lang=lang, tokenizer_repo_id=tok_repo, dataset_repo_id=ds_repo,
        target=target, output_dir=output_dir, upload=upload,
        cleanup_local=cleanup_local,
    )


# =============================================================================
# Bilingual
# =============================================================================

def pretokenize_pair(
    l1: str,
    l2: str,
    l1_target: str = "50M",
    l2_target: Optional[str] = None,
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
    cleanup_local: bool = False,
) -> List[dict]:
    """Pretokenize both languages of a bilingual pair.

    For pair (l1=nld, l2=eng), produces:
      - Beetle-HumanScale/nld-50M-eng-nld  (L1: Dutch text, eng-nld tokenizer)
      - Beetle-HumanScale/eng-for-nld-50M-eng-nld  (L2: English text for Dutch pair)

    Args:
        l1: First language (L1).
        l2: Second language (L2).
        l1_target: Token target for L1 side (default "50M").
        l2_target: Token target for L2 side (default: same as l1_target).
                   Use different targets for B4 classroom (e.g., l1="80M", l2="20M").
    """
    if l2_target is None:
        l2_target = l1_target

    ensure_bilingual_tokenizer(l1, l2)
    tok_repo = bilingual_tokenizer_repo(l1, l2)

    results = []

    # L1 side
    ds_repo_l1 = l1_dataset_repo(l1, l2, l1_target)
    stats_l1 = pretokenize_one(
        lang=l1, tokenizer_repo_id=tok_repo, dataset_repo_id=ds_repo_l1,
        target=l1_target, output_dir=output_dir, upload=upload,
        cleanup_local=cleanup_local,
    )
    results.append(stats_l1)

    # L2 side
    ds_repo_l2 = l2_dataset_repo(l1, l2, l2_target)
    stats_l2 = pretokenize_one(
        lang=l2, tokenizer_repo_id=tok_repo, dataset_repo_id=ds_repo_l2,
        target=l2_target, output_dir=output_dir, upload=upload,
        cleanup_local=cleanup_local,
    )
    results.append(stats_l2)

    return results


# =============================================================================
# Trilingual
# =============================================================================

def pretokenize_triple(
    l1: str,
    l2: str,
    l3: str,
    target: str = "33M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
    cleanup_local: bool = False,
) -> List[dict]:
    """Pretokenize three languages with a trilingual tokenizer.

    Produces three separate datasets, one per language, all tokenized with
    the same trilingual tokenizer.

    For (l1=eng, l2=nld, l3=zho), produces:
      - Beetle-HumanScale/eng-33M-eng-nld-zho
      - Beetle-HumanScale/nld-33M-eng-nld-zho
      - Beetle-HumanScale/zho-33M-eng-nld-zho
    """
    ensure_trilingual_tokenizer(l1, l2, l3)
    tok_repo = trilingual_tokenizer_repo(l1, l2, l3)

    results = []
    for lang in sorted(set([l1, l2, l3])):
        ds_repo = tri_dataset_repo(lang, l1, l2, l3, target)
        stats = pretokenize_one(
            lang=lang, tokenizer_repo_id=tok_repo, dataset_repo_id=ds_repo,
            target=target, output_dir=output_dir, upload=upload,
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


def pretokenize_pilot(
    target: str = "50M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
) -> List[dict]:
    """Pretokenize MECO pilot pairs: nld-eng, zho-eng, deu-eng."""
    pairs = _unique_pairs(PILOT_PAIRS)
    log.info("Pilot mode: %d unique pairs → %d Arrow datasets",
             len(pairs), len(pairs) * 2)
    results = []
    for l1, l2 in pairs:
        results.extend(pretokenize_pair(l1, l2, target, output_dir=output_dir, upload=upload))
    return results


def pretokenize_all(
    target: str = "50M",
    output_dir: str = "./pipeline_output/babybabel",
    upload: bool = True,
) -> List[dict]:
    """Pretokenize all 36 bilingual pairs (72 Arrow datasets)."""
    pairs = _unique_pairs(TOKENIZER_PAIRS)
    log.info("Full mode: %d unique pairs → %d Arrow datasets",
             len(pairs), len(pairs) * 2)
    results = []
    for l1, l2 in pairs:
        results.extend(pretokenize_pair(l1, l2, target, output_dir=output_dir, upload=upload))
    return results


# =============================================================================
# Upload
# =============================================================================

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
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize BabyBabel human-scale data for BeetleLM",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mono", action="store_true",
                      help="Pretokenize a single language with monolingual tokenizer")
    mode.add_argument("--pair", nargs=2, metavar=("L1", "L2"),
                      help="Pretokenize a bilingual pair (e.g., --pair nld eng)")
    mode.add_argument("--triple", nargs=3, metavar=("L1", "L2", "L3"),
                      help="Pretokenize a trilingual triple (e.g., --triple eng nld zho)")
    mode.add_argument("--pilot", action="store_true",
                      help="Pretokenize MECO pilot pairs: nld-eng, zho-eng, deu-eng")
    mode.add_argument("--all", action="store_true",
                      help="Pretokenize all 36 bilingual pairs (72 Arrow datasets)")

    parser.add_argument("--lang", type=str, default=None,
                        help="Language for --mono mode (e.g., nld)")
    parser.add_argument("--target", default=None,
                        help="Token target per language (default: 100M mono, 50M bi, 33M tri)")
    parser.add_argument("--l1-target", default=None,
                        help="Token target for L1 side (bilingual only, e.g., 80M for B4 classroom)")
    parser.add_argument("--l2-target", default=None,
                        help="Token target for L2 side (bilingual only, e.g., 20M for B4 classroom)")
    parser.add_argument("--output-dir", default="./pipeline_output/babybabel",
                        help="Local output directory (default: ./pipeline_output/babybabel)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading to HuggingFace Hub")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete local Arrow files after successful upload")

    args = parser.parse_args()
    upload = not args.no_upload

    t0 = time.time()
    results = []

    if args.mono:
        if not args.lang:
            parser.error("--mono requires --lang")
        if args.lang not in BABYBABEL_LANGS:
            parser.error(f"Unknown language: {args.lang}. Available: {sorted(BABYBABEL_LANGS)}")
        target = args.target or "100M"
        if target not in TARGETS:
            parser.error(f"Unknown target: {target}. Available: {list(TARGETS.keys())}")
        results = [pretokenize_mono(args.lang, target, args.output_dir, upload, args.cleanup)]

    elif args.pair:
        l1, l2 = args.pair
        for lang in [l1, l2]:
            if lang not in BABYBABEL_LANGS:
                parser.error(f"Unknown language: {lang}. Available: {sorted(BABYBABEL_LANGS)}")
        l1_target = args.l1_target or args.target or "50M"
        l2_target = args.l2_target or args.target or "50M"
        results = pretokenize_pair(l1, l2, l1_target, l2_target, args.output_dir, upload, args.cleanup)

    elif args.triple:
        l1, l2, l3 = args.triple
        for lang in [l1, l2, l3]:
            if lang not in BABYBABEL_LANGS:
                parser.error(f"Unknown language: {lang}. Available: {sorted(BABYBABEL_LANGS)}")
        target = args.target or "33M"
        results = pretokenize_triple(l1, l2, l3, target, args.output_dir, upload, args.cleanup)

    elif args.pilot:
        target = args.target or "50M"
        results = pretokenize_pilot(target, args.output_dir, upload)

    elif args.all:
        target = args.target or "50M"
        results = pretokenize_all(target, args.output_dir, upload)

    total_time = time.time() - t0
    total_chunks = sum(r["chunks_created"] for r in results)
    total_tokens = sum(r["tokens_total"] for r in results)

    log.info("=" * 70)
    log.info("DONE: %d datasets, %d chunks, %d tokens, %.1f min",
             len(results), total_chunks, total_tokens, total_time / 60)
    for r in results:
        log.info("  %s [%s]: %d chunks → %s",
                 r["lang"], r["tokenizer"], r["chunks_created"], r["dataset_repo"])


if __name__ == "__main__":
    main()
