"""push_held_out_babybabel.py — Publish held-out BabyBabel shards to HF.

Reads held-out (100M ∖ 50M) parquet shards and their stats files produced by
``pipeline.stream_held_out_babybabel`` and pushes one dataset repo per
language to the target HF organisation (default: ``Beetle-HumanScale``).

Layout expected under ``--output-dir``::

    held_out_babybabel/
      {lang}/
        {lang}_held_out_00000.parquet
        {lang}_held_out_stats.json

One repo per language is created as ``{org}/BabyBabel-{lang}-held-out``.
Uploads use ``HfApi.upload_large_folder`` with a ``upload_folder`` fallback
(same batching pattern as ``pipeline.decontaminate_stream``) so we stay well
under HF's 128-commits/hour per-repo limit.

Usage::

    python -m pipeline.push_held_out_babybabel --langs zho nld eng deu \
        --output-dir /root/beetle-data/pipeline_output
    python -m pipeline.push_held_out_babybabel --lang deu --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("PushHeldOutBabyBabel")


DEFAULT_ORG = "Beetle-HumanScale"


def _repo_id(org: str, lang: str) -> str:
    return f"{org}/BabyBabel-{lang}-held-out"


def _load_stats(stats_path: Path) -> Optional[dict]:
    if not stats_path.exists():
        return None
    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not parse %s: %s", stats_path, e)
        return None


def push_lang(
    lang: str,
    output_dir: Path,
    org: str,
    token: Optional[str],
    dry_run: bool,
) -> bool:
    """Push a single language's held-out shards to HF. Returns True on success."""
    lang_dir = output_dir / "held_out_babybabel" / lang
    if not lang_dir.is_dir():
        log.error("[%s] held-out directory not found: %s", lang, lang_dir)
        return False

    parquet_files = sorted(lang_dir.glob("*.parquet"))
    if not parquet_files:
        log.error("[%s] no parquet shards in %s", lang, lang_dir)
        return False

    stats_name = f"{lang}_held_out_stats.json"
    stats_path = lang_dir / stats_name
    stats = _load_stats(stats_path)
    if stats is None:
        log.warning("[%s] stats file missing or unreadable: %s", lang, stats_path)

    repo_id = _repo_id(org, lang)
    allow_patterns = ["*.parquet", stats_name]

    log.info(
        "[%s] → %s  (%d parquet shard(s), stats=%s, held_out_docs=%s)",
        lang,
        repo_id,
        len(parquet_files),
        "yes" if stats is not None else "MISSING",
        stats.get("docs_held_out") if stats else "?",
    )
    for p in parquet_files:
        log.info("    shard: %s (%.1f MB)", p.name, p.stat().st_size / 1e6)

    if dry_run:
        log.info("[%s] dry-run: skipping create_repo + upload", lang)
        return True

    from huggingface_hub import HfApi

    api = HfApi(token=token or None)

    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        log.error("[%s] create_repo failed for %s: %s", lang, repo_id, e)
        return False

    try:
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(
                folder_path=str(lang_dir),
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=allow_patterns,
            )
        else:
            log.info("[%s] upload_large_folder unavailable; using upload_folder", lang)
            api.upload_folder(
                folder_path=str(lang_dir),
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                commit_message=(
                    f"Upload held-out BabyBabel {lang} "
                    f"({len(parquet_files)} shard(s))"
                ),
            )
    except Exception as e:
        log.error("[%s] upload failed for %s: %s", lang, repo_id, e)
        return False

    log.info("[%s] ✓ pushed to https://huggingface.co/datasets/%s", lang, repo_id)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push held-out BabyBabel shards to HF (one repo per language).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", type=str, help="Single ISO-3 language code")
    group.add_argument("--langs", nargs="+", help="Multiple language codes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline_output",
        help="Base pipeline output directory (default: pipeline_output)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=DEFAULT_ORG,
        help=f"Target HF organisation (default: {DEFAULT_ORG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned uploads without creating repos or uploading files.",
    )
    args = parser.parse_args()

    langs: List[str] = [args.lang] if args.lang else args.langs
    output_dir = Path(args.output_dir).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    log.info("Pushing held-out BabyBabel for langs=%s", langs)
    log.info("  output_dir: %s", output_dir)
    log.info("  org:        %s", args.org)
    log.info("  token:      %s", "set" if token else "not set (using cached login)")
    log.info("  dry_run:    %s", args.dry_run)

    failed: List[str] = []
    for lang in langs:
        ok = push_lang(
            lang=lang,
            output_dir=output_dir,
            org=args.org,
            token=token,
            dry_run=args.dry_run,
        )
        if not ok:
            failed.append(lang)

    if failed:
        log.error("FAILED langs: %s", failed)
        return 1
    log.info("All %d language(s) pushed successfully.", len(langs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
