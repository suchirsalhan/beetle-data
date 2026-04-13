"""
run_pipeline.py — CLI orchestrator for the decontamination-to-pretokenization pipeline.

Provides a single entry point to run any or all pipeline stages:
  Stage 1: Build 13-gram benchmark index
  Stage 2: Stream FineWeb + decontaminate + write Parquet
  Stage 3: Pretokenize clean Parquet to Arrow

Usage:
    # Run all stages for a single language
    python -m pipeline.run_pipeline --lang pl --project-root /path/to/PHD

    # Run all stages for a SLURM node
    python -m pipeline.run_pipeline --node-id 0 --project-root /path/to/PHD

    # Run specific stage(s)
    python -m pipeline.run_pipeline --stage 1 --project-root /path/to/PHD
    python -m pipeline.run_pipeline --stage 2 --lang pl --index benchmark_13gram.pkl
    python -m pipeline.run_pipeline --stage 3 --lang pl

    # Run stages 2+3 for a node
    python -m pipeline.run_pipeline --stage 2 3 --node-id 0 --index benchmark_13gram.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import shutil

from .config import (
    LANG_REGISTRY, NON_EN_LANGS, PipelineConfig,
    BeetleStreamConfig, StreamMode,
    check_disk_space, langs_for_node,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("Pipeline")


def run_stage_1(cfg: PipelineConfig) -> str:
    """Stage 1: Build benchmark 13-gram index."""
    from .benchmark_index import build_benchmark_index

    log.info("=" * 60)
    log.info("STAGE 1: Building benchmark 13-gram index")
    log.info("=" * 60)

    t0 = time.time()
    idx = build_benchmark_index(cfg)
    idx.save(cfg.benchmark_index_path)
    dt = time.time() - t0

    log.info("Stage 1 complete in %.1f min", dt / 60)
    log.info(idx.summary())

    return cfg.benchmark_index_path


def run_stage_2_lang(lang: str, cfg: PipelineConfig) -> dict:
    """Stage 2: Stream, decontaminate, write Parquet for ONE language."""
    from .benchmark_index import BenchmarkIndex
    from .decontaminate_stream import decontaminate_language

    index = BenchmarkIndex.load(cfg.benchmark_index_path)
    t0 = time.time()
    stats = decontaminate_language(lang, index, cfg)
    dt = time.time() - t0
    log.info("  %s decontaminated in %.1f min", lang, dt / 60)
    return stats.to_dict()


def run_stage_3_lang(lang: str, cfg: PipelineConfig) -> dict:
    """Stage 3: Pretokenize + upload + cleanup for ONE language pair."""
    from .pretokenize_arrow import pretokenize_pair

    t0 = time.time()
    l1_stats, en_stats = pretokenize_pair(lang, cfg)
    dt = time.time() - t0
    log.info("  %s pretokenized + uploaded in %.1f min", lang, dt / 60)
    return {f"{lang}_l1": l1_stats.to_dict(), f"{lang}_en": en_stats.to_dict()}


def run_stage_A(cfg: PipelineConfig, langs: List[str]) -> dict:
    """Stage A: Teacher annotation of sampled documents."""
    from .teacher_annotate import AnnotatorConfig, annotate_all_languages

    log.info("=" * 60)
    log.info("STAGE A: Teacher Annotation (sampling %d docs across %d langs)",
             cfg.beetlestream.teacher_sample_size if cfg.beetlestream else 500_000,
             len(langs))
    log.info("=" * 60)

    bs = cfg.beetlestream or BeetleStreamConfig()
    annotator_cfg = AnnotatorConfig(
        model=bs.teacher_model,
        base_url=bs.teacher_base_url,
        sample_size=bs.teacher_sample_size,
        batch_size=bs.teacher_batch_size,
        max_concurrent=bs.teacher_max_concurrent,
        kidlm_repo=bs.kidlm_repo,
        clc_repo=bs.clc_repo,
        kidlm_samples=bs.kidlm_samples,
        samples_per_cefr_level=bs.samples_per_cefr_level,
    )

    decon_dir = str(Path(cfg.output_dir) / "decontaminated")
    t0 = time.time()
    stats = annotate_all_languages(decon_dir, cfg.output_dir, langs, annotator_cfg)
    dt = time.time() - t0
    log.info("Stage A complete in %.1f hours", dt / 3600)
    return stats.to_dict()


def run_stage_B(cfg: PipelineConfig, langs: List[str]) -> dict:
    """Stage B: Feature transform."""
    from .feature_transform import transform_all_languages

    log.info("=" * 60)
    log.info("STAGE B: Feature Transform")
    log.info("=" * 60)

    t0 = time.time()
    all_stats = transform_all_languages(cfg.output_dir, langs)
    dt = time.time() - t0
    log.info("Stage B complete in %.1f min", dt / 60)
    return {lang: s.to_dict() for lang, s in all_stats.items()}


def run_stage_C(cfg: PipelineConfig) -> dict:
    """Stage C: Student model training."""
    from .student_model import StudentConfig, train_student_model

    log.info("=" * 60)
    log.info("STAGE C: Student Model Training")
    log.info("=" * 60)

    bs = cfg.beetlestream or BeetleStreamConfig()
    student_cfg = StudentConfig(
        embedding_model=bs.embedding_model,
        embedding_dim=bs.embedding_dim,
        batch_size=bs.student_batch_size,
    )

    t0 = time.time()
    stats = train_student_model(cfg.output_dir, student_cfg)
    dt = time.time() - t0
    log.info("Stage C complete in %.1f min", dt / 60)
    return stats.to_dict()


def run_stage_D_lang(lang: str, cfg: PipelineConfig) -> dict:
    """Stage D: Score + index one language."""
    from .heuristic_filters import HeuristicConfig
    from .score_and_index import IndexConfig, score_and_index_all

    bs = cfg.beetlestream or BeetleStreamConfig()
    heuristic_cfg = HeuristicConfig(
        stopword_density_min=bs.stopword_density_min,
        stopword_density_max=bs.stopword_density_max,
        max_fk_grade=bs.max_fk_grade,
        min_script_consistency=bs.min_script_consistency,
        min_unique_5gram_ratio=bs.min_unique_5gram_ratio,
    )

    index_cfg = IndexConfig(
        n_clusters=bs.n_clusters,
        shard_size=bs.index_shard_size,
        num_workers=cfg.num_workers,
        embedding_batch_size=bs.student_batch_size,
        upload_to_hf=bs.upload_indexed_to_hf,
        hf_user=cfg.hf_user,
        heuristic_cfg=heuristic_cfg,
    )

    t0 = time.time()
    all_stats = score_and_index_all([lang], cfg.output_dir, index_cfg)
    dt = time.time() - t0
    log.info("  %s indexed in %.1f hours", lang, dt / 3600)
    return {lang: s.to_dict() for lang, s in all_stats.items()}


def run_storage_optimized(languages: List[str], cfg: PipelineConfig) -> None:
    """Run Stages 2+3 with storage-optimized per-language processing.

    Flow:
      1. Decontaminate English first (kept on disk for all pairs)
      2. For each non-English language:
         a. Decontaminate L1 → Parquet (~70 GB)
         b. Pretokenize L1 → Arrow → upload to HF → delete local
         c. Pretokenize EN → Arrow → upload to HF → delete local
         d. Delete L1 Parquet
      3. Delete EN Parquet (after all pairs done)

    Peak disk: ~270 GB at any point.
    """
    all_stats = {}
    non_en = [l for l in languages if l != "en"]

    # Step 1: Decontaminate English (needed for all pairs)
    en_parquet_dir = Path(cfg.output_dir) / "decontaminated" / "en"
    if not en_parquet_dir.exists() or not list(en_parquet_dir.glob("*.parquet")):
        log.info("=" * 60)
        log.info("STAGE 2: Decontaminating English (shared across all pairs)")
        log.info("=" * 60)
        all_stats["en"] = run_stage_2_lang("en", cfg)
    else:
        log.info("English Parquet already exists at %s — skipping", en_parquet_dir)

    # Step 2: Process each non-English language
    for i, lang in enumerate(non_en, 1):
        if lang not in LANG_REGISTRY:
            log.error("Unknown language: %s — skipping", lang)
            continue

        log.info("=" * 60)
        log.info("LANGUAGE %d/%d: %s", i, len(non_en), lang)
        log.info("=" * 60)

        # 2a. Decontaminate L1
        log.info("Stage 2: Decontaminating %s ...", lang)
        all_stats[lang] = run_stage_2_lang(lang, cfg)

        # 2b+2c. Pretokenize both sides, upload, cleanup
        log.info("Stage 3: Pretokenizing %s + EN, uploading to HF ...", lang)
        pair_stats = run_stage_3_lang(lang, cfg)
        all_stats.update(pair_stats)

        # Report disk usage
        _report_disk_usage(cfg.output_dir)

    # Step 3: Delete EN Parquet (all pairs done)
    if cfg.cleanup_stage2_after_pretok and en_parquet_dir.exists():
        shutil.rmtree(en_parquet_dir, ignore_errors=True)
        log.info("Cleaned up EN Parquet: %s", en_parquet_dir)

    # Save combined stats
    summary_path = Path(cfg.output_dir) / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    log.info("Pipeline complete. Summary: %s", summary_path)


def _report_disk_usage(output_dir: str) -> None:
    """Log current disk usage of the output directory."""
    import subprocess
    try:
        result = subprocess.run(
            ["du", "-sh", output_dir],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log.info("  Current disk usage: %s", result.stdout.strip())
    except Exception:
        pass

    # Also report free space on the mount
    try:
        usage = shutil.disk_usage(output_dir)
        free_gb = usage.free / (1024 ** 3)
        log.info("  Free space on mount: %.1f GB", free_gb)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Beetle-Data Pipeline: Decontamination to Pretokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline, all languages (storage-optimized: <300 GB peak)
  python -m pipeline.run_pipeline --project-root /path/to/PHD

  # Full pipeline, single language
  python -m pipeline.run_pipeline --lang pl --project-root /path/to/PHD

  # Stage 1 only (index build)
  python -m pipeline.run_pipeline --stage 1 --project-root /path/to/PHD

  # No upload (keep local files)
  python -m pipeline.run_pipeline --lang pl --no-upload --project-root /path/to/PHD
        """
    )

    parser.add_argument("--stage", type=str, nargs="+", default=["1", "2", "3"],
                        help="Pipeline stages to run (1, 2, 3 for static; A, B, C, D for curriculum)")
    parser.add_argument("--lang", type=str, nargs="+", default=None,
                        help="Language code(s) to process (default: all 19)")
    parser.add_argument("--node-id", type=int, default=None,
                        help="SLURM node ID (0-3) for assigned languages")
    parser.add_argument("--project-root", type=str, default="",
                        help="Project root (for local benchmark files)")
    parser.add_argument("--output-dir", type=str, default="pipeline_output",
                        help="Base output directory")
    parser.add_argument("--index", type=str, default=None,
                        help="Path to pre-built benchmark_13gram.pkl (skips Stage 1)")
    parser.add_argument("--hf-user", type=str, default="Beetle-Data",
                        help="HuggingFace user/org")
    parser.add_argument("--num-workers", type=int, default=24,
                        help="Multiprocessing workers per stage")
    parser.add_argument("--target-words", type=int, default=None,
                        help="Override target words per language")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for pretokenization")
    parser.add_argument("--no-ud", action="store_true",
                        help="Skip UD Treebanks in Stage 1")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HuggingFace upload (keep local files)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep all local files (no deletion after upload)")
    parser.add_argument("--dataset-suffix", type=str, default="24B",
                        help="HF dataset repo suffix (default: 24B)")
    parser.add_argument("--skip-disk-check", action="store_true",
                        help="Skip the 300 GB disk space pre-flight check (for smoke tests)")

    # BeetleStream v2 arguments
    parser.add_argument("--stream-mode", type=str, default="static",
                        choices=["static", "curriculum"],
                        help="Data streaming mode (default: static)")
    parser.add_argument("--beetlestream-config", type=str, default=None,
                        help="Path to beetlestream_curriculum.yaml")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Override teacher model (e.g., meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL for Stage A")

    args = parser.parse_args()

    # Determine languages
    if args.lang:
        languages = args.lang
    elif args.node_id is not None:
        languages = langs_for_node(args.node_id)
    elif 1 in args.stage and len(args.stage) == 1:
        languages = []
    else:
        # Default: all non-English languages (EN is handled automatically)
        languages = list(NON_EN_LANGS)

    # Build config
    cfg = PipelineConfig(
        project_root=args.project_root,
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
    if args.target_words:
        cfg.target_words_per_lang = args.target_words

    # Set index path
    if args.index:
        cfg.benchmark_index_path = args.index
    else:
        cfg.benchmark_index_path = str(Path(args.output_dir) / "benchmark_13gram.pkl")

    # Parse BeetleStream config if provided
    if args.beetlestream_config:
        import yaml
        with open(args.beetlestream_config, "r") as f:
            bs_yaml = yaml.safe_load(f)
        bs_section = bs_yaml.get("beetlestream", {})
        teacher = bs_section.get("teacher", {})
        student = bs_section.get("student", {})
        heuristics = bs_section.get("heuristics", {})
        indexing = bs_section.get("indexing", {})
        cfg.beetlestream = BeetleStreamConfig(
            teacher_model=teacher.get("model", "meta-llama/Meta-Llama-3-70B-Instruct"),
            teacher_base_url=args.vllm_url,
            teacher_sample_size=teacher.get("sample_size", 500_000),
            teacher_batch_size=teacher.get("batch_size", 16),
            teacher_max_concurrent=teacher.get("max_concurrent", 32),
            kidlm_repo=teacher.get("calibration", {}).get("kidlm_repo", "tafseer-nayeem/KidLM-corpus"),
            clc_repo=teacher.get("calibration", {}).get("clc_repo", "ADALM/CLC-L1-CEFR"),
            kidlm_samples=teacher.get("calibration", {}).get("kidlm_samples", 20),
            samples_per_cefr_level=teacher.get("calibration", {}).get("samples_per_cefr_level", 3),
            embedding_model=student.get("embedding_model", "intfloat/multilingual-e5-base"),
            embedding_dim=student.get("embedding_dim", 768),
            student_batch_size=student.get("batch_size", 256),
            stopword_density_min=heuristics.get("stopword_density_range", [0.03, 0.60])[0],
            stopword_density_max=heuristics.get("stopword_density_range", [0.03, 0.60])[1],
            max_fk_grade=heuristics.get("max_fk_grade", 18.0),
            min_script_consistency=heuristics.get("min_script_consistency", 0.60),
            min_unique_5gram_ratio=heuristics.get("min_unique_5gram_ratio", 0.30),
            n_clusters=indexing.get("n_clusters", 200),
            index_shard_size=indexing.get("shard_size", 10_000),
            upload_indexed_to_hf=indexing.get("upload_to_hf", True),
        )
    if args.teacher_model and cfg.beetlestream:
        cfg.beetlestream = BeetleStreamConfig(
            **{**cfg.beetlestream.__dict__, "teacher_model": args.teacher_model}
        )

    cfg.stream_mode = args.stream_mode

    # Pre-flight disk check
    required_gb = 4000 if args.stream_mode == "curriculum" else 300
    if not args.skip_disk_check and not check_disk_space(args.output_dir, required_gb=required_gb):
        log.error("Insufficient disk space. Need at least %d GB free.", required_gb)
        sys.exit(1)

    # Normalize stages
    stages = [s.upper() if s.isalpha() else s for s in args.stage]
    numeric_stages = [int(s) for s in stages if s.isdigit()]
    letter_stages = [s for s in stages if s.isalpha()]

    pipeline_t0 = time.time()

    # Stage 1: Build benchmark index
    if "1" in args.stage or 1 in numeric_stages:
        index_path = run_stage_1(cfg)
        cfg.benchmark_index_path = index_path

    # Stage 2: Decontaminate
    if "2" in args.stage or 2 in numeric_stages:
        if not Path(cfg.benchmark_index_path).exists():
            log.error("Benchmark index not found at %s. Run Stage 1 first.",
                      cfg.benchmark_index_path)
            sys.exit(1)

        # When running with letter stages (curriculum), don't auto-combine with Stage 3
        if "3" in args.stage and "A" not in letter_stages:
            run_storage_optimized(languages, cfg)
        else:
            for lang in languages:
                if lang not in LANG_REGISTRY:
                    continue
                run_stage_2_lang(lang, cfg)
            if any(l != "en" for l in languages):
                en_dir = Path(cfg.output_dir) / "decontaminated" / "en"
                if not en_dir.exists() or not list(en_dir.glob("*.parquet")):
                    run_stage_2_lang("en", cfg)

    # BeetleStream Stages A-D
    if "A" in letter_stages:
        # Discover available languages from decontaminated output
        decon_path = Path(cfg.output_dir) / "decontaminated"
        available_langs = sorted([
            d.name for d in decon_path.iterdir()
            if d.is_dir() and list(d.glob("*.parquet"))
        ]) if decon_path.exists() else languages
        run_stage_A(cfg, available_langs)

    if "B" in letter_stages:
        run_stage_B(cfg, languages if languages else None)

    if "C" in letter_stages:
        run_stage_C(cfg)

    if "D" in letter_stages:
        for lang in languages:
            if lang not in LANG_REGISTRY:
                continue
            run_stage_D_lang(lang, cfg)

    # Stage 3: Pretokenize (works for both static and curriculum modes)
    if "3" in args.stage and "A" in letter_stages:
        # Curriculum mode: pretokenize from indexed shards (handled separately)
        for lang in languages:
            if lang not in LANG_REGISTRY or lang == "en":
                continue
            run_stage_3_lang(lang, cfg)
    elif "3" in args.stage and 3 in numeric_stages and "A" not in letter_stages:
        # Static mode: pretokenize from raw decontaminated Parquet
        if 2 not in numeric_stages:
            for lang in languages:
                if lang not in LANG_REGISTRY or lang == "en":
                    continue
                run_stage_3_lang(lang, cfg)

    total_time = time.time() - pipeline_t0
    log.info("=" * 60)
    log.info("Pipeline complete. Total time: %.1f min (%.1f hours)",
             total_time / 60, total_time / 3600)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
    # Force-exit to avoid PyGILState_Release crash from orphaned httpx/aiohttp
    # background threads left by HuggingFace datasets streaming iterators.
    import os
    os._exit(0)
