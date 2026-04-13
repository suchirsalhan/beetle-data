"""
config.py — Language registry and pipeline configuration.

Central source-of-truth for:
  - FineWeb-2 language names (matches tok/multi-train-tok.py)
  - Tokenizer HuggingFace repo IDs (Beetle-Data/tokenizer-{lang}-en)
  - FLORES-200 column tags (matches beetle-analyze/analyze/ppl_utils.py)
  - Benchmark definitions for decontamination
  - Node assignment for multi-node SLURM runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# BeetleStream v2: Stream Mode
# ═══════════════════════════════════════════════════════════════════════════════

class StreamMode(str, Enum):
    """Data streaming mode for the pipeline."""
    STATIC = "static"           # Original: decontaminate → pretokenize → Arrow
    RANDOM = "random_stream"    # Naive streaming baseline (no curriculum)
    CURRICULUM = "curriculum"    # Full BeetleStream v2 with quality/topic indexing


@dataclass(frozen=True)
class BeetleStreamConfig:
    """Configuration for BeetleStream v2 pedagogical pipeline (Stages A-D)."""

    # Teacher annotation (Stage A)
    teacher_model: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    teacher_backend: str = "vllm"
    teacher_base_url: str = "http://localhost:8000/v1"
    teacher_sample_size: int = 500_000
    teacher_batch_size: int = 16
    teacher_max_concurrent: int = 32
    kidlm_repo: str = "tafseer-nayeem/KidLM-corpus"
    clc_repo: str = "ADALM/CLC-L1-CEFR"
    kidlm_samples: int = 20
    samples_per_cefr_level: int = 3

    # Student model (Stage C)
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    student_batch_size: int = 256

    # Heuristic filters
    stopword_density_min: float = 0.03
    stopword_density_max: float = 0.60
    max_fk_grade: float = 18.0
    min_script_consistency: float = 0.60
    min_unique_5gram_ratio: float = 0.30

    # Indexing (Stage D)
    n_clusters: int = 200
    index_shard_size: int = 10_000
    upload_indexed_to_hf: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Language Registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LangConfig:
    """Per-language metadata needed by every pipeline stage."""
    code: str               # 2-letter ISO 639-1 (pl, nl, es, …)
    name: str               # Human-readable
    fw2_name: str           # FineWeb-2 subset name (pol_Latn, nld_Latn, …)
    flores_tag: str = ""    # FLORES-200 column suffix (nld_Latn → sentence_nld_Latn)
    is_english: bool = False
    tier: str = "core"      # "core" | "extension" | "low_resource"


# Mirrors tok/multi-train-tok.py:35-61 exactly
LANG_REGISTRY: Dict[str, LangConfig] = {
    "pl": LangConfig("pl", "Polish",     "pol_Latn", "pol_Latn"),
    "nl": LangConfig("nl", "Dutch",      "nld_Latn", "nld_Latn"),
    "es": LangConfig("es", "Spanish",    "spa_Latn", "spa_Latn"),
    "el": LangConfig("el", "Greek",      "ell_Grek", "ell_Grek"),
    "ja": LangConfig("ja", "Japanese",   "jpn_Jpan", "jpn_Jpan"),
    "fr": LangConfig("fr", "French",     "fra_Latn", "fra_Latn"),
    "zh": LangConfig("zh", "Chinese",    "cmn_Hani", "zho_Hans"),
    "de": LangConfig("de", "German",     "deu_Latn", "deu_Latn"),
    "it": LangConfig("it", "Italian",    "ita_Latn", "ita_Latn"),
    "eu": LangConfig("eu", "Basque",     "eus_Latn", "eus_Latn"),
    "tr": LangConfig("tr", "Turkish",    "tur_Latn", "tur_Latn"),
    "id": LangConfig("id", "Indonesian", "ind_Latn", "ind_Latn"),
    "tl": LangConfig("tl", "Tagalog",    "fil_Latn", "fil_Latn"),
    "fa": LangConfig("fa", "Persian",    "fas_Arab", "pes_Arab"),
    "hi": LangConfig("hi", "Hindi",      "hin_Deva", "hin_Deva"),
    "ta": LangConfig("ta", "Tamil",      "tam_Taml", "tam_Taml"),
    "sv": LangConfig("sv", "Swedish",    "swe_Latn", "swe_Latn"),
    "ru": LangConfig("ru", "Russian",    "rus_Cyrl", "rus_Cyrl"),
    "ca": LangConfig("ca", "Catalan",    "cat_Latn", "cat_Latn"),
    "en": LangConfig("en", "English",    "",          "eng_Latn", is_english=True),
    # ── Core: Arabic ─────────────────────────────────────────────────────
    "ar": LangConfig("ar", "Arabic",     "arb_Arab", "arb_Arab"),
    # ── Extension languages ──────────────────────────────────────────────
    "ur": LangConfig("ur", "Urdu",       "urd_Arab", "urd_Arab", tier="extension"),
    "bn": LangConfig("bn", "Bengali",    "ben_Beng", "ben_Beng", tier="extension"),
    "cs": LangConfig("cs", "Czech",      "ces_Latn", "ces_Latn", tier="extension"),
    "gu": LangConfig("gu", "Gujarati",   "guj_Gujr", "guj_Gujr", tier="extension"),
    "th": LangConfig("th", "Thai",       "tha_Thai", "tha_Thai", tier="extension"),
    "vi": LangConfig("vi", "Vietnamese", "vie_Latn", "vie_Latn", tier="extension"),
    "ko": LangConfig("ko", "Korean",     "kor_Hang", "kor_Hang", tier="extension"),
    "da": LangConfig("da", "Danish",     "dan_Latn", "dan_Latn", tier="extension"),
    # ── Low-resource European ────────────────────────────────────────────
    "hu": LangConfig("hu", "Hungarian",  "hun_Latn", "hun_Latn", tier="low_resource"),
    "bg": LangConfig("bg", "Bulgarian",  "bul_Cyrl", "bul_Cyrl", tier="low_resource"),
    "hr": LangConfig("hr", "Croatian",   "hrv_Latn", "hrv_Latn", tier="low_resource"),
    "uk": LangConfig("uk", "Ukrainian",  "ukr_Cyrl", "ukr_Cyrl", tier="low_resource"),
    "sl": LangConfig("sl", "Slovenian",  "slv_Latn", "slv_Latn", tier="low_resource"),
    # ── Low-resource African ─────────────────────────────────────────────
    "so": LangConfig("so", "Somali",     "som_Latn", "som_Latn", tier="low_resource"),
    "am": LangConfig("am", "Amharic",    "amh_Ethi", "amh_Ethi", tier="low_resource"),
    "yo": LangConfig("yo", "Yoruba",     "yor_Latn", "yor_Latn", tier="low_resource"),
    "wo": LangConfig("wo", "Wolof",      "wol_Latn", "wol_Latn", tier="low_resource"),
}

ALL_LANGS = sorted(LANG_REGISTRY.keys())
NON_EN_LANGS = [c for c in ALL_LANGS if c != "en"]

# Tier-based language lists
CORE_LANGS = [c for c, lc in LANG_REGISTRY.items() if lc.tier == "core" and not lc.is_english]
EXTENSION_LANGS = [c for c, lc in LANG_REGISTRY.items() if lc.tier == "extension"]
LOW_RESOURCE_LANGS = [c for c, lc in LANG_REGISTRY.items() if lc.tier == "low_resource"]


def tokenizer_repo(lang: str, hf_user: str = "Beetle-Data") -> str:
    """Return HuggingFace tokenizer repo ID for a bilingual {lang}-en pair."""
    return f"{hf_user}/tokenizer-{lang}-en"


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Dataset Sources
# ═══════════════════════════════════════════════════════════════════════════════

FINEWEB_EDU = "HuggingFaceFW/fineweb-edu"
FINEWEB_2   = "HuggingFaceFW/fineweb-2"


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Definitions for Decontamination
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BenchmarkDef:
    """Describes one HuggingFace benchmark to decontaminate against."""
    name: str
    hf_id: str
    text_columns: List[str]
    split: str = "train"
    config_mode: str = "default"        # default | all_configs | per_lang | split_per_lang
    lang_configs: Optional[Dict] = None # per_lang / split_per_lang mappings
    local_path: Optional[str] = None    # for local TSV files


BENCHMARK_DEFS: List[BenchmarkDef] = [
    # --- FLORES-200 (perplexity eval) ---
    BenchmarkDef(
        name="flores200",
        hf_id="crystina-z/flores200",
        text_columns=[],  # populated dynamically from FLORES tags
        split="devtest",
        config_mode="flores",
    ),

    # --- BLiMP (English) ---
    BenchmarkDef(
        name="blimp_eng",
        hf_id="nyu-mll/blimp",
        text_columns=["sentence_good", "sentence_bad"],
        split="train",
        config_mode="all_configs",
    ),

    # --- ZhoBLiMP (Chinese) ---
    BenchmarkDef(
        name="zhoblimp",
        hf_id="Junrui1202/zhoblimp",
        text_columns=["sentence_good", "sentence_bad"],
        split="train",
        config_mode="all_configs",
    ),

    # --- BLiMP-NL (Dutch) ---
    BenchmarkDef(
        name="blimp_nl",
        hf_id="juletxara/blimp-nl",
        text_columns=["sentence_good", "sentence_bad"],
        split="train",
        config_mode="all_configs",
    ),

    # --- MultiBLiMP (multilingual minimal pairs) ---
    BenchmarkDef(
        name="multiblimp",
        hf_id="jumelet/multiblimp",
        text_columns=["sen", "wrong_sen"],
        split="train",
        config_mode="per_lang",
        lang_configs={
            "nld": "Dutch", "deu": "German", "fra": "French",
            "fas": "Persian", "bul": "Bulgarian",
        },
    ),

    # --- RuBLiMP (Russian) ---
    BenchmarkDef(
        name="rublimp",
        hf_id="RussianNLP/rublimp",
        text_columns=["source_sentence", "target_sentence"],
        split="train",
        config_mode="all_configs",
    ),

    # --- TurBLiMP (Turkish) ---
    BenchmarkDef(
        name="turblimp",
        hf_id="juletxara/turblimp",
        text_columns=["sentence_good", "sentence_bad"],
        split="train",
        config_mode="all_configs",
    ),

    # --- JBLiMP (Japanese) ---
    BenchmarkDef(
        name="jblimp",
        hf_id="polm-stability/jblimp",
        text_columns=["good_sentence", "bad_sentence"],
        split="train",
        config_mode="default",
    ),

    # --- SLING (Chinese) ---
    BenchmarkDef(
        name="sling",
        hf_id="suchirsalhan/SLING",
        text_columns=["sentence_good", "sentence_bad"],
        split="train",
        config_mode="default",
    ),

    # --- CLiMP (Chinese) ---
    # CLiMP uses unnamed columns: col 2 = sentence, col 3 = label (1=good, 0=bad)
    # Handled via special loader in benchmark_index.py
    BenchmarkDef(
        name="climp",
        hf_id="suchirsalhan/CLiMP",
        text_columns=[],  # special handling: label-filtered column
        split="train",
        config_mode="climp_special",
    ),

    # --- XCOMPS (compositionality minimal pairs) ---
    BenchmarkDef(
        name="xcomps",
        hf_id="fpadovani/xcomps-dataset",
        text_columns=["acceptable_sent", "unacceptable_sent"],
        split="train",
        config_mode="split_per_lang",
        lang_configs={
            "fra": "comps_fr", "deu": "comps_de", "ukr": "comps_uk",
            "zho": "comps_zh", "fas": "comps_fa",
        },
    ),

    # --- XNLI (NLI) ---
    BenchmarkDef(
        name="xnli",
        hf_id="xnli",
        text_columns=["premise", "hypothesis"],
        split="validation",
        config_mode="per_lang",
        lang_configs={
            "en": "en", "fr": "fr", "de": "de",
            "zh": "zh", "bg": "bg",
        },
    ),

    # --- MECO-L2 (reading time stimuli, local TSV) ---
    BenchmarkDef(
        name="meco_l2",
        hf_id="",
        text_columns=["FullText"],
        split="",
        config_mode="local_tsv",
        local_path="beetlelm-rts-private/stimuli/meco_l2_stims.tsv",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Global settings for the decontamination-to-pretokenization pipeline.

    Storage-optimized: processes languages one at a time, uploads Arrow
    datasets to HuggingFace, then deletes local files. Peak disk usage
    stays under 300 GB at any point.

    Storage flow per language:
      1. EN Parquet stays on disk (~70 GB, shared across all pairs)
      2. L1 Parquet written (~70 GB)
      3. L1 Arrow built (~64 GB) → uploaded to HF → deleted locally
      4. EN Arrow built (~32 GB) → uploaded to HF → deleted locally
      5. L1 Parquet deleted
      Peak: ~270 GB  |  After cleanup: ~90 GB (EN Parquet + index + cache)
    """

    # Paths
    project_root: str = ""
    output_dir: str = "pipeline_output"
    benchmark_index_path: str = "pipeline_output/benchmark_13gram.pkl"

    # Decontamination
    ngram_size: int = 13
    min_doc_chars: int = 200
    max_doc_chars: int = 100_000

    # Token targets (per language, approximate via word count)
    target_words_per_lang: int = 22_000_000_000   # ~28B tokens at ~1.3 BPE/word
    word_overshoot_factor: float = 1.15            # 15% overshoot

    # Pretokenization (must match beetlelm)
    seq_len: int = 512
    chunk_len: int = 513    # seq_len + 1 (input + label)

    # Bilingual token ratio (L1 : EN) — 50:50 split
    l1_ratio: float = 1 / 2
    en_ratio: float = 1 / 2

    # HuggingFace
    hf_user: str = "Beetle-Data"
    hf_token: str = ""

    # Storage management — upload to HF and delete local files
    upload_to_hf: bool = True               # upload Arrow datasets to HF after building
    cleanup_after_upload: bool = True        # delete local Arrow after successful upload
    cleanup_stage2_after_pretok: bool = True # delete L1 Parquet after both sides pretokenized
    hf_dataset_suffix: str = "24B"          # repo naming: Beetle-Data/{lang}-{suffix}
    max_local_disk_gb: int = 1000           # pre-flight check: abort if less than this free

    # Parallelization
    num_workers: int = 24
    shard_size: int = 50_000    # docs per Parquet shard
    batch_size: int = 1_000     # docs per worker batch

    # BeetleStream v2 (curriculum mode)
    stream_mode: str = "static"                           # "static" | "curriculum"
    beetlestream: Optional[BeetleStreamConfig] = None     # None → static mode

    def __post_init__(self):
        if not self.hf_token:
            import os
            self.hf_token = os.environ.get("HF_TOKEN", "")

    def hf_dataset_repo(self, lang: str, side: str) -> str:
        """Return HuggingFace dataset repo ID for a pretokenized dataset.

        Examples:
            hf_dataset_repo("pl", "l1") → "Beetle-Data/pl-24B"
            hf_dataset_repo("pl", "en") → "Beetle-Data/en-for-pl-24B"
        """
        if side == "l1":
            return f"{self.hf_user}/{lang}-{self.hf_dataset_suffix}"
        return f"{self.hf_user}/en-for-{lang}-{self.hf_dataset_suffix}"


def check_disk_space(path: str, required_gb: int = 300) -> bool:
    """Pre-flight check: verify enough disk space is available.

    Args:
        path: Directory to check (will be created if needed).
        required_gb: Minimum free space in GB.

    Returns:
        True if enough space, False otherwise.
    """
    import shutil
    from pathlib import Path as _Path

    _Path(path).mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < required_gb:
        print(f"WARNING: Only {free_gb:.1f} GB free on {path}, need {required_gb} GB")
        return False
    print(f"Disk check OK: {free_gb:.1f} GB free on {path} (need {required_gb} GB)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Node Assignment for Multi-Node SLURM
# ═══════════════════════════════════════════════════════════════════════════════

NODE_ASSIGNMENTS: Dict[int, List[str]] = {
    0: ["pl", "nl", "es", "el", "ja"],
    1: ["fr", "zh", "de", "it", "eu"],
    2: ["tr", "id", "tl", "fa", "hi"],
    3: ["ta", "sv", "ru", "ca", "ar"],
    4: ["en"],  # English shared across all pairs
}

# Extension languages — separate pipeline runs (lower priority)
NODE_ASSIGNMENTS_EXTENSION: Dict[int, List[str]] = {
    0: ["ur", "bn", "cs", "gu"],
    1: ["th", "vi", "ko", "da"],
}

# Low-resource languages — separate pipeline runs (lowest priority)
NODE_ASSIGNMENTS_LOW_RESOURCE: Dict[int, List[str]] = {
    0: ["hu", "bg", "hr", "uk", "sl"],
    1: ["so", "am", "yo", "wo"],
}


def langs_for_node(node_id: int, tier: str = "core") -> List[str]:
    """Return the language codes assigned to a SLURM node.

    Args:
        node_id: SLURM node index.
        tier: "core" (default), "extension", or "low_resource".
    """
    assignments = {
        "core": NODE_ASSIGNMENTS,
        "extension": NODE_ASSIGNMENTS_EXTENSION,
        "low_resource": NODE_ASSIGNMENTS_LOW_RESOURCE,
    }
    return assignments.get(tier, NODE_ASSIGNMENTS).get(node_id, [])
