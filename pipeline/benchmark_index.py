"""
benchmark_index.py — Stage 1: Build 13-gram index from all evaluation benchmarks.

Loads all benchmarks defined in config.BENCHMARK_DEFS, extracts every 13-gram
from their text columns, and serializes the combined index to disk. This is a
one-time operation (~5 minutes on a single node).

The BenchmarkIndex is loaded into memory at the start of Stage 2 and shared
read-only across forked worker processes for contamination checking.

Usage:
    python -m pipeline.benchmark_index --output benchmark_13gram.pkl
    python -m pipeline.benchmark_index --output benchmark_13gram.pkl --project-root /path/to/PHD
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from .config import (
    BENCHMARK_DEFS,
    LANG_REGISTRY,
    BenchmarkDef,
    PipelineConfig,
)
from .utils import extract_ngrams_from_text, tokenize_for_ngrams, extract_ngrams

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("BenchmarkIndex")


class BenchmarkIndex:
    """13-gram index built from evaluation benchmarks for decontamination.

    Attributes:
        ngram_size: Size of n-grams (default 13).
        index: Set of n-gram tuples for O(1) membership testing.
        stats: Per-benchmark counts for reporting.
    """

    def __init__(self, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self.index: Set[Tuple[str, ...]] = set()
        self.stats: Dict[str, int] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def add_texts(self, benchmark_name: str, texts: List[str]) -> int:
        """Extract n-grams from a list of texts and add to the index.

        Returns the number of new unique n-grams added.
        """
        before = len(self.index)
        for text in texts:
            if not text or not text.strip():
                continue
            ngrams = extract_ngrams_from_text(text, self.ngram_size)
            self.index.update(ngrams)
        added = len(self.index) - before
        self.stats[benchmark_name] = self.stats.get(benchmark_name, 0) + added
        return added

    def is_contaminated(self, text: str) -> bool:
        """Check if a document text contains any benchmark 13-gram."""
        if not self.index:
            return False
        tokens = tokenize_for_ngrams(text)
        ngrams = extract_ngrams(tokens, self.ngram_size)
        return bool(ngrams & self.index)

    def find_overlapping_ngrams(self, text: str) -> List[Tuple[str, ...]]:
        """Return all benchmark n-grams found in the text (for post-hoc analysis)."""
        if not self.index:
            return []
        tokens = tokenize_for_ngrams(text)
        ngrams = extract_ngrams(tokens, self.ngram_size)
        return sorted(ngrams & self.index)

    def save(self, path: str) -> None:
        """Serialize index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "stats": self.stats,
                         "ngram_size": self.ngram_size}, f, protocol=5)
        log.info("Saved index (%d n-grams) to %s", len(self.index), path)

    @classmethod
    def load(cls, path: str) -> "BenchmarkIndex":
        """Deserialize index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(ngram_size=data["ngram_size"])
        obj.index = data["index"]
        obj.stats = data["stats"]
        log.info("Loaded index (%d n-grams) from %s", len(obj.index), path)
        return obj

    def summary(self) -> str:
        """Human-readable summary of the index."""
        lines = [f"BenchmarkIndex: {len(self.index):,} unique {self.ngram_size}-grams"]
        for name, count in sorted(self.stats.items()):
            lines.append(f"  {name}: +{count:,} n-grams")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hf_texts(bdef: BenchmarkDef) -> List[str]:
    """Load text strings from a HuggingFace benchmark dataset."""
    from datasets import load_dataset, get_dataset_config_names

    texts: List[str] = []

    if bdef.config_mode == "flores":
        # FLORES-200: one column per language, load all relevant columns
        ds = load_dataset(bdef.hf_id, "all", split="devtest",
                          trust_remote_code=False)
        for lc in LANG_REGISTRY.values():
            if lc.flores_tag:
                col = f"sentence_{lc.flores_tag}"
                if col in ds.column_names:
                    texts.extend(s for s in ds[col] if isinstance(s, str) and s.strip())
                    log.info("  FLORES %s: %d sentences", lc.flores_tag,
                             sum(1 for s in ds[col] if isinstance(s, str) and s.strip()))
        return texts

    if bdef.config_mode == "all_configs":
        # Load every config (e.g., each BLiMP phenomenon)
        try:
            configs = get_dataset_config_names(bdef.hf_id)
        except Exception:
            configs = [None]
        for cfg in configs:
            try:
                ds = load_dataset(bdef.hf_id, cfg, split=bdef.split)
            except Exception:
                # Some splits may not exist for all configs
                try:
                    ds = load_dataset(bdef.hf_id, cfg)
                    # Take first available split
                    split_name = list(ds.keys())[0]
                    ds = ds[split_name]
                except Exception as e:
                    log.warning("  Skipping config %s/%s: %s", bdef.hf_id, cfg, e)
                    continue
            for col in bdef.text_columns:
                if col in ds.column_names:
                    texts.extend(s for s in ds[col] if isinstance(s, str) and s.strip())
        return texts

    if bdef.config_mode == "per_lang":
        # Load per-language configs
        for lang_key, lang_name in (bdef.lang_configs or {}).items():
            try:
                ds = load_dataset(bdef.hf_id, lang_key, split=bdef.split)
            except Exception:
                try:
                    ds = load_dataset(bdef.hf_id, lang_name, split=bdef.split)
                except Exception as e:
                    log.warning("  Skipping %s/%s: %s", bdef.hf_id, lang_key, e)
                    continue
            for col in bdef.text_columns:
                if col in ds.column_names:
                    texts.extend(s for s in ds[col] if isinstance(s, str) and s.strip())
        return texts

    if bdef.config_mode == "split_per_lang":
        # Each language has its own split/config name
        for lang_key, split_name in (bdef.lang_configs or {}).items():
            try:
                ds = load_dataset(bdef.hf_id, split_name, split=bdef.split)
            except Exception:
                try:
                    ds = load_dataset(bdef.hf_id, split=split_name)
                except Exception as e:
                    log.warning("  Skipping %s/%s: %s", bdef.hf_id, split_name, e)
                    continue
            for col in bdef.text_columns:
                if col in ds.column_names:
                    texts.extend(s for s in ds[col] if isinstance(s, str) and s.strip())
        return texts

    if bdef.config_mode == "climp_special":
        # CLiMP uses unnamed CSV columns: col index 2 = sentence, col index 3 = label
        # label 1 = grammatical (good), label 0 = ungrammatical (bad)
        # We extract ALL sentences (both good and bad) for decontamination
        try:
            ds = load_dataset(bdef.hf_id, split=bdef.split)
            # Try common column name patterns for CLiMP
            for col_name in ds.column_names:
                col_data = ds[col_name]
                # Heuristic: the sentence column has long strings
                sample = [x for x in col_data[:10] if isinstance(x, str)]
                if sample and any(len(s) > 10 for s in sample):
                    texts.extend(s for s in col_data if isinstance(s, str) and s.strip())
        except Exception as e:
            log.warning("  CLiMP load failed (may have inconsistent CSVs): %s", e)
            # Fallback: try loading individual CSV files if available
            try:
                import pandas as _pd
                from huggingface_hub import hf_hub_download, list_repo_files
                repo_files = list_repo_files(bdef.hf_id, repo_type="dataset")
                csv_files = [f for f in repo_files if f.endswith(".csv")]
                for csv_f in csv_files:
                    try:
                        path = hf_hub_download(bdef.hf_id, csv_f, repo_type="dataset")
                        df = _pd.read_csv(path, header=None)
                        # Column 2 is typically the sentence
                        if len(df.columns) > 2:
                            col_data = df.iloc[:, 2].dropna().astype(str)
                            texts.extend(s for s in col_data if len(s.strip()) > 5)
                    except Exception:
                        continue
            except Exception as e2:
                log.warning("  CLiMP fallback also failed: %s", e2)
        return texts

    # Default: single config, single split
    try:
        ds = load_dataset(bdef.hf_id, split=bdef.split)
    except Exception:
        ds = load_dataset(bdef.hf_id)
        split_name = list(ds.keys())[0]
        ds = ds[split_name]
    for col in bdef.text_columns:
        if col in ds.column_names:
            texts.extend(s for s in ds[col] if isinstance(s, str) and s.strip())
    return texts


def _load_hf_tsv(bdef: BenchmarkDef) -> List[str]:
    """Load text strings from a TSV file hosted on HuggingFace.

    Downloads the TSV via huggingface_hub and reads the specified text columns.
    Used for MECO-L2 (suchirsalhan/MECO, file: meco_l2_stims.tsv).

    Columns in the MECO TSV: itemid, wordnum, word, FullText, FullTextMarked
    FullText repeats for every word position in a passage → deduplicated.
    """
    from huggingface_hub import hf_hub_download

    try:
        tsv_path = hf_hub_download(
            repo_id=bdef.hf_id,
            filename=bdef.local_path,
            repo_type="dataset",
        )
    except Exception as e:
        log.warning("Could not download %s/%s from HuggingFace: %s",
                    bdef.hf_id, bdef.local_path, e)
        return []

    df = pd.read_csv(tsv_path, sep="\t")
    texts: List[str] = []
    for col in bdef.text_columns:
        if col in df.columns:
            texts.extend(s for s in df[col].dropna().astype(str) if s.strip())
        else:
            log.warning("  Column '%s' not found in %s (available: %s)",
                        col, bdef.local_path, list(df.columns))
    # Deduplicate: MECO repeats FullText once per word position in the passage
    texts = list(set(texts))
    log.info("  MECO HF TSV: %d unique passages loaded from %s/%s",
             len(texts), bdef.hf_id, bdef.local_path)
    return texts


def _load_local_tsv(bdef: BenchmarkDef, project_root: str) -> List[str]:
    """Load text strings from a local TSV file (backwards-compatibility fallback)."""
    tsv_path = Path(project_root) / bdef.local_path
    if not tsv_path.exists():
        log.warning("Local file not found: %s", tsv_path)
        return []
    df = pd.read_csv(tsv_path, sep="\t")
    texts: List[str] = []
    for col in bdef.text_columns:
        if col in df.columns:
            texts.extend(s for s in df[col].dropna().astype(str) if s.strip())
    # Deduplicate (MECO has repeated FullText for each word position)
    texts = list(set(texts))
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# UD Treebanks Loader
# ═══════════════════════════════════════════════════════════════════════════════

# UD treebank names for each language we train on
UD_TREEBANKS: Dict[str, List[str]] = {
    "en": ["en_ewt", "en_gum"],
    "nl": ["nl_alpino", "nl_lassysmall"],
    "de": ["de_gsd", "de_hdt"],
    "fr": ["fr_gsd", "fr_sequoia"],
    "es": ["es_gsd", "es_ancora"],
    "it": ["it_isdt", "it_vit"],
    "pl": ["pl_pdb", "pl_lfg"],
    "ru": ["ru_syntagrus", "ru_gsd"],
    "zh": ["zh_gsd", "zh_gsdsimp"],
    "ja": ["ja_gsd", "ja_bccwj"],
    "tr": ["tr_imst", "tr_boun"],
    "id": ["id_gsd", "id_csui"],
    "el": ["el_gdt"],
    "ca": ["ca_ancora"],
    "sv": ["sv_talbanken"],
    "fa": ["fa_perdt", "fa_seraji"],
    "hi": ["hi_hdtb"],
    "ta": ["ta_ttb"],
    "eu": ["eu_bdt"],
}


def _load_ud_treebank_texts() -> List[str]:
    """Load sentence texts from UD Treebanks for all target languages."""
    from datasets import load_dataset

    texts: List[str] = []
    for lang, treebanks in UD_TREEBANKS.items():
        for tb in treebanks:
            for split in ["train", "validation", "test"]:
                try:
                    ds = load_dataset("universal_dependencies", tb, split=split)
                    for row in ds:
                        text = row.get("text", "")
                        if isinstance(text, str) and text.strip():
                            texts.append(text)
                except Exception as e:
                    log.debug("  UD %s/%s: %s", tb, split, e)
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# Build Full Index
# ═══════════════════════════════════════════════════════════════════════════════

def build_benchmark_index(
    cfg: Optional[PipelineConfig] = None,
    include_ud: bool = True,
) -> BenchmarkIndex:
    """Build the complete 13-gram benchmark index from all defined benchmarks.

    Args:
        cfg: Pipeline config (for project_root and ngram_size).
        include_ud: Whether to include UD Treebanks (slower, ~2 min extra).

    Returns:
        Populated BenchmarkIndex ready for is_contaminated() calls.
    """
    if cfg is None:
        cfg = PipelineConfig()

    idx = BenchmarkIndex(ngram_size=cfg.ngram_size)

    for bdef in tqdm(BENCHMARK_DEFS, desc="Loading benchmarks"):
        log.info("Loading benchmark: %s (%s)", bdef.name, bdef.hf_id or bdef.local_path)

        if bdef.config_mode == "hf_tsv":
            # TSV hosted on HuggingFace (e.g., suchirsalhan/MECO)
            texts = _load_hf_tsv(bdef)
        elif bdef.config_mode == "local_tsv":
            # Legacy: local TSV file (requires --project-root)
            texts = _load_local_tsv(bdef, cfg.project_root)
        else:
            texts = _load_hf_texts(bdef)

        if texts:
            added = idx.add_texts(bdef.name, texts)
            log.info("  %s: %d texts -> +%d n-grams (total: %d)",
                     bdef.name, len(texts), added, len(idx.index))
        else:
            log.warning("  %s: no texts loaded", bdef.name)

    # UD Treebanks (loaded separately as they use a different HF structure)
    if include_ud:
        log.info("Loading UD Treebanks...")
        ud_texts = _load_ud_treebank_texts()
        if ud_texts:
            added = idx.add_texts("ud_treebanks", ud_texts)
            log.info("  UD Treebanks: %d texts -> +%d n-grams (total: %d)",
                     len(ud_texts), added, len(idx.index))

    log.info("\n%s", idx.summary())
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Build 13-gram benchmark index for decontamination."
    )
    parser.add_argument("--output", type=str, default="pipeline_output/benchmark_13gram.pkl",
                        help="Output path for the serialized index")
    parser.add_argument("--project-root", type=str, default="",
                        help="Root of the PHD project (for local benchmark files)")
    parser.add_argument("--ngram-size", type=int, default=13,
                        help="N-gram size for decontamination (default: 13)")
    parser.add_argument("--no-ud", action="store_true",
                        help="Skip UD Treebanks (faster build)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        project_root=args.project_root,
        ngram_size=args.ngram_size,
        benchmark_index_path=args.output,
    )

    idx = build_benchmark_index(cfg, include_ud=not args.no_ud)
    idx.save(args.output)

    print(f"\nDone. Index saved to {args.output}")
    print(idx.summary())


if __name__ == "__main__":
    main()
