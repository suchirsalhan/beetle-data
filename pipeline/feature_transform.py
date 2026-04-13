"""
feature_transform.py — Stage B: Convert teacher annotations to structured features.

Reads JSONL annotation files from Stage A, extracts numeric and boolean features
from the pedagogical rubric JSON, and writes a feature Parquet per language.

Feature schema:
  - doc_id:          int64     (from original decontaminated data)
  - text:            utf8      (original document text, for student model training)
  - quality_score:   int8      (0-5, additive pedagogical quality)
  - difficulty_level: int8     (1-3, readability level)
  - vocab_complexity: bool     (complex vocabulary present)
  - has_jargon:      bool      (technical jargon present)
  - engagement:      bool      (engaging for learners)
  - topic_label:     utf8      (science / math / social_science / language / other)

Usage:
    python -m pipeline.run_pipeline --stage B --output-dir pipeline_output

    # Direct invocation:
    python -m pipeline.feature_transform --output-dir pipeline_output
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("FeatureTransform")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_quality_score(annotation: dict) -> int:
    """Compute additive quality score (0-5) from annotation JSON.

    Adds 1 point for each of:
      - readability.elementary = true  (accessible to young learners)
      - comprehension.easy = true      (easy to understand)
      - pedagogy.curriculum_relevant = true  (curriculum-aligned)
      - engagement.engaging = true     (engaging for students)
      - vocabulary.simple = true       (simple vocabulary)

    Returns integer 0-5.
    """
    score = 0

    readability = annotation.get("readability", {})
    if readability.get("elementary", False):
        score += 1

    comprehension = annotation.get("comprehension", {})
    if comprehension.get("easy", False):
        score += 1

    pedagogy = annotation.get("pedagogy", {})
    if pedagogy.get("curriculum_relevant", False):
        score += 1

    engagement = annotation.get("engagement", {})
    if engagement.get("engaging", False):
        score += 1

    vocabulary = annotation.get("vocabulary", {})
    if vocabulary.get("simple", False):
        score += 1

    return min(score, 5)


def compute_difficulty_level(annotation: dict) -> int:
    """Compute difficulty level (1-3) from readability flags.

    Returns the highest level for which the text is readable:
      - 3 if high = true
      - 2 if middle = true (and high is false)
      - 1 if elementary = true (and middle/high are false)
      - 2 as default if none are true (moderate assumption)
    """
    readability = annotation.get("readability", {})

    if readability.get("high", False):
        return 3
    if readability.get("middle", False):
        return 2
    if readability.get("elementary", False):
        return 1
    return 2  # default: moderate difficulty


def extract_topic_label(annotation: dict) -> str:
    """Extract primary topic label from annotation.

    Returns first true topic from: science, math, social_science, language.
    Falls back to "other" if none are true.
    """
    topic = annotation.get("topic", {})

    if topic.get("science", False):
        return "science"
    if topic.get("math", False):
        return "math"
    if topic.get("social_science", False):
        return "social_science"
    if topic.get("language", False):
        return "language"
    return "other"


def extract_features(annotation: dict) -> Dict[str, Any]:
    """Extract all features from a single annotation JSON.

    Returns dict with: quality_score, difficulty_level, vocab_complexity,
    has_jargon, engagement, topic_label.
    """
    vocab = annotation.get("vocabulary", {})
    engage = annotation.get("engagement", {})

    return {
        "quality_score": compute_quality_score(annotation),
        "difficulty_level": compute_difficulty_level(annotation),
        "vocab_complexity": bool(vocab.get("complex", False)),
        "has_jargon": bool(vocab.get("jargon", False)),
        "engagement": bool(engage.get("engaging", False)),
        "topic_label": extract_topic_label(annotation),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("text", pa.utf8()),
    ("quality_score", pa.int8()),
    ("difficulty_level", pa.int8()),
    ("vocab_complexity", pa.bool_()),
    ("has_jargon", pa.bool_()),
    ("engagement", pa.bool_()),
    ("topic_label", pa.utf8()),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Transform Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransformStats:
    """Track transform progress."""
    total_processed: int = 0
    total_rejected: int = 0
    quality_distribution: Dict[int, int] = None
    difficulty_distribution: Dict[int, int] = None
    topic_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {i: 0 for i in range(6)}
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {i: 0 for i in range(1, 4)}
        if self.topic_distribution is None:
            self.topic_distribution = {}

    def to_dict(self) -> dict:
        return {
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "quality_distribution": self.quality_distribution,
            "difficulty_distribution": self.difficulty_distribution,
            "topic_distribution": self.topic_distribution,
        }


def transform_language(
    annotations_dir: str,
    decontaminated_dir: str,
    output_dir: str,
    lang: str,
) -> TransformStats:
    """Transform annotations for one language into feature Parquet.

    Reads annotation JSONL files, looks up original text from decontaminated
    Parquet (via doc_id), extracts features, and writes features.parquet.

    Args:
        annotations_dir: Path to pipeline_output/annotations/
        decontaminated_dir: Path to pipeline_output/decontaminated/
        output_dir: Path to pipeline_output/features/
        lang: Language code.
    """
    stats = TransformStats()
    lang_ann_dir = Path(annotations_dir) / lang
    lang_decon_dir = Path(decontaminated_dir) / lang

    if not lang_ann_dir.exists():
        log.warning("No annotations for %s", lang)
        return stats

    # Read all annotations
    annotations: Dict[int, dict] = {}  # doc_id → annotation
    for jsonl_file in sorted(lang_ann_dir.glob("annotations_*.jsonl")):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    doc_id = record["doc_id"]
                    annotations[doc_id] = record["annotation"]
                except (json.JSONDecodeError, KeyError):
                    stats.total_rejected += 1
                    continue

    log.info("  %s: loaded %d annotations", lang, len(annotations))

    # Look up original texts from decontaminated Parquet
    doc_texts: Dict[int, str] = {}
    if lang_decon_dir.exists():
        for shard_path in sorted(lang_decon_dir.glob("*.parquet")):
            try:
                table = pq.read_table(shard_path, columns=["text", "doc_id"])
                for i in range(table.num_rows):
                    did = table.column("doc_id")[i].as_py()
                    if did in annotations:
                        doc_texts[did] = table.column("text")[i].as_py()
                del table
            except Exception:
                continue

    # Extract features
    rows = {
        "doc_id": [], "text": [], "quality_score": [], "difficulty_level": [],
        "vocab_complexity": [], "has_jargon": [], "engagement": [],
        "topic_label": [],
    }

    for doc_id, annotation in annotations.items():
        features = extract_features(annotation)

        # Validate ranges
        if not (0 <= features["quality_score"] <= 5):
            stats.total_rejected += 1
            continue
        if not (1 <= features["difficulty_level"] <= 3):
            stats.total_rejected += 1
            continue

        text = doc_texts.get(doc_id, "")

        rows["doc_id"].append(doc_id)
        rows["text"].append(text)
        rows["quality_score"].append(features["quality_score"])
        rows["difficulty_level"].append(features["difficulty_level"])
        rows["vocab_complexity"].append(features["vocab_complexity"])
        rows["has_jargon"].append(features["has_jargon"])
        rows["engagement"].append(features["engagement"])
        rows["topic_label"].append(features["topic_label"])

        stats.total_processed += 1
        stats.quality_distribution[features["quality_score"]] += 1
        stats.difficulty_distribution[features["difficulty_level"]] += 1
        topic = features["topic_label"]
        stats.topic_distribution[topic] = stats.topic_distribution.get(topic, 0) + 1

    # Write Parquet
    out_path = Path(output_dir) / lang
    out_path.mkdir(parents=True, exist_ok=True)
    feature_path = out_path / "features.parquet"

    table = pa.table(rows, schema=FEATURE_SCHEMA)
    pq.write_table(table, feature_path, compression="snappy")

    log.info("  %s: wrote %d features to %s (rejected %d)",
             lang, stats.total_processed, feature_path, stats.total_rejected)

    return stats


def transform_all_languages(
    output_dir: str,
    langs: Optional[List[str]] = None,
) -> Dict[str, TransformStats]:
    """Transform annotations for all languages.

    Args:
        output_dir: Pipeline output directory.
        langs: Language codes (auto-detected if None).

    Returns:
        Dict mapping lang → TransformStats.
    """
    annotations_dir = str(Path(output_dir) / "annotations")
    decontaminated_dir = str(Path(output_dir) / "decontaminated")
    features_dir = str(Path(output_dir) / "features")

    if langs is None:
        ann_path = Path(annotations_dir)
        if ann_path.exists():
            langs = sorted([d.name for d in ann_path.iterdir()
                          if d.is_dir() and list(d.glob("*.jsonl"))])
        else:
            log.error("No annotations at %s", annotations_dir)
            return {}

    all_stats = {}
    for lang in langs:
        log.info("Transforming features for %s...", lang)
        stats = transform_language(
            annotations_dir, decontaminated_dir, features_dir, lang,
        )
        all_stats[lang] = stats

    # Write combined stats
    stats_path = Path(output_dir) / "features" / "transform_stats.json"
    with open(stats_path, "w") as f:
        json.dump({lang: s.to_dict() for lang, s in all_stats.items()}, f, indent=2)

    total = sum(s.total_processed for s in all_stats.values())
    log.info("Feature transform complete: %d features across %d languages",
             total, len(all_stats))

    return all_stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage B: Feature Transform")
    parser.add_argument("--output-dir", type=str, default="pipeline_output")
    parser.add_argument("--lang", type=str, nargs="+", default=None)
    args = parser.parse_args()

    transform_all_languages(args.output_dir, args.lang)


if __name__ == "__main__":
    main()
