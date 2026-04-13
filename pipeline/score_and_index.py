"""
score_and_index.py — Stage D: Full corpus scoring + indexed shard creation.

Two-pass processing per language:
  Pass 1 (heuristic): Apply fast text filters to discard non-textual content.
  Pass 2 (student model): Embed surviving docs, predict quality/difficulty,
           cluster into topics, and write Hive-partitioned indexed shards.

Output structure:
  pipeline_output/indexed/
    lang={en}/
      topic={0..199}/
        shard_00000.parquet
    manifest.json
    cluster_centroids.pkl
    topic_distribution.json

Usage:
    python -m pipeline.run_pipeline --stage D --node-id 0 \\
        --output-dir pipeline_output --beetlestream-config configs/beetlestream_curriculum.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .heuristic_filters import HeuristicConfig, passes_heuristics

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("ScoreAndIndex")


# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

INDEXED_SCHEMA = pa.schema([
    ("text", pa.utf8()),
    ("url", pa.utf8()),
    ("doc_id", pa.int64()),
    ("quality", pa.float32()),
    ("difficulty", pa.int8()),
    ("topic_id", pa.int16()),
    ("engagement", pa.float32()),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndexStats:
    """Statistics for scoring and indexing one language."""
    lang: str = ""
    total_docs: int = 0
    docs_after_heuristic: int = 0
    docs_indexed: int = 0
    heuristic_reject_rate: float = 0.0
    avg_quality: float = 0.0
    n_clusters: int = 0
    n_shards: int = 0
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "total_docs": self.total_docs,
            "docs_after_heuristic": self.docs_after_heuristic,
            "docs_indexed": self.docs_indexed,
            "heuristic_reject_rate": round(self.heuristic_reject_rate, 4),
            "avg_quality": round(self.avg_quality, 3),
            "n_clusters": self.n_clusters,
            "n_shards": self.n_shards,
            "wall_time_hours": round(self.wall_time_seconds / 3600, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1: Heuristic Filtering (multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_worker(args: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    """Worker function for parallel heuristic filtering.

    Reads one Parquet shard, applies heuristic filters, returns passing docs.
    """
    shard_path, lang, cfg_json = args
    cfg = HeuristicConfig(**json.loads(cfg_json))

    passing = []
    try:
        table = pq.read_table(shard_path)
        for i in range(table.num_rows):
            text = table.column("text")[i].as_py()
            url = table.column("url")[i].as_py() if "url" in table.column_names else ""
            doc_id = table.column("doc_id")[i].as_py()

            if passes_heuristics(text, lang, cfg):
                passing.append({
                    "text": text,
                    "url": url,
                    "doc_id": doc_id,
                })
        del table
    except Exception as e:
        log.warning("Error processing %s: %s", shard_path, e)

    return passing


def heuristic_filter_language(
    decontaminated_dir: str,
    lang: str,
    cfg: HeuristicConfig,
    num_workers: int = 24,
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply heuristic filters to all shards for one language.

    Returns (passing_docs, total_docs_seen).
    """
    lang_dir = Path(decontaminated_dir) / lang
    shards = sorted(lang_dir.glob("*.parquet"))
    if not shards:
        return [], 0

    cfg_json = json.dumps({
        "stopword_density_min": cfg.stopword_density_min,
        "stopword_density_max": cfg.stopword_density_max,
        "max_fk_grade": cfg.max_fk_grade,
        "min_fk_grade": cfg.min_fk_grade,
        "min_script_consistency": cfg.min_script_consistency,
        "min_unique_5gram_ratio": cfg.min_unique_5gram_ratio,
        "min_avg_sentence_len": cfg.min_avg_sentence_len,
        "max_avg_sentence_len": cfg.max_avg_sentence_len,
        "min_char_word_ratio": cfg.min_char_word_ratio,
        "max_char_word_ratio": cfg.max_char_word_ratio,
    })

    worker_args = [(str(s), lang, cfg_json) for s in shards]

    # Count total docs
    total_docs = 0
    for s in shards:
        try:
            meta = pq.read_metadata(s)
            total_docs += meta.num_rows
        except Exception:
            pass

    log.info("  Heuristic filtering %d shards (%d docs) with %d workers...",
             len(shards), total_docs, num_workers)

    all_passing = []
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(_filter_worker, worker_args):
            all_passing.extend(result)

    log.info("  Heuristic filter: %d/%d passed (%.1f%% rejected)",
             len(all_passing), total_docs,
             100 * (1 - len(all_passing) / max(1, total_docs)))

    return all_passing, total_docs


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2: Student Model Scoring + Topic Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def score_and_cluster(
    docs: List[Dict[str, Any]],
    scorer,  # StudentScorer
    n_clusters: int = 200,
    embedding_batch_size: int = 256,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Score documents and cluster into topics.

    Args:
        docs: List of {text, url, doc_id} dicts.
        scorer: StudentScorer instance.
        n_clusters: Number of topic clusters.
        embedding_batch_size: Batch size for embedding.

    Returns:
        scored_docs: docs with quality/difficulty/engagement/topic_id added
        embeddings: numpy array of shape (n_docs, dim)
        centroids: cluster centroid matrix (n_clusters, dim)
    """
    if not docs:
        return [], np.array([]), np.array([])

    texts = [d["text"] for d in docs]

    # Score in batches
    log.info("  Scoring %d documents with student model...", len(docs))
    all_scores = []
    for i in range(0, len(texts), embedding_batch_size):
        batch = texts[i:i + embedding_batch_size]
        scores = scorer.score_batch(batch)
        all_scores.extend(scores)

        if (i + embedding_batch_size) % (embedding_batch_size * 100) == 0:
            log.info("    Scored %d/%d", min(i + embedding_batch_size, len(texts)), len(texts))

    # Compute embeddings for clustering (reuse scorer's encoder)
    log.info("  Computing embeddings for topic clustering...")
    scorer._ensure_encoder()
    truncated = [t[:512] for t in texts]
    embeddings = scorer._encoder.encode(
        truncated,
        batch_size=embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # K-means clustering
    log.info("  Running k-means clustering (k=%d)...", n_clusters)
    actual_k = min(n_clusters, len(docs))

    try:
        import faiss
        kmeans = faiss.Kmeans(
            embeddings.shape[1], actual_k,
            niter=20, gpu=True, verbose=False,
        )
        kmeans.train(embeddings.astype(np.float32))
        _, assignments = kmeans.index.search(embeddings.astype(np.float32), 1)
        topic_ids = assignments.flatten()
        centroids = kmeans.centroids.copy()
    except (ImportError, RuntimeError):
        log.warning("  faiss-gpu not available, falling back to sklearn KMeans")
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=actual_k, random_state=42, batch_size=4096)
        topic_ids = km.fit_predict(embeddings)
        centroids = km.cluster_centers_

    # Merge scores into docs
    scored_docs = []
    for i, (doc, score) in enumerate(zip(docs, all_scores)):
        doc_scored = {
            "text": doc["text"],
            "url": doc["url"],
            "doc_id": doc["doc_id"],
            "quality": score.quality,
            "difficulty": score.difficulty,
            "topic_id": int(topic_ids[i]),
            "engagement": 1.0 if score.engagement else 0.0,
        }
        scored_docs.append(doc_scored)

    return scored_docs, embeddings, centroids


# ═══════════════════════════════════════════════════════════════════════════════
# Indexed Shard Writing
# ═══════════════════════════════════════════════════════════════════════════════

def write_indexed_shards(
    scored_docs: List[Dict[str, Any]],
    lang: str,
    output_dir: str,
    shard_size: int = 10_000,
) -> int:
    """Write scored documents as Hive-partitioned indexed shards.

    Output structure:
      output_dir/indexed/lang={lang}/topic={id}/shard_{idx:05d}.parquet

    Returns number of shards written.
    """
    # Group by topic
    by_topic: Dict[int, List[Dict]] = defaultdict(list)
    for doc in scored_docs:
        by_topic[doc["topic_id"]].append(doc)

    n_shards = 0
    for topic_id, topic_docs in sorted(by_topic.items()):
        topic_dir = Path(output_dir) / "indexed" / f"lang={lang}" / f"topic={topic_id}"
        topic_dir.mkdir(parents=True, exist_ok=True)

        for shard_start in range(0, len(topic_docs), shard_size):
            shard_docs = topic_docs[shard_start:shard_start + shard_size]
            shard_idx = shard_start // shard_size

            rows = {
                "text": [d["text"] for d in shard_docs],
                "url": [d["url"] for d in shard_docs],
                "doc_id": [d["doc_id"] for d in shard_docs],
                "quality": [d["quality"] for d in shard_docs],
                "difficulty": [d["difficulty"] for d in shard_docs],
                "topic_id": [d["topic_id"] for d in shard_docs],
                "engagement": [d["engagement"] for d in shard_docs],
            }

            table = pa.table(rows, schema=INDEXED_SCHEMA)
            shard_path = topic_dir / f"shard_{shard_idx:05d}.parquet"
            pq.write_table(table, shard_path, compression="snappy")
            n_shards += 1

    return n_shards


# ═══════════════════════════════════════════════════════════════════════════════
# Topic Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def compute_topic_distribution(
    scored_docs: List[Dict[str, Any]],
    lang: str,
) -> Dict[str, Any]:
    """Compute topic distribution statistics for one language.

    Returns dict with per-topic: count, avg_quality, difficulty_histogram.
    """
    by_topic: Dict[int, List[Dict]] = defaultdict(list)
    for doc in scored_docs:
        by_topic[doc["topic_id"]].append(doc)

    distribution = {}
    for topic_id, docs in sorted(by_topic.items()):
        qualities = [d["quality"] for d in docs]
        difficulties = [d["difficulty"] for d in docs]

        distribution[str(topic_id)] = {
            "count": len(docs),
            "avg_quality": round(float(np.mean(qualities)), 3),
            "difficulty_histogram": {
                "1": sum(1 for d in difficulties if d == 1),
                "2": sum(1 for d in difficulties if d == 2),
                "3": sum(1 for d in difficulties if d == 3),
            },
        }

    return {"lang": lang, "topics": distribution}


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndexConfig:
    """Configuration for scoring and indexing."""
    n_clusters: int = 200
    shard_size: int = 10_000
    num_workers: int = 24
    embedding_batch_size: int = 256
    upload_to_hf: bool = True
    hf_user: str = "Beetle-Data"
    hf_dataset_suffix: str = "indexed-24B"
    heuristic_cfg: HeuristicConfig = field(default_factory=HeuristicConfig)


def score_and_index_language(
    lang: str,
    output_dir: str,
    scorer,  # StudentScorer
    cfg: IndexConfig,
) -> IndexStats:
    """Run the full scoring + indexing pipeline for one language.

    Steps:
      1. Heuristic filter decontaminated Parquet
      2. Score with student model + cluster topics
      3. Write indexed shards
      4. Compute topic distribution

    Args:
        lang: Language code.
        output_dir: Pipeline output directory.
        scorer: Trained StudentScorer.
        cfg: Index configuration.

    Returns:
        IndexStats for this language.
    """
    stats = IndexStats(lang=lang)
    t0 = time.time()
    decontaminated_dir = str(Path(output_dir) / "decontaminated")

    # Pass 1: Heuristic filter
    log.info("=" * 50)
    log.info("Stage D: %s — Pass 1: Heuristic filtering", lang)
    passing_docs, total_docs = heuristic_filter_language(
        decontaminated_dir, lang, cfg.heuristic_cfg, cfg.num_workers,
    )
    stats.total_docs = total_docs
    stats.docs_after_heuristic = len(passing_docs)
    stats.heuristic_reject_rate = 1.0 - len(passing_docs) / max(1, total_docs)

    if not passing_docs:
        log.warning("  No documents passed heuristic filter for %s", lang)
        stats.wall_time_seconds = time.time() - t0
        return stats

    # Pass 2: Student model scoring + topic clustering
    log.info("Stage D: %s — Pass 2: Scoring + clustering", lang)
    scored_docs, embeddings, centroids = score_and_cluster(
        passing_docs, scorer, cfg.n_clusters, cfg.embedding_batch_size,
    )
    stats.docs_indexed = len(scored_docs)
    stats.n_clusters = cfg.n_clusters
    if scored_docs:
        stats.avg_quality = float(np.mean([d["quality"] for d in scored_docs]))

    # Write indexed shards
    log.info("Stage D: %s — Writing indexed shards", lang)
    stats.n_shards = write_indexed_shards(
        scored_docs, lang, output_dir, cfg.shard_size,
    )

    # Save centroids (append to existing or create new)
    centroids_path = Path(output_dir) / "indexed" / "cluster_centroids.pkl"
    centroids_data = {}
    if centroids_path.exists():
        with open(centroids_path, "rb") as f:
            centroids_data = pickle.load(f)
    centroids_data[lang] = centroids
    centroids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(centroids_path, "wb") as f:
        pickle.dump(centroids_data, f)

    # Compute and save topic distribution
    topic_dist = compute_topic_distribution(scored_docs, lang)
    dist_path = Path(output_dir) / "indexed" / "topic_distribution.json"
    all_dist = {}
    if dist_path.exists():
        with open(dist_path, "r") as f:
            all_dist = json.load(f)
    all_dist[lang] = topic_dist
    with open(dist_path, "w") as f:
        json.dump(all_dist, f, indent=2)

    # Update manifest
    manifest_path = Path(output_dir) / "indexed" / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    lang_manifest = {}
    indexed_dir = Path(output_dir) / "indexed" / f"lang={lang}"
    if indexed_dir.exists():
        for topic_dir in sorted(indexed_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            topic_id = topic_dir.name.replace("topic=", "")
            shards_info = []
            for shard in sorted(topic_dir.glob("*.parquet")):
                try:
                    meta = pq.read_metadata(shard)
                    shards_info.append({
                        "path": str(shard.relative_to(Path(output_dir) / "indexed")),
                        "n_docs": meta.num_rows,
                    })
                except Exception:
                    pass
            lang_manifest[topic_id] = shards_info
    manifest[lang] = lang_manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    stats.wall_time_seconds = time.time() - t0
    log.info("Stage D: %s complete — %d docs indexed in %d shards (%.1f hrs)",
             lang, stats.docs_indexed, stats.n_shards,
             stats.wall_time_seconds / 3600)

    # Save per-language stats
    stats_path = Path(output_dir) / "indexed" / f"{lang}_index_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    return stats


def score_and_index_all(
    langs: List[str],
    output_dir: str,
    cfg: IndexConfig,
) -> Dict[str, IndexStats]:
    """Score and index multiple languages.

    Loads the student model once and processes languages sequentially.
    """
    from .student_model import StudentScorer

    model_path = str(Path(output_dir) / "student_model" / "model.pkl")
    if not Path(model_path).exists():
        log.error("Student model not found at %s. Run Stage C first.", model_path)
        return {}

    scorer = StudentScorer.load(model_path)

    all_stats = {}
    for i, lang in enumerate(langs, 1):
        log.info("=" * 60)
        log.info("Language %d/%d: %s", i, len(langs), lang)
        log.info("=" * 60)
        stats = score_and_index_language(lang, output_dir, scorer, cfg)
        all_stats[lang] = stats

    # Write combined stats
    summary_path = Path(output_dir) / "indexed" / "index_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {lang: s.to_dict() for lang, s in all_stats.items()},
            f, indent=2,
        )

    total_indexed = sum(s.docs_indexed for s in all_stats.values())
    log.info("Indexing complete: %d total docs across %d languages",
             total_indexed, len(all_stats))

    return all_stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    from .config import LANG_REGISTRY, langs_for_node

    parser = argparse.ArgumentParser(description="Stage D: Score and Index")
    parser.add_argument("--output-dir", type=str, default="pipeline_output")
    parser.add_argument("--lang", type=str, nargs="+", default=None)
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()

    if args.lang:
        langs = args.lang
    elif args.node_id is not None:
        langs = langs_for_node(args.node_id)
    else:
        decon_path = Path(args.output_dir) / "decontaminated"
        if decon_path.exists():
            langs = sorted([d.name for d in decon_path.iterdir()
                          if d.is_dir() and list(d.glob("*.parquet"))])
        else:
            log.error("No decontaminated data found")
            return

    cfg = IndexConfig(
        n_clusters=args.n_clusters,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
    )

    score_and_index_all(langs, args.output_dir, cfg)


if __name__ == "__main__":
    main()
