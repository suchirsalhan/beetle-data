"""
student_model.py — Stage C: Train cheap embedding-based quality approximator.

Trains sklearn regressors/classifiers on frozen multilingual embeddings to
approximate the teacher LLM's pedagogical scores. The trained model is used
in Stage D to score the full corpus without requiring the LLM.

Architecture:
  - Embedding: intfloat/multilingual-e5-base (768-dim) via sentence-transformers
  - Quality head:     Ridge regression (0-5 continuous)
  - Difficulty head:  LogisticRegression (1-3 classification)
  - Engagement head:  LogisticRegression (binary)
  - Vocab head:       LogisticRegression (binary)

Training: sklearn on frozen embeddings. Minutes on CPU after GPU embedding.

Usage:
    python -m pipeline.run_pipeline --stage C --output-dir pipeline_output \\
        --beetlestream-config configs/beetlestream_curriculum.yaml

    # Direct invocation:
    python -m pipeline.student_model --output-dir pipeline_output
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("StudentModel")


# ═══════════════════════════════════════════════════════════════════════════════
# DocScore output type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocScore:
    """Prediction output for a single document."""
    quality: float       # 0-5 (continuous)
    difficulty: int      # 1-3
    engagement: bool
    vocab_complexity: bool


# ═══════════════════════════════════════════════════════════════════════════════
# Student Model Training
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudentConfig:
    """Configuration for student model training."""
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    batch_size: int = 256
    val_fraction: float = 0.1
    seed: int = 42


@dataclass
class TrainingStats:
    """Validation metrics for the trained student model."""
    n_train: int = 0
    n_val: int = 0
    quality_mae: float = 0.0
    quality_r2: float = 0.0
    difficulty_accuracy: float = 0.0
    engagement_accuracy: float = 0.0
    vocab_accuracy: float = 0.0
    embedding_time_seconds: float = 0.0
    training_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "quality_mae": round(self.quality_mae, 4),
            "quality_r2": round(self.quality_r2, 4),
            "difficulty_accuracy": round(self.difficulty_accuracy, 4),
            "engagement_accuracy": round(self.engagement_accuracy, 4),
            "vocab_accuracy": round(self.vocab_accuracy, 4),
            "embedding_time_seconds": round(self.embedding_time_seconds, 1),
            "training_time_seconds": round(self.training_time_seconds, 1),
            "total_time_seconds": round(self.total_time_seconds, 1),
        }


def load_feature_data(features_dir: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load feature Parquet files from all languages.

    Returns:
        texts: list of document texts
        quality: array of quality scores (0-5)
        difficulty: array of difficulty levels (1-3)
        engagement: array of engagement flags (0/1)
        vocab_complexity: array of vocab complexity flags (0/1)
    """
    texts = []
    quality = []
    difficulty = []
    engagement = []
    vocab_complexity = []

    features_path = Path(features_dir)
    for lang_dir in sorted(features_path.iterdir()):
        if not lang_dir.is_dir():
            continue
        fp = lang_dir / "features.parquet"
        if not fp.exists():
            continue

        table = pq.read_table(fp)
        for i in range(table.num_rows):
            text = table.column("text")[i].as_py()
            if not text:
                continue
            texts.append(text)
            quality.append(table.column("quality_score")[i].as_py())
            difficulty.append(table.column("difficulty_level")[i].as_py())
            engagement.append(int(table.column("engagement")[i].as_py()))
            vocab_complexity.append(int(table.column("vocab_complexity")[i].as_py()))
        del table

    return (
        texts,
        np.array(quality, dtype=np.float32),
        np.array(difficulty, dtype=np.int32),
        np.array(engagement, dtype=np.int32),
        np.array(vocab_complexity, dtype=np.int32),
    )


def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute sentence embeddings for all texts using sentence-transformers.

    Returns numpy array of shape (n_texts, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    log.info("Computing embeddings for %d texts (batch_size=%d)...",
             len(texts), batch_size)

    # Truncate texts to reasonable length for embedding
    truncated = [t[:512] for t in texts]

    embeddings = model.encode(
        truncated,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


def train_student_model(
    output_dir: str,
    cfg: Optional[StudentConfig] = None,
) -> TrainingStats:
    """Train the student model on feature data.

    Steps:
      1. Load features from all languages
      2. Compute embeddings
      3. Train sklearn models (quality Ridge, difficulty/engagement/vocab LogReg)
      4. Evaluate on validation split
      5. Save model + metrics

    Args:
        output_dir: Pipeline output directory.
        cfg: Student model configuration.

    Returns:
        TrainingStats with validation metrics.
    """
    if cfg is None:
        cfg = StudentConfig()

    stats = TrainingStats()
    t0 = time.time()

    # 1. Load features
    features_dir = str(Path(output_dir) / "features")
    texts, quality, difficulty, engagement, vocab = load_feature_data(features_dir)
    log.info("Loaded %d documents with features", len(texts))

    if len(texts) < 100:
        log.error("Too few documents (%d) for student model training", len(texts))
        return stats

    # 2. Compute embeddings
    t_embed = time.time()
    embeddings = compute_embeddings(texts, cfg.embedding_model, cfg.batch_size)
    stats.embedding_time_seconds = time.time() - t_embed

    # 3. Train/val split
    from sklearn.model_selection import train_test_split

    n = len(texts)
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.val_fraction, random_state=cfg.seed,
    )

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    stats.n_train = len(train_idx)
    stats.n_val = len(val_idx)

    log.info("Train: %d, Val: %d", stats.n_train, stats.n_val)

    # 4. Train models
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

    t_train = time.time()

    # Quality head (regression, 0-5)
    quality_model = Ridge(alpha=1.0)
    quality_model.fit(X_train, quality[train_idx])
    quality_pred = quality_model.predict(X_val)
    stats.quality_mae = float(mean_absolute_error(quality[val_idx], quality_pred))
    stats.quality_r2 = float(r2_score(quality[val_idx], quality_pred))

    # Difficulty head (classification, 1-3)
    difficulty_model = LogisticRegression(
        max_iter=1000, multi_class="multinomial", random_state=cfg.seed,
    )
    difficulty_model.fit(X_train, difficulty[train_idx])
    difficulty_pred = difficulty_model.predict(X_val)
    stats.difficulty_accuracy = float(accuracy_score(
        difficulty[val_idx], difficulty_pred,
    ))

    # Engagement head (binary)
    engagement_model = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    engagement_model.fit(X_train, engagement[train_idx])
    engagement_pred = engagement_model.predict(X_val)
    stats.engagement_accuracy = float(accuracy_score(
        engagement[val_idx], engagement_pred,
    ))

    # Vocab complexity head (binary)
    vocab_model = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    vocab_model.fit(X_train, vocab[train_idx])
    vocab_pred = vocab_model.predict(X_val)
    stats.vocab_accuracy = float(accuracy_score(
        vocab[val_idx], vocab_pred,
    ))

    stats.training_time_seconds = time.time() - t_train

    log.info("Quality MAE: %.3f, R²: %.3f", stats.quality_mae, stats.quality_r2)
    log.info("Difficulty accuracy: %.3f", stats.difficulty_accuracy)
    log.info("Engagement accuracy: %.3f", stats.engagement_accuracy)
    log.info("Vocab accuracy: %.3f", stats.vocab_accuracy)

    # 5. Save model
    model_dir = Path(output_dir) / "student_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "embedding_model": cfg.embedding_model,
        "embedding_dim": cfg.embedding_dim,
        "quality_model": quality_model,
        "difficulty_model": difficulty_model,
        "engagement_model": engagement_model,
        "vocab_model": vocab_model,
    }

    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    log.info("Saved student model to %s", model_path)

    # Save metrics
    stats.total_time_seconds = time.time() - t0
    metrics_path = model_dir / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info("Student model training complete in %.1f min",
             stats.total_time_seconds / 60)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Student Scorer (Inference)
# ═══════════════════════════════════════════════════════════════════════════════

class StudentScorer:
    """Load a trained student model and score new documents.

    Usage:
        scorer = StudentScorer.load("pipeline_output/student_model/model.pkl")
        scores = scorer.score_batch(["text1", "text2", ...])
    """

    def __init__(self, model_bundle: dict):
        self._bundle = model_bundle
        self._encoder = None  # lazy loaded

    @classmethod
    def load(cls, model_path: str) -> "StudentScorer":
        """Load a trained student model from disk."""
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        log.info("Loaded student model from %s (embedding: %s)",
                 model_path, bundle["embedding_model"])
        return cls(bundle)

    def _ensure_encoder(self):
        """Lazy-load the sentence-transformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                self._bundle["embedding_model"]
            )

    def score_batch(self, texts: List[str]) -> List[DocScore]:
        """Score a batch of documents.

        Args:
            texts: List of document texts.

        Returns:
            List of DocScore predictions.
        """
        self._ensure_encoder()

        # Truncate and embed
        truncated = [t[:512] for t in texts]
        embeddings = self._encoder.encode(
            truncated,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Predict with each head
        quality_pred = self._bundle["quality_model"].predict(embeddings)
        difficulty_pred = self._bundle["difficulty_model"].predict(embeddings)
        engagement_pred = self._bundle["engagement_model"].predict(embeddings)
        vocab_pred = self._bundle["vocab_model"].predict(embeddings)

        # Clamp quality to 0-5 range
        quality_pred = np.clip(quality_pred, 0.0, 5.0)

        scores = []
        for i in range(len(texts)):
            scores.append(DocScore(
                quality=float(quality_pred[i]),
                difficulty=int(difficulty_pred[i]),
                engagement=bool(engagement_pred[i]),
                vocab_complexity=bool(vocab_pred[i]),
            ))

        return scores

    @property
    def embedding_model_name(self) -> str:
        return self._bundle["embedding_model"]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage C: Student Model Training")
    parser.add_argument("--output-dir", type=str, default="pipeline_output")
    parser.add_argument("--embedding-model", type=str,
                        default="intfloat/multilingual-e5-base")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = StudentConfig(
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_student_model(args.output_dir, cfg)


if __name__ == "__main__":
    main()
