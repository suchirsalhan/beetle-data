"""
EduWeb Pipeline — Multi-dimensional Educational Corpus Builder
==============================================================
Implements: quality × safety × education × learner-level × contamination-free

Pipeline stages:
  1.  Basic Cleaning + Dedup
  2.  Language ID + Code-mix Filtering
  3.  Safety Filtering (toxicity + PII)
  4.  Decontamination (benchmark n-gram overlap)
  5.  Educational Quality Scoring (LLM → regressor)
  6.  Learner-Level Classification (age/grade)
  7.  Source + Genre Filtering
  8.  Infinigram-style Indexing
  9.  Stratified Sampling → final corpora

Install requirements:
    pip install datasets transformers sentence-transformers \
                scikit-learn langdetect ftfy tqdm numpy pandas \
                peft accelerate torch
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("EduPipeline")


# ===========================================================================
# 0. Config
# ===========================================================================

@dataclass
class PipelineConfig:
    # ── paths ──────────────────────────────────────────────────────────────
    input_jsonl:          str = "raw_data.jsonl"
    output_dir:           str = "outputs"
    benchmark_dir:        str = "benchmarks"     # folder with .txt benchmark files
    trusted_sources_file: str = "trusted_sources.txt"
    scorer_model_path:    str = "edu_scorer.pkl"  # persisted sklearn model

    # ── language filtering ─────────────────────────────────────────────────
    target_language:      str  = "en"             # ISO 639-1 code
    langid_threshold:     float = 0.80
    langid_strict:        float = 0.90

    # ── dedup ──────────────────────────────────────────────────────────────
    minhash_threshold:    float = 0.85
    minhash_num_perm:     int   = 128

    # ── safety ─────────────────────────────────────────────────────────────
    toxicity_hard_threshold:  float = 0.70
    toxicity_soft_threshold:  float = 0.40

    # ── decontamination ────────────────────────────────────────────────────
    ngram_size:           int   = 13

    # ── educational scoring ────────────────────────────────────────────────
    edu_tier_strict:      float = 4.0
    edu_tier_mid:         float = 3.0
    edu_tier_loose:       float = 2.0
    annotation_sample_size: int = 2000   # docs to LLM-annotate for regressor training

    # ── learner level ──────────────────────────────────────────────────────
    # one of: K1, 2-3, 4-5, 6-8, 9-12, Adult
    target_audience:      Optional[str] = None    # None = keep all

    # ── sampling ───────────────────────────────────────────────────────────
    final_corpus_size:    int = 10_000             # docs (set higher for real runs)
    sampling_weights: Dict[str, float] = field(default_factory=lambda: {
        "edu_high":   0.40,
        "edu_mid":    0.30,
        "general":    0.20,
        "kids":       0.10,
    })


CFG = PipelineConfig()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 1. Data Model
# ===========================================================================

@dataclass
class Document:
    id:          str
    text:        str
    url:         str  = ""
    source:      str  = ""
    lang:        str  = ""
    lang_score:  float = 0.0

    # scores populated during pipeline
    toxicity:    float = 0.0
    edu_score:   float = 0.0
    learner_level: str = "unknown"
    audience:    str  = "unknown"   # kids / teens / adults

    # flags
    is_duplicate:        bool = False
    is_contaminated:     bool = False
    is_safe:             bool = True
    passes_lang_filter:  bool = False
    trusted_source:      bool = False

    # tier assigned at end
    tier: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ===========================================================================
# 2. STAGE 1 — Basic Cleaning + Exact Deduplication
# ===========================================================================

class BasicCleaner:
    """Unicode normalization, whitespace normalization, minimal length filter."""

    MIN_CHARS = 200
    MAX_CHARS = 100_000

    @staticmethod
    def clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = ftfy_fix(text)
        # collapse runs of whitespace / zero-width chars
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def passes(self, text: str) -> bool:
        return self.MIN_CHARS <= len(text) <= self.MAX_CHARS


def ftfy_fix(text: str) -> str:
    """Use ftfy if available, else identity."""
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        return text


class ExactDeduplicator:
    """SHA-256 based exact deduplication."""

    def __init__(self):
        self._seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False


# ===========================================================================
# 3. STAGE 2 — Language Identification + Code-mix Filtering
# ===========================================================================

class LangIDFilter:
    """
    Ensemble: langdetect (lid.176.bin style) + langid library.
    Falls back gracefully if either is missing.
    """

    def __init__(self, target_lang: str, threshold: float, strict: float):
        self.target  = target_lang
        self.thresh  = threshold
        self.strict  = strict
        self._init_detectors()

    def _init_detectors(self):
        self._has_langdetect = False
        self._has_langid     = False
        try:
            from langdetect import detect_langs
            self._detect_langs = detect_langs
            self._has_langdetect = True
            log.info("langdetect loaded ✓")
        except ImportError:
            log.warning("langdetect not installed — using fallback only")

        try:
            import langid
            langid.set_languages([self.target, "fr", "ar", "es", "de", "zh"])
            self._langid = langid
            self._has_langid = True
            log.info("langid loaded ✓")
        except ImportError:
            log.warning("langid not installed")

    def detect(self, text: str) -> Tuple[str, float]:
        """Return (lang, confidence) using ensemble."""
        results = {}

        if self._has_langdetect:
            try:
                probs = {r.lang: r.prob for r in self._detect_langs(text[:1000])}
                results["langdetect"] = probs.get(self.target, 0.0)
            except Exception:
                pass

        if self._has_langid:
            try:
                lang, conf_raw = self._langid.classify(text[:1000])
                # langid returns log-prob; map to ~[0,1] via sigmoid
                conf = 1 / (1 + np.exp(-abs(conf_raw) / 10))
                results["langid"] = conf if lang == self.target else (1 - conf)
            except Exception:
                pass

        if not results:
            # no detector — pass everything through
            return self.target, 1.0

        avg = np.mean(list(results.values()))
        agreement = all(v >= 0.5 for v in results.values())
        return self.target, float(avg) if agreement else float(avg * 0.7)

    def passes(self, text: str) -> Tuple[bool, float]:
        lang, score = self.detect(text)
        ok = score >= self.strict or (score >= self.thresh and lang == self.target)
        return ok, score

    def has_codemix(self, text: str) -> bool:
        """Detect high degree of code-switching via sentence-level sampling."""
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 40]
        if not sentences:
            return False
        foreign = 0
        for s in sentences[:20]:
            _, sc = self.detect(s)
            if sc < 0.7:
                foreign += 1
        return foreign / max(len(sentences[:20]), 1) > 0.30


# ===========================================================================
# 4. STAGE 3 — Safety Filtering (Toxicity + PII)
# ===========================================================================

class ToxicityFilter:
    """
    Rule-based lexicon toxicity scorer (production: replace with a classifier
    like toxic-bert or Detoxify).
    """

    _TOXIC_PATTERNS = [
        r"\b(fuck|shit|ass|bitch|cunt|bastard|damn|hell)\b",
        r"\b(kill\s+your?self|kys|suicide\s+method)\b",
        r"\b(n[i1]gg[ae]r|f[a4]gg[o0]t|ch[i1]nk|sp[i1]c)\b",
        r"\b(rape|molest|pedophil)\b",
        r"\b(isis|al.qaeda|terrorism\s+guide)\b",
    ]
    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]

    def score(self, text: str) -> float:
        hits = sum(1 for p in self._COMPILED if p.search(text))
        return min(hits / 3.0, 1.0)          # normalise to [0,1]

    def passes(self, text: str, hard: float, soft: float) -> Tuple[bool, float]:
        sc = self.score(text)
        if sc >= hard:
            return False, sc
        # soft: still keep, just flagged
        return True, sc


class PIIRemover:
    """Regex-based PII removal."""

    _PATTERNS = {
        "email":   re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        "phone":   re.compile(r"\b(\+?\d[\d\s\-().]{7,}\d)\b"),
        "ssn":     re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "url_pii": re.compile(r"https?://\S+"),   # optionally strip URLs
    }

    def remove(self, text: str) -> str:
        text = self._PATTERNS["email"].sub("[EMAIL]", text)
        text = self._PATTERNS["phone"].sub("[PHONE]", text)
        text = self._PATTERNS["ssn"].sub("[SSN]", text)
        return text


# ===========================================================================
# 5. STAGE 4 — Benchmark Decontamination (13-gram overlap)
# ===========================================================================

class Decontaminator:
    """
    Goldfish-style 13-gram decontamination.
    Loads .txt benchmark files from benchmark_dir.
    """

    def __init__(self, benchmark_dir: str, ngram_size: int = 13):
        self.n = ngram_size
        self._index: set[tuple] = set()
        self._load(benchmark_dir)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _build_ngrams(self, tokens: List[str]) -> set[tuple]:
        return {tuple(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)}

    def _load(self, d: str):
        p = Path(d)
        if not p.exists():
            log.warning(f"Benchmark dir '{d}' not found — skipping decontamination")
            return
        for f in p.glob("*.txt"):
            tokens = self._tokenize(f.read_text(errors="ignore"))
            self._index |= self._build_ngrams(tokens)
        log.info(f"Decontaminator: {len(self._index):,} benchmark n-grams indexed")

    def is_contaminated(self, text: str) -> bool:
        if not self._index:
            return False
        tokens = self._tokenize(text)
        ngrams = self._build_ngrams(tokens)
        return bool(ngrams & self._index)


# ===========================================================================
# 6. STAGE 5 — Educational Quality Scoring
# ===========================================================================

class EducationalScorer:
    """
    Two-phase scorer:
      Phase A  — rule/heuristic scoring (fast, used when no trained model)
      Phase B  — LLM annotation + sklearn regressor (optional, better quality)

    The regressor is trained once on LLM-annotated samples and persisted.
    """

    # ── heuristics ────────────────────────────────────────────────────────

    _EDU_POSITIVE = re.compile(
        r"\b(learn|study|explain|definition|example|concept|exercise|"
        r"understand|knowledge|teach|lesson|chapter|textbook|homework|"
        r"science|math|history|biology|chemistry|physics|literature|"
        r"grammar|vocabulary|geography|theorem|formula|experiment)\b",
        re.IGNORECASE,
    )
    _EDU_NEGATIVE = re.compile(
        r"\b(click here|subscribe|promo|advertisement|coupon|"
        r"buy now|limited offer|casino|gambling|adult|xxx)\b",
        re.IGNORECASE,
    )

    def __init__(self, model_path: Optional[str] = None):
        self._regressor = None
        self._embedder  = None
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    # ── public API ────────────────────────────────────────────────────────

    def score(self, text: str) -> float:
        """Returns score in [0, 5]."""
        if self._regressor is not None:
            return self._regressor_score(text)
        return self._heuristic_score(text)

    # ── heuristic fallback ────────────────────────────────────────────────

    def _heuristic_score(self, text: str) -> float:
        score = 0.0

        # positive keyword density
        words = text.split()
        if not words:
            return 0.0
        pos_hits = len(self._EDU_POSITIVE.findall(text))
        neg_hits = len(self._EDU_NEGATIVE.findall(text))
        score += min(pos_hits / max(len(words) / 50, 1), 2.5)
        score -= min(neg_hits / max(len(words) / 50, 1), 2.0)

        # readability proxy: avg sentence length 10-25 words is good
        sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]
        if sentences:
            avg_sent_len = np.mean([len(s.split()) for s in sentences])
            if 10 <= avg_sent_len <= 25:
                score += 1.0
            elif avg_sent_len < 5 or avg_sent_len > 50:
                score -= 0.5

        # structural signals
        if re.search(r"\n#+\s", text):        score += 0.5   # headers
        if re.search(r"\d+\.", text):         score += 0.3   # numbered lists
        if re.search(r"\bfigure\b|\btable\b", text, re.I): score += 0.3

        return float(np.clip(score, 0.0, 5.0))

    # ── ML-based scoring ─────────────────────────────────────────────────

    def _load_model(self, path: str):
        import pickle
        with open(path, "rb") as f:
            self._regressor = pickle.load(f)
        log.info(f"Edu scorer loaded from {path}")
        self._init_embedder()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Sentence embedder loaded ✓")
        except ImportError:
            log.warning("sentence-transformers not installed — using heuristic scorer")

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._embedder is None:
            raise RuntimeError("No embedder")
        return self._embedder.encode(texts, show_progress_bar=False)

    def _regressor_score(self, text: str) -> float:
        try:
            emb = self._embed([text[:512]])
            return float(np.clip(self._regressor.predict(emb)[0], 0.0, 5.0))
        except Exception:
            return self._heuristic_score(text)

    # ── training helpers ─────────────────────────────────────────────────

    @staticmethod
    def llm_annotate(texts: List[str], api_key: Optional[str] = None) -> List[float]:
        """
        Call an LLM to get educational scores. Requires OPENAI_API_KEY or
        ANTHROPIC_API_KEY env var. Falls back to heuristic if unavailable.
        """
        scorer = EducationalScorer()   # heuristic fallback
        scores = []
        for t in tqdm(texts, desc="LLM annotation"):
            scores.append(scorer._heuristic_score(t))   # swap with API call
        return scores

    def train_and_save(
        self,
        texts: List[str],
        scores: Optional[List[float]] = None,
        save_path: str = "edu_scorer.pkl",
    ):
        """Train regressor on (text, score) pairs and persist."""
        import pickle
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._init_embedder()
        if scores is None:
            scores = self.llm_annotate(texts)

        log.info(f"Training edu regressor on {len(texts)} samples…")
        X = self._embed(texts)
        y = np.array(scores)

        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X, y)
        self._regressor = model

        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Edu scorer saved to {save_path}")


# ===========================================================================
# 7. STAGE 6 — Learner-Level Classification
# ===========================================================================

@dataclass
class ReadabilityFeatures:
    avg_word_length:   float
    avg_sent_length:   float
    flesch_kincaid:    float
    type_token_ratio:  float
    long_word_ratio:   float   # words > 6 chars


class LearnerLevelClassifier:
    """
    Rule-based grade-level classifier (production: replace with a fine-tuned
    text classifier trained on grade-tagged corpora).

    Grade bands:
        K1     → Flesch-Kincaid grade ≤ 1,   avg word len ≤ 4
        2-3    → FK ≤ 3
        4-5    → FK ≤ 5
        6-8    → FK ≤ 8
        9-12   → FK ≤ 12
        Adult  → FK > 12
    """

    BANDS = ["K1", "2-3", "4-5", "6-8", "9-12", "Adult"]

    def featurize(self, text: str) -> ReadabilityFeatures:
        words = re.findall(r"\b\w+\b", text)
        sents = [s for s in re.split(r"[.!?]", text) if s.strip()]
        if not words or not sents:
            return ReadabilityFeatures(5, 15, 8, 0.5, 0.3)

        avg_wl = np.mean([len(w) for w in words])
        avg_sl = len(words) / len(sents)
        syllables = sum(self._count_syllables(w) for w in words)
        fk = 0.39 * avg_sl + 11.8 * (syllables / max(len(words), 1)) - 15.59
        ttr = len(set(w.lower() for w in words)) / len(words)
        lwr = sum(1 for w in words if len(w) > 6) / len(words)
        return ReadabilityFeatures(avg_wl, avg_sl, fk, ttr, lwr)

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def classify(self, text: str) -> Tuple[str, str]:
        """Returns (learner_level, audience)."""
        feat = self.featurize(text)
        fk   = feat.flesch_kincaid

        if fk <= 1  or feat.avg_word_length <= 3.5: level = "K1"
        elif fk <= 3:  level = "2-3"
        elif fk <= 5:  level = "4-5"
        elif fk <= 8:  level = "6-8"
        elif fk <= 12: level = "9-12"
        else:          level = "Adult"

        # coarse audience mapping
        audience_map = {
            "K1":    "kids",
            "2-3":   "kids",
            "4-5":   "kids",
            "6-8":   "teens",
            "9-12":  "teens",
            "Adult": "adults",
        }
        return level, audience_map[level]


# ===========================================================================
# 8. STAGE 7 — Source / Genre Filtering
# ===========================================================================

class SourceFilter:
    """Boosts or removes documents based on their URL / domain."""

    def __init__(self, trusted_file: str):
        self._trusted: set[str] = set()
        p = Path(trusted_file)
        if p.exists():
            self._trusted = {line.strip().lower() for line in p.read_text().splitlines() if line.strip()}
            log.info(f"Trusted sources: {len(self._trusted)} domains loaded")
        else:
            # built-in defaults
            self._trusted = {
                "wikipedia.org", "britannica.com", "khanacademy.org",
                "nationalgeographic.com", "bbc.co.uk/cbbc", "scholastic.com",
                "readworks.org", "newsela.com", "ducksters.com", "kidsnewsroom.org",
                "pbs.org", "smithsonianmag.com", "nature.com", "sciencedaily.com",
            }

    def is_trusted(self, url: str) -> bool:
        url = url.lower()
        return any(domain in url for domain in self._trusted)


# ===========================================================================
# 9. STAGE 8 — Infinigram-style N-gram Index
# ===========================================================================

class InfinigramIndex:
    """
    Lightweight n-gram frequency index for memorization / diversity analysis.
    (Full Infinigram uses suffix arrays; this is a dict-based approximation.)
    """

    def __init__(self, n: int = 5):
        self.n = n
        self._index: Counter = Counter()

    def add(self, text: str):
        tokens = re.findall(r"\b\w+\b", text.lower())
        for i in range(len(tokens) - self.n + 1):
            ng = " ".join(tokens[i:i+self.n])
            self._index[ng] += 1

    def memorization_score(self, text: str) -> float:
        """Fraction of n-grams that appear ≥ 3 times in the corpus."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        ngrams = [" ".join(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]
        if not ngrams:
            return 0.0
        repeated = sum(1 for ng in ngrams if self._index[ng] >= 3)
        return repeated / len(ngrams)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dict(self._index.most_common(500_000)), f)

    @classmethod
    def load(cls, path: str, n: int = 5) -> "InfinigramIndex":
        idx = cls(n)
        with open(path) as f:
            idx._index = Counter(json.load(f))
        return idx


# ===========================================================================
# 10. STAGE 9 — Stratified Sampler
# ===========================================================================

class StratifiedSampler:
    """
    Assigns a tier to each document, then samples according to target quotas.
    """

    TIER_ORDER = ["edu_high", "edu_mid", "general", "kids"]

    def assign_tier(self, doc: Document) -> str:
        if doc.learner_level in ("K1", "2-3", "4-5") and doc.is_safe:
            return "kids"
        if doc.edu_score >= CFG.edu_tier_strict:
            return "edu_high"
        if doc.edu_score >= CFG.edu_tier_mid:
            return "edu_mid"
        return "general"

    def sample(
        self,
        docs: List[Document],
        total: int,
        weights: Dict[str, float],
    ) -> List[Document]:
        buckets: Dict[str, List[Document]] = defaultdict(list)
        for d in docs:
            buckets[d.tier].append(d)

        results: List[Document] = []
        for tier, w in weights.items():
            n = int(total * w)
            pool = buckets.get(tier, [])
            # sort by edu_score descending within tier
            pool.sort(key=lambda x: x.edu_score, reverse=True)
            results.extend(pool[:n])
            log.info(f"  tier={tier:10s}  requested={n:5d}  available={len(pool):5d}  sampled={min(n,len(pool)):5d}")

        return results


# ===========================================================================
# 11. I/O Helpers
# ===========================================================================

def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning(f"Skipping bad JSON at line {i}")


def save_jsonl(docs: List[Document], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"Saved {len(docs):,} docs → {path}")


def save_stats(stats: dict, path: str):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats → {path}")


# ===========================================================================
# 12. MAIN PIPELINE ORCHESTRATOR
# ===========================================================================

class EduPipeline:

    def __init__(self, cfg: PipelineConfig = CFG):
        self.cfg = cfg
        log.info("Initialising pipeline components…")
        self.cleaner      = BasicCleaner()
        self.deduper      = ExactDeduplicator()
        self.lang_filter  = LangIDFilter(cfg.target_language, cfg.langid_threshold, cfg.langid_strict)
        self.tox_filter   = ToxicityFilter()
        self.pii_remover  = PIIRemover()
        self.decontam     = Decontaminator(cfg.benchmark_dir, cfg.ngram_size)
        self.edu_scorer   = EducationalScorer(cfg.scorer_model_path)
        self.level_clf    = LearnerLevelClassifier()
        self.src_filter   = SourceFilter(cfg.trusted_sources_file)
        self.ng_index     = InfinigramIndex(n=5)
        self.sampler      = StratifiedSampler()

        self.stats = {
            "total_input": 0,
            "after_clean": 0,
            "after_dedup": 0,
            "after_lang":  0,
            "after_safety":0,
            "after_decontam": 0,
            "final_sampled": 0,
            "tier_counts": {},
            "level_counts": {},
        }

    # ── core processing ───────────────────────────────────────────────────

    def process_document(self, raw: dict) -> Optional[Document]:
        """Process one raw dict through all filter stages. Returns None if rejected."""
        self.stats["total_input"] += 1

        # ── build Document ────────────────────────────────────────────────
        text = raw.get("text", raw.get("content", ""))
        doc  = Document(
            id     = raw.get("id", hashlib.md5(text[:200].encode()).hexdigest()),
            text   = text,
            url    = raw.get("url", ""),
            source = raw.get("source", ""),
        )

        # ── STAGE 1: clean ────────────────────────────────────────────────
        doc.text = self.cleaner.clean(doc.text)
        if not self.cleaner.passes(doc.text):
            return None
        self.stats["after_clean"] += 1

        # ── STAGE 1b: exact dedup ─────────────────────────────────────────
        if self.deduper.is_duplicate(doc.text):
            doc.is_duplicate = True
            return None
        self.stats["after_dedup"] += 1

        # ── STAGE 2: language ─────────────────────────────────────────────
        ok, score = self.lang_filter.passes(doc.text)
        doc.lang_score = score
        doc.lang       = self.cfg.target_language
        if not ok:
            doc.passes_lang_filter = False
            return None
        if self.lang_filter.has_codemix(doc.text):
            return None
        doc.passes_lang_filter = True
        self.stats["after_lang"] += 1

        # ── STAGE 3: safety ───────────────────────────────────────────────
        safe, tox_sc = self.tox_filter.passes(
            doc.text, self.cfg.toxicity_hard_threshold, self.cfg.toxicity_soft_threshold
        )
        doc.toxicity = tox_sc
        doc.is_safe  = safe
        if not safe:
            return None
        doc.text = self.pii_remover.remove(doc.text)
        self.stats["after_safety"] += 1

        # ── STAGE 4: decontamination ──────────────────────────────────────
        if self.decontam.is_contaminated(doc.text):
            doc.is_contaminated = True
            return None
        self.stats["after_decontam"] += 1

        # ── STAGE 5: educational score ────────────────────────────────────
        doc.edu_score = self.edu_scorer.score(doc.text)

        # ── STAGE 6: learner level ────────────────────────────────────────
        doc.learner_level, doc.audience = self.level_clf.classify(doc.text)

        # optional audience filter
        if self.cfg.target_audience and doc.learner_level != self.cfg.target_audience:
            return None

        # ── STAGE 7: source boosting ──────────────────────────────────────
        doc.trusted_source = self.src_filter.is_trusted(doc.url)
        if doc.trusted_source:
            doc.edu_score = min(doc.edu_score + 0.5, 5.0)

        # ── STAGE 8: n-gram index ─────────────────────────────────────────
        self.ng_index.add(doc.text)

        return doc

    # ── pipeline runner ───────────────────────────────────────────────────

    def run(self) -> Dict[str, List[Document]]:
        log.info("=" * 60)
        log.info("EduPipeline starting…")
        log.info("=" * 60)

        processed: List[Document] = []

        for raw in tqdm(load_jsonl(self.cfg.input_jsonl), desc="Processing docs"):
            doc = self.process_document(raw)
            if doc is not None:
                processed.append(doc)

        log.info(f"\n── Filter funnel ──────────────────────────────")
        log.info(f"  Input:             {self.stats['total_input']:>8,}")
        log.info(f"  After clean:       {self.stats['after_clean']:>8,}")
        log.info(f"  After dedup:       {self.stats['after_dedup']:>8,}")
        log.info(f"  After lang filter: {self.stats['after_lang']:>8,}")
        log.info(f"  After safety:      {self.stats['after_safety']:>8,}")
        log.info(f"  After decontam:    {self.stats['after_decontam']:>8,}")
        log.info(f"  Total kept:        {len(processed):>8,}")

        # ── STAGE 9: assign tiers + sample ───────────────────────────────
        for doc in processed:
            doc.tier = self.sampler.assign_tier(doc)

        tier_counts = Counter(d.tier for d in processed)
        level_counts = Counter(d.learner_level for d in processed)
        self.stats["tier_counts"]  = dict(tier_counts)
        self.stats["level_counts"] = dict(level_counts)
        log.info(f"\n── Tier distribution ─────────────────────────")
        for t, c in tier_counts.most_common():
            log.info(f"  {t:12s}: {c:,}")
        log.info(f"\n── Learner-level distribution ────────────────")
        for l, c in level_counts.most_common():
            log.info(f"  {l:8s}: {c:,}")

        log.info(f"\n── Stratified sampling (target={self.cfg.final_corpus_size:,}) ─")
        final = self.sampler.sample(processed, self.cfg.final_corpus_size, self.cfg.sampling_weights)
        self.stats["final_sampled"] = len(final)

        # ── save outputs ──────────────────────────────────────────────────
        out = Path(self.cfg.output_dir)

        # all processed (pre-sampling)
        save_jsonl(processed, str(out / "corpus_full.jsonl"))

        # final sampled corpus
        save_jsonl(final, str(out / "corpus_final.jsonl"))

        # per-tier sub-corpora
        corpora: Dict[str, List[Document]] = defaultdict(list)
        for d in final:
            corpora[d.tier].append(d)

        for tier, docs in corpora.items():
            save_jsonl(docs, str(out / f"corpus_{tier}.jsonl"))

        # n-gram index
        self.ng_index.save(str(out / "ngram_index.json"))

        # stats
        save_stats(self.stats, str(out / "pipeline_stats.json"))

        log.info("\n✅  Pipeline complete.")
        log.info(f"   Final corpus: {len(final):,} documents")
        log.info(f"   Outputs in:   {self.cfg.output_dir}/")
        return dict(corpora)


# ===========================================================================
# 13. OPTIONAL: Train educational scorer from scratch
# ===========================================================================

def train_edu_scorer(
    input_jsonl: str,
    sample_size: int = 2000,
    save_path: str = "edu_scorer.pkl",
):
    """
    Convenience function: sample docs → heuristic annotation → train + save regressor.
    Swap heuristic annotation with real LLM calls for better quality.
    """
    texts = []
    for i, raw in enumerate(load_jsonl(input_jsonl)):
        if i >= sample_size:
            break
        t = raw.get("text", raw.get("content", "")).strip()
        if len(t) > 200:
            texts.append(t[:2048])

    log.info(f"Collected {len(texts)} texts for scorer training")
    scorer = EducationalScorer()
    scores = scorer.llm_annotate(texts)
    scorer.train_and_save(texts, scores, save_path)
    log.info("Scorer training complete.")


# ===========================================================================
# 14. CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EduWeb Pipeline")
    parser.add_argument("--input",       default=CFG.input_jsonl,   help="Input JSONL file")
    parser.add_argument("--output-dir",  default=CFG.output_dir,    help="Output directory")
    parser.add_argument("--lang",        default=CFG.target_language)
    parser.add_argument("--target-size", default=CFG.final_corpus_size, type=int)
    parser.add_argument("--benchmark-dir", default=CFG.benchmark_dir)
    parser.add_argument("--train-scorer", action="store_true", help="Train edu scorer before running")
    parser.add_argument("--scorer-path",  default=CFG.scorer_model_path)
    args = parser.parse_args()

    CFG.input_jsonl       = args.input
    CFG.output_dir        = args.output_dir
    CFG.target_language   = args.lang
    CFG.final_corpus_size = args.target_size
    CFG.benchmark_dir     = args.benchmark_dir
    CFG.scorer_model_path = args.scorer_path

    if args.train_scorer:
        train_edu_scorer(CFG.input_jsonl, CFG.annotation_sample_size, CFG.scorer_model_path)

    pipeline = EduPipeline(CFG)
    pipeline.run()
