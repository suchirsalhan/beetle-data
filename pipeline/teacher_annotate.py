"""
teacher_annotate.py — Stage A: LLM-based pedagogical annotation of sampled documents.

Reservoir-samples documents from decontaminated Parquet shards across all languages,
annotates them with a pedagogical rubric using an open-source LLM (Llama-3-70B or 8B)
served via vLLM, and writes structured JSON annotations.

Supported teacher models (all open-source, from HuggingFace):
  - meta-llama/Meta-Llama-3-70B-Instruct  (default, highest quality)
  - meta-llama/Meta-Llama-3-8B-Instruct   (faster, lower quality)

Calibration examples are loaded from HuggingFace:
  - tafseer-nayeem/KidLM-corpus  (high-quality educational text anchors)
  - ADALM/CLC-L1-CEFR           (CEFR-leveled learner text anchors)

Usage:
    # With vLLM server already running on port 8000:
    python -m pipeline.run_pipeline --stage A --output-dir pipeline_output \\
        --beetlestream-config configs/beetlestream_curriculum.yaml

    # Direct invocation:
    python -m pipeline.teacher_annotate \\
        --output-dir pipeline_output \\
        --model meta-llama/Meta-Llama-3-70B-Instruct \\
        --vllm-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("TeacherAnnotate")


# ═══════════════════════════════════════════════════════════════════════════════
# Pedagogical Rubric Prompt
# ═══════════════════════════════════════════════════════════════════════════════

TEACHER_PROMPT = """You are an expert in education, linguistics, and language acquisition.
Your task is to evaluate a piece of text for its suitability for learners across different proficiency levels. You must assess readability, conceptual difficulty, vocabulary, topic, and pedagogical relevance.

IMPORTANT:
* Answer ALL questions.
* Be consistent and objective.
* Base your answers ONLY on the text provided.
* Do NOT explain your answers.
* Output ONLY valid JSON.

---
TEXT:
{input_text}
---

Answer the following questions with true or false:

[READABILITY & LEVEL]
1. Is this text readable for an elementary school student?
2. Is this text readable for a middle school student?
3. Is this text readable for a high school student?
4. Is this text easy to understand for its intended audience?
5. Could an average student at the appropriate level understand this text without assistance?
6. Could most students complete a task based on this text without significant difficulty?

[VOCABULARY & LANGUAGE]
7. Does this text use simple and common vocabulary?
8. Does this text contain complex or advanced vocabulary?
9. Does this text contain technical jargon?
10. Does this text include figurative language (e.g., metaphors, idioms)?

[PEDAGOGICAL RELEVANCE]
11. Is this text relevant to typical school curriculum topics?
12. Is this text appropriate for the knowledge and skills of students at its level?
13. Is this text aligned with educational or instructional content?

[ENGAGEMENT & EXPERIENCE]
14. Is this text relevant to everyday experiences of students?
15. Could an average student engage with this content?
16. Is the length appropriate for students at the intended level?

[TOPIC CLASSIFICATION]
17. Is this text about science?
18. Is this text about mathematics?
19. Is this text about social science (e.g., history, geography)?
20. Is this text about language or literature?

[OUTPUT FORMAT]
Return ONLY a JSON object with the following structure:
{
  "readability": {
    "elementary": true/false,
    "middle": true/false,
    "high": true/false
  },
  "comprehension": {
    "easy": true/false,
    "independent": true/false,
    "low_difficulty": true/false
  },
  "vocabulary": {
    "simple": true/false,
    "complex": true/false,
    "jargon": true/false,
    "figurative": true/false
  },
  "pedagogy": {
    "curriculum_relevant": true/false,
    "level_appropriate": true/false,
    "instructional": true/false
  },
  "engagement": {
    "experiential": true/false,
    "engaging": true/false,
    "length_appropriate": true/false
  },
  "topic": {
    "science": true/false,
    "math": true/false,
    "social_science": true/false,
    "language": true/false
  }
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnnotationStats:
    """Track annotation progress and quality."""
    total_sampled: int = 0
    total_annotated: int = 0
    total_failed: int = 0
    total_retried: int = 0
    per_lang_counts: Dict[str, int] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    model_name: str = ""

    def to_dict(self) -> dict:
        return {
            "total_sampled": self.total_sampled,
            "total_annotated": self.total_annotated,
            "total_failed": self.total_failed,
            "total_retried": self.total_retried,
            "per_lang_counts": self.per_lang_counts,
            "wall_time_seconds": self.wall_time_seconds,
            "wall_time_hours": self.wall_time_seconds / 3600,
            "model_name": self.model_name,
            "success_rate": (self.total_annotated /
                            max(1, self.total_sampled)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_kidlm_samples(repo: str, n_samples: int = 20) -> List[str]:
    """Load sample passages from KidLM corpus on HuggingFace.

    Args:
        repo: HuggingFace dataset repo (e.g., "tafseer-nayeem/KidLM-corpus")
        n_samples: Number of passages to sample.

    Returns:
        List of text passages.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, split="train", streaming=True)
        samples = []
        for i, entry in enumerate(ds):
            text = entry.get("text", "").strip()
            if text and len(text) > 100:
                samples.append(text[:500])  # truncate to ~500 chars
            if len(samples) >= n_samples * 3:
                break
        # Randomly select n_samples from collected
        if len(samples) > n_samples:
            samples = random.sample(samples, n_samples)
        log.info("Loaded %d KidLM calibration samples from %s", len(samples), repo)
        return samples
    except Exception as e:
        log.warning("Failed to load KidLM samples from %s: %s", repo, e)
        return []


def load_clc_cefr_samples(
    repo: str,
    samples_per_level: int = 3,
) -> Dict[str, List[str]]:
    """Load CEFR-leveled learner text samples from ADALM/CLC-L1-CEFR.

    The repo contains directories a1/ a2/ b1/ b2/ c1/ c2/ with text files.

    Args:
        repo: HuggingFace dataset repo (e.g., "ADALM/CLC-L1-CEFR")
        samples_per_level: Number of texts per CEFR level.

    Returns:
        Dict mapping CEFR level → list of text passages.
    """
    from huggingface_hub import HfApi, hf_hub_download

    cefr_levels = ["a1", "a2", "b1", "b2", "c1", "c2"]
    samples: Dict[str, List[str]] = {}

    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo, repo_type="dataset")

        for level in cefr_levels:
            level_files = [f for f in repo_files
                          if f.startswith(f"{level}/") and f.endswith(".txt")]
            if not level_files:
                continue

            selected = random.sample(
                level_files, min(samples_per_level, len(level_files))
            )
            texts = []
            for fpath in selected:
                try:
                    local = hf_hub_download(
                        repo, fpath, repo_type="dataset"
                    )
                    with open(local, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read().strip()
                    if text:
                        texts.append(text[:500])
                except Exception:
                    continue
            samples[level] = texts

        total = sum(len(v) for v in samples.values())
        log.info("Loaded %d CLC-CEFR calibration samples across %d levels from %s",
                 total, len(samples), repo)
    except Exception as e:
        log.warning("Failed to load CLC-CEFR samples from %s: %s", repo, e)

    return samples


def build_calibration_prompt(
    kidlm_samples: List[str],
    cefr_samples: Dict[str, List[str]],
) -> str:
    """Build few-shot calibration context from reference corpora."""
    parts = []

    if kidlm_samples:
        parts.append("## HIGH EDUCATIONAL QUALITY REFERENCE EXAMPLES")
        parts.append("The following are examples of text written for children "
                     "(high pedagogical quality, score 4-5):\n")
        for i, text in enumerate(kidlm_samples[:5], 1):
            parts.append(f"Example {i}: \"{text[:200]}...\"")
        parts.append("")

    if cefr_samples:
        parts.append("## DIFFICULTY CALIBRATION EXAMPLES (CEFR Levels)")
        parts.append("The following learner texts illustrate different "
                     "difficulty levels:\n")
        level_to_difficulty = {
            "a1": "elementary (difficulty 1)",
            "a2": "elementary-middle (difficulty 1-2)",
            "b1": "middle (difficulty 2)",
            "b2": "middle-high (difficulty 2-3)",
            "c1": "high (difficulty 3)",
            "c2": "high (difficulty 3)",
        }
        for level, texts in sorted(cefr_samples.items()):
            if texts:
                desc = level_to_difficulty.get(level, level)
                parts.append(f"{level.upper()} ({desc}): \"{texts[0][:150]}...\"")
        parts.append("")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Reservoir Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def reservoir_sample_from_parquet(
    decontaminated_dir: str,
    langs: List[str],
    total_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Reservoir-sample documents across all languages from Parquet shards.

    Samples proportionally (~total_samples / len(langs)) per language.
    Each returned dict has: text, lang, doc_id, shard_file.
    """
    rng = random.Random(seed)
    per_lang = max(1, total_samples // max(1, len(langs)))
    all_samples: List[Dict[str, Any]] = []

    for lang in langs:
        lang_dir = Path(decontaminated_dir) / lang
        if not lang_dir.exists():
            log.warning("No decontaminated data for %s at %s", lang, lang_dir)
            continue

        shards = sorted(lang_dir.glob("*.parquet"))
        if not shards:
            log.warning("No Parquet shards for %s", lang)
            continue

        # Reservoir sampling across shards
        reservoir: List[Dict[str, Any]] = []
        seen = 0

        for shard_path in shards:
            try:
                table = pq.read_table(shard_path, columns=["text", "doc_id"])
            except Exception as e:
                log.warning("Failed to read %s: %s", shard_path, e)
                continue

            for i in range(table.num_rows):
                text = table.column("text")[i].as_py()
                doc_id = table.column("doc_id")[i].as_py()

                if seen < per_lang:
                    reservoir.append({
                        "text": text,
                        "lang": lang,
                        "doc_id": doc_id,
                        "shard_file": str(shard_path),
                    })
                else:
                    j = rng.randint(0, seen)
                    if j < per_lang:
                        reservoir[j] = {
                            "text": text,
                            "lang": lang,
                            "doc_id": doc_id,
                            "shard_file": str(shard_path),
                        }
                seen += 1

            # Free memory
            del table

        log.info("  %s: sampled %d from %d docs across %d shards",
                 lang, len(reservoir), seen, len(shards))
        all_samples.extend(reservoir)

    rng.shuffle(all_samples)
    log.info("Total reservoir sample: %d documents across %d languages",
             len(all_samples), len(langs))
    return all_samples


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher Annotator
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherAnnotator(ABC):
    """Base class for LLM-based pedagogical annotation."""

    @abstractmethod
    async def annotate_batch(
        self, texts: List[str], calibration_ctx: str,
    ) -> List[Optional[dict]]:
        """Annotate a batch of texts. Returns parsed JSON dicts or None on failure."""
        ...


class VLLMAnnotator(TeacherAnnotator):
    """Annotator using a local vLLM server via OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-70B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        max_concurrent: int = 32,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_api(
        self,
        text: str,
        calibration_ctx: str,
        session,
    ) -> Optional[dict]:
        """Single API call with retry."""
        prompt = TEACHER_PROMPT.format(input_text=text[:2000])

        messages = []
        if calibration_ctx:
            messages.append({"role": "system", "content": calibration_ctx})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(2):  # 1 retry on malformed JSON
            try:
                async with self._semaphore:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        timeout=120,
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            log.warning("API error %d: %s", resp.status, body[:200])
                            return None
                        data = await resp.json()

                content = data["choices"][0]["message"]["content"]
                # Extract JSON from response (may have markdown fences)
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                parsed = json.loads(content)
                return parsed

            except json.JSONDecodeError:
                if attempt == 0:
                    continue  # retry once
                return None
            except Exception as e:
                if attempt == 0:
                    continue
                log.debug("Annotation failed after retry: %s", e)
                return None

        return None

    async def annotate_batch(
        self,
        texts: List[str],
        calibration_ctx: str,
    ) -> List[Optional[dict]]:
        """Annotate a batch of texts concurrently."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_api(text, calibration_ctx, session)
                for text in texts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            r if isinstance(r, dict) else None
            for r in results
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Main Annotation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnnotatorConfig:
    """Configuration for the annotation pipeline."""
    model: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    sample_size: int = 500_000
    batch_size: int = 16
    max_concurrent: int = 32
    kidlm_repo: str = "tafseer-nayeem/KidLM-corpus"
    clc_repo: str = "ADALM/CLC-L1-CEFR"
    kidlm_samples: int = 20
    samples_per_cefr_level: int = 3
    seed: int = 42


def annotate_all_languages(
    decontaminated_dir: str,
    output_dir: str,
    langs: List[str],
    cfg: AnnotatorConfig,
) -> AnnotationStats:
    """Run the full annotation pipeline across all languages.

    Steps:
      1. Load calibration examples from HF (KidLM + CLC-CEFR)
      2. Reservoir-sample documents across all languages
      3. Annotate in batches using vLLM
      4. Write JSONL per language + stats

    Args:
        decontaminated_dir: Path to pipeline_output/decontaminated/
        output_dir: Path to pipeline_output/ (annotations written to annotations/)
        langs: Language codes to process.
        cfg: Annotator configuration.

    Returns:
        AnnotationStats with counts and timing.
    """
    stats = AnnotationStats(model_name=cfg.model)
    t0 = time.time()

    # 1. Load calibration data
    log.info("Loading calibration data from HuggingFace...")
    kidlm_samples = load_kidlm_samples(cfg.kidlm_repo, cfg.kidlm_samples)
    cefr_samples = load_clc_cefr_samples(cfg.clc_repo, cfg.samples_per_cefr_level)
    calibration_ctx = build_calibration_prompt(kidlm_samples, cefr_samples)
    log.info("Calibration context: %d chars", len(calibration_ctx))

    # 2. Reservoir sample
    log.info("Reservoir sampling %d documents across %d languages...",
             cfg.sample_size, len(langs))
    samples = reservoir_sample_from_parquet(
        decontaminated_dir, langs, cfg.sample_size, cfg.seed,
    )
    stats.total_sampled = len(samples)

    # 3. Prepare output directories
    annotations_dir = Path(output_dir) / "annotations"
    # Create per-language output dirs
    for lang in langs:
        (annotations_dir / lang).mkdir(parents=True, exist_ok=True)

    # 4. Annotate in batches
    annotator = VLLMAnnotator(
        model=cfg.model,
        base_url=cfg.base_url,
        max_concurrent=cfg.max_concurrent,
    )

    # Group samples for batch processing
    # Write results per language as we go
    lang_writers: Dict[str, Any] = {}
    lang_shard_counts: Dict[str, int] = {}
    lang_buffers: Dict[str, List[dict]] = {lang: [] for lang in langs}
    SHARD_SIZE = 10_000

    log.info("Annotating %d documents in batches of %d (concurrency=%d)...",
             len(samples), cfg.batch_size, cfg.max_concurrent)

    for batch_start in range(0, len(samples), cfg.batch_size):
        batch_samples = samples[batch_start:batch_start + cfg.batch_size]
        batch_texts = [s["text"] for s in batch_samples]

        # Run async annotation
        results = asyncio.run(
            annotator.annotate_batch(batch_texts, calibration_ctx)
        )

        for sample, annotation in zip(batch_samples, results):
            lang = sample["lang"]
            if annotation is not None:
                record = {
                    "doc_id": sample["doc_id"],
                    "lang": lang,
                    "annotation": annotation,
                }
                lang_buffers[lang].append(record)
                stats.total_annotated += 1
                stats.per_lang_counts[lang] = stats.per_lang_counts.get(lang, 0) + 1

                # Flush buffer to JSONL shard
                if len(lang_buffers[lang]) >= SHARD_SIZE:
                    shard_idx = lang_shard_counts.get(lang, 0)
                    shard_path = (annotations_dir / lang /
                                 f"annotations_{shard_idx:05d}.jsonl")
                    with open(shard_path, "w") as f:
                        for rec in lang_buffers[lang]:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    lang_shard_counts[lang] = shard_idx + 1
                    lang_buffers[lang] = []
            else:
                stats.total_failed += 1

        # Progress log
        done = batch_start + len(batch_samples)
        if done % (cfg.batch_size * 100) == 0 or done >= len(samples):
            log.info("  Progress: %d/%d (%.1f%%) | annotated=%d failed=%d",
                     done, len(samples), 100 * done / len(samples),
                     stats.total_annotated, stats.total_failed)

    # 5. Flush remaining buffers
    for lang, buffer in lang_buffers.items():
        if buffer:
            shard_idx = lang_shard_counts.get(lang, 0)
            shard_path = (annotations_dir / lang /
                         f"annotations_{shard_idx:05d}.jsonl")
            with open(shard_path, "w") as f:
                for rec in buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lang_shard_counts[lang] = shard_idx + 1

    # 6. Write stats
    stats.wall_time_seconds = time.time() - t0
    stats_path = Path(output_dir) / "annotations" / "annotation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info("Annotation complete: %d/%d annotated (%.1f%%) in %.1f hours",
             stats.total_annotated, stats.total_sampled,
             100 * stats.total_annotated / max(1, stats.total_sampled),
             stats.wall_time_seconds / 3600)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage A: Teacher Annotation")
    parser.add_argument("--output-dir", type=str, default="pipeline_output")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--vllm-url", type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--lang", type=str, nargs="+", default=None,
                        help="Languages to sample from (default: all available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Discover available languages from decontaminated output
    decontaminated_dir = str(Path(args.output_dir) / "decontaminated")
    if args.lang:
        langs = args.lang
    else:
        decon_path = Path(decontaminated_dir)
        if decon_path.exists():
            langs = sorted([
                d.name for d in decon_path.iterdir()
                if d.is_dir() and list(d.glob("*.parquet"))
            ])
        else:
            log.error("No decontaminated data at %s", decontaminated_dir)
            return

    log.info("Languages: %s", langs)

    cfg = AnnotatorConfig(
        model=args.model,
        base_url=args.vllm_url,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
    )

    annotate_all_languages(decontaminated_dir, args.output_dir, langs, cfg)


if __name__ == "__main__":
    main()
