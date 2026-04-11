"""
utils.py — Shared text normalization and n-gram extraction utilities.

Reuses the tokenization approach from data/decontamination.py:329-330
for consistency with the existing Decontaminator class.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# Text Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """NFKC normalize, strip control characters, collapse whitespace.

    Mirrors BasicCleaner.clean() from data/decontamination.py:148-155.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def passes_length_filter(text: str, min_chars: int = 200, max_chars: int = 100_000) -> bool:
    """Check document meets length requirements."""
    n = len(text)
    return min_chars <= n <= max_chars


# ═══════════════════════════════════════════════════════════════════════════════
# N-Gram Extraction (for decontamination)
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_for_ngrams(text: str) -> List[str]:
    r"""Extract word tokens for n-gram matching.

    Uses the same regex as data/decontamination.py:329-330:
        re.findall(r"\b\w+\b", text.lower())

    This handles all scripts (Latin, CJK, Arabic, Cyrillic, etc.) by
    matching Unicode word characters, and lowercases for case-insensitive
    matching.
    """
    return re.findall(r"\b\w+\b", text.lower())


def extract_ngrams(tokens: List[str], n: int = 13) -> Set[Tuple[str, ...]]:
    """Extract all n-grams of size `n` from a token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def extract_ngrams_from_text(text: str, n: int = 13) -> Set[Tuple[str, ...]]:
    """Convenience: tokenize then extract n-grams."""
    return extract_ngrams(tokenize_for_ngrams(text), n)


# ═══════════════════════════════════════════════════════════════════════════════
# Word Count (proxy for BPE token count)
# ═══════════════════════════════════════════════════════════════════════════════

def word_count(text: str) -> int:
    """Fast whitespace word count as proxy for BPE token count.

    The ratio of BPE tokens to whitespace words varies by language:
      - English: ~1.3x
      - Most European languages: ~1.3-1.5x
      - Agglutinative (Turkish, Finnish): ~1.5-1.8x
      - CJK: ~2-3x (characters, not words)

    We overshoot the word target by 15% in the pipeline config to compensate.
    """
    return len(text.split())
