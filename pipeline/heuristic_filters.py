"""
heuristic_filters.py — Fast first-pass text triage for BeetleStream v2.

Stateless filtering functions that cheaply discard obviously low-quality
documents before expensive student model scoring. Targets ~10-15% removal
of clearly non-textual or boilerplate content.

Filters:
  - stopword_density:      Per-language stopword ratio check
  - flesch_kincaid_grade:   English-only readability (FK formula)
  - sentence_length_ratio:  Non-English readability proxy
  - script_consistency:     Expected script check per language
  - repetition_ratio:       5-gram uniqueness check
  - passes_heuristics:      Combined gate (calls all above)

All functions are pure and stateless — safe for multiprocessing.

Usage:
    from pipeline.heuristic_filters import passes_heuristics

    if passes_heuristics(text, lang="fr", cfg=heuristic_cfg):
        # document passes → send to student model
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HeuristicConfig:
    """Thresholds for heuristic text filtering."""
    stopword_density_min: float = 0.03
    stopword_density_max: float = 0.60
    max_fk_grade: float = 18.0
    min_fk_grade: float = 1.0
    min_script_consistency: float = 0.60
    min_unique_5gram_ratio: float = 0.30
    # Sentence length bounds for non-English readability proxy
    min_avg_sentence_len: float = 3.0    # words per sentence
    max_avg_sentence_len: float = 80.0   # words per sentence
    min_char_word_ratio: float = 2.0     # avg chars per word
    max_char_word_ratio: float = 20.0    # avg chars per word


DEFAULT_CONFIG = HeuristicConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# Stopword Lists (common function words per language)
# ═══════════════════════════════════════════════════════════════════════════════

# Compact stopword sets for major language families.
# These are intentionally small (high-frequency function words only)
# to keep the filter fast and memory-light.

_STOPWORDS: Dict[str, Set[str]] = {
    "en": {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
           "have", "has", "had", "do", "does", "did", "will", "would", "could",
           "should", "may", "might", "shall", "can", "need", "dare", "ought",
           "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
           "as", "into", "through", "during", "before", "after", "above",
           "below", "between", "out", "off", "over", "under", "again",
           "further", "then", "once", "here", "there", "when", "where", "why",
           "how", "all", "each", "every", "both", "few", "more", "most",
           "other", "some", "such", "no", "nor", "not", "only", "own", "same",
           "so", "than", "too", "very", "just", "because", "but", "and", "or",
           "if", "while", "about", "up", "it", "its", "this", "that", "these",
           "those", "i", "me", "my", "myself", "we", "our", "ours", "you",
           "your", "he", "him", "his", "she", "her", "they", "them", "their",
           "what", "which", "who", "whom"},
    "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
           "et", "est", "en", "que", "qui", "dans", "ce", "il", "ne", "sur",
           "pas", "plus", "par", "je", "se", "son", "sa", "ses", "avec",
           "tout", "mais", "ou", "comme", "pour", "nous", "vous", "leur",
           "cette", "mon", "ton", "bien", "aussi", "entre", "autre", "même"},
    "de": {"der", "die", "das", "ein", "eine", "und", "ist", "in", "den",
           "von", "zu", "mit", "auf", "für", "an", "dem", "des", "sich",
           "nicht", "als", "auch", "es", "ich", "er", "sie", "wir", "aber",
           "wie", "hat", "noch", "nach", "aus", "bei", "nur", "so", "wenn",
           "war", "kann", "über", "vor", "oder", "werden", "sein", "diese"},
    "es": {"el", "la", "los", "las", "un", "una", "de", "en", "que", "y",
           "es", "por", "con", "no", "se", "del", "al", "lo", "como", "más",
           "pero", "su", "le", "ya", "o", "fue", "este", "ha", "si", "porque",
           "esta", "son", "entre", "cuando", "muy", "sin", "sobre", "ser",
           "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo"},
    "zh": set(),   # Chinese: character-based, stopword density not meaningful
    "ja": set(),   # Japanese: character-based, stopword density not meaningful
    "ar": {"في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي",
           "الذي", "التي", "كان", "قد", "لا", "ما", "أن", "هو", "هي",
           "بين", "كل", "ذلك", "بعد", "عند", "لم", "حتى", "إذا", "ثم"},
    "ru": {"и", "в", "не", "на", "я", "что", "он", "с", "это", "как",
           "а", "но", "по", "она", "к", "из", "то", "за", "его", "от",
           "же", "все", "так", "мы", "бы", "вы", "да", "ее", "уже", "для"},
    "tr": {"bir", "bu", "da", "de", "ve", "için", "ile", "çok", "daha",
           "ne", "var", "ben", "o", "gibi", "ama", "en", "ya", "kadar",
           "her", "hem", "olan", "olarak", "sonra", "bunu", "ki", "ayrıca"},
    "hi": {"का", "की", "के", "में", "है", "और", "को", "से", "पर", "ने",
           "यह", "एक", "हैं", "नहीं", "कि", "भी", "या", "था", "इस", "तो"},
    "nl": {"de", "het", "een", "van", "en", "in", "is", "dat", "op", "te",
           "voor", "met", "zijn", "er", "aan", "niet", "ook", "om", "maar",
           "als", "uit", "nog", "bij", "wel", "dan", "dit", "wat", "door"},
    "pl": {"i", "w", "nie", "na", "się", "z", "do", "to", "że", "jest",
           "ale", "o", "co", "jak", "za", "tak", "od", "po", "ten", "już",
           "czy", "ty", "tego", "być", "by", "jej", "go", "jego", "ja"},
    "it": {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di",
           "a", "da", "in", "con", "su", "per", "tra", "fra", "che", "è",
           "non", "ci", "si", "ne", "ma", "anche", "come", "io", "se", "più"},
    "sv": {"och", "i", "att", "en", "det", "som", "är", "för", "av", "den",
           "till", "på", "med", "har", "de", "inte", "om", "ett", "var",
           "från", "kan", "men", "så", "vi", "ska", "jag", "han", "hon"},
    "pt": {"o", "a", "os", "as", "um", "uma", "de", "em", "que", "e",
           "é", "para", "com", "não", "se", "por", "mais", "do", "da",
           "ao", "como", "mas", "foi", "também", "eu", "ele", "ela", "nos"},
}

# Languages that use character-based scripts where stopword density
# is not a meaningful signal (skip stopword check for these).
_SKIP_STOPWORD_LANGS = {"zh", "ja"}


def _get_stopwords(lang: str) -> Set[str]:
    """Return stopword set for a language, falling back to empty."""
    return _STOPWORDS.get(lang, set())


# ═══════════════════════════════════════════════════════════════════════════════
# Script Detection (for script_consistency filter)
# ═══════════════════════════════════════════════════════════════════════════════

# Map language codes to expected Unicode script names.
# Derived from LANG_REGISTRY fw2_name suffixes (e.g., "pol_Latn" → Latin).
_LANG_SCRIPTS: Dict[str, str] = {
    "pl": "LATIN", "nl": "LATIN", "es": "LATIN", "fr": "LATIN",
    "de": "LATIN", "it": "LATIN", "eu": "LATIN", "tr": "LATIN",
    "id": "LATIN", "tl": "LATIN", "sv": "LATIN", "ca": "LATIN",
    "da": "LATIN", "cs": "LATIN", "vi": "LATIN", "hr": "LATIN",
    "sl": "LATIN", "hu": "LATIN", "so": "LATIN", "yo": "LATIN",
    "wo": "LATIN", "pt": "LATIN", "en": "LATIN",
    "el": "GREEK",
    "ru": "CYRILLIC", "bg": "CYRILLIC", "uk": "CYRILLIC",
    "ar": "ARABIC", "fa": "ARABIC", "ur": "ARABIC",
    "hi": "DEVANAGARI", "gu": "GUJARATI", "ta": "TAMIL",
    "bn": "BENGALI", "am": "ETHIOPIC", "th": "THAI",
    "ko": "HANGUL",
    "ja": "CJK",  # Mix of CJK + Hiragana + Katakana
    "zh": "CJK",
}

# Unicode category prefixes for each script
_SCRIPT_CATEGORIES: Dict[str, set] = {}


def _char_matches_script(char: str, script: str) -> bool:
    """Check if a character belongs to the expected script."""
    if not char.strip():
        return True  # whitespace is script-neutral

    try:
        name = unicodedata.name(char, "")
    except ValueError:
        return False

    name_upper = name.upper()

    if script == "LATIN":
        return "LATIN" in name_upper
    elif script == "GREEK":
        return "GREEK" in name_upper
    elif script == "CYRILLIC":
        return "CYRILLIC" in name_upper
    elif script == "ARABIC":
        return "ARABIC" in name_upper
    elif script == "DEVANAGARI":
        return "DEVANAGARI" in name_upper
    elif script == "GUJARATI":
        return "GUJARATI" in name_upper
    elif script == "TAMIL":
        return "TAMIL" in name_upper
    elif script == "BENGALI":
        return "BENGALI" in name_upper or "BENGALI" in name_upper
    elif script == "ETHIOPIC":
        return "ETHIOPIC" in name_upper
    elif script == "THAI":
        return "THAI" in name_upper
    elif script == "HANGUL":
        return "HANGUL" in name_upper or "CJK" in name_upper
    elif script == "CJK":
        return "CJK" in name_upper or "HIRAGANA" in name_upper or "KATAKANA" in name_upper
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Filter Functions
# ═══════════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?。！？]+", re.UNICODE)


def stopword_density(text: str, lang: str) -> float:
    """Compute fraction of tokens that are stopwords.

    Returns 0.0 for character-based languages (zh, ja) where stopword
    density is not meaningful.
    """
    if lang in _SKIP_STOPWORD_LANGS:
        return 0.0  # always passes (handled by other filters)

    stops = _get_stopwords(lang)
    if not stops:
        return 0.0  # no stopword list → skip this check

    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0

    n_stops = sum(1 for w in words if w in stops)
    return n_stops / len(words)


def flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level. English only.

    FK Grade = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Returns -1.0 if text is too short to compute reliably.
    """
    words = _WORD_RE.findall(text)
    if len(words) < 10:
        return -1.0

    # Count sentences (split on sentence-ending punctuation)
    sentences = _SENTENCE_RE.split(text)
    sentences = [s for s in sentences if s.strip()]
    n_sentences = max(1, len(sentences))
    n_words = len(words)

    # Estimate syllables (vowel-group heuristic)
    n_syllables = 0
    for word in words:
        word_lower = word.lower()
        vowels = re.findall(r"[aeiouy]+", word_lower)
        syl = max(1, len(vowels))
        # Subtract silent 'e' at end
        if word_lower.endswith("e") and syl > 1:
            syl -= 1
        n_syllables += syl

    grade = (0.39 * (n_words / n_sentences) +
             11.8 * (n_syllables / n_words) -
             15.59)
    return grade


def sentence_length_ratio(text: str) -> tuple:
    """Language-agnostic readability proxy for non-English text.

    Returns (avg_sentence_length_in_words, avg_chars_per_word).
    These proxy readability without language-specific formulas.
    """
    words = _WORD_RE.findall(text)
    if len(words) < 5:
        return (0.0, 0.0)

    sentences = _SENTENCE_RE.split(text)
    sentences = [s for s in sentences if s.strip()]
    n_sentences = max(1, len(sentences))

    avg_sent_len = len(words) / n_sentences
    avg_char_per_word = sum(len(w) for w in words) / len(words)

    return (avg_sent_len, avg_char_per_word)


def script_consistency(text: str, lang: str) -> float:
    """Fraction of non-whitespace characters matching expected script.

    Returns 1.0 if no script mapping exists for the language.
    """
    expected_script = _LANG_SCRIPTS.get(lang)
    if expected_script is None:
        return 1.0  # unknown lang → pass

    # Sample characters (check first 500 non-whitespace chars for speed)
    chars = [c for c in text if not c.isspace() and not c.isdigit()
             and c not in '.,;:!?-()[]{}"\'/\\@#$%^&*+=<>~`|_']
    if not chars:
        return 0.0

    sample = chars[:500]
    matches = sum(1 for c in sample if _char_matches_script(c, expected_script))
    return matches / len(sample)


def repetition_ratio(text: str) -> float:
    """Ratio of unique 5-grams to total 5-grams.

    Low values indicate highly repetitive text (boilerplate, spam).
    Returns 1.0 for very short texts.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < 10:
        return 1.0

    ngrams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    if not ngrams:
        return 1.0

    return len(set(ngrams)) / len(ngrams)


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Gate
# ═══════════════════════════════════════════════════════════════════════════════

def passes_heuristics(
    text: str,
    lang: str,
    cfg: Optional[HeuristicConfig] = None,
) -> bool:
    """Apply all heuristic filters. Returns True if document passes.

    Dispatches FK for English, sentence_length_ratio for non-English.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # 1. Stopword density
    sd = stopword_density(text, lang)
    if lang not in _SKIP_STOPWORD_LANGS:
        stopwords = _get_stopwords(lang)
        if stopwords:  # only check if we have a stopword list
            if sd < cfg.stopword_density_min or sd > cfg.stopword_density_max:
                return False

    # 2. Readability
    if lang == "en":
        fk = flesch_kincaid_grade(text)
        if fk >= 0 and (fk > cfg.max_fk_grade or fk < cfg.min_fk_grade):
            return False
    else:
        avg_sent_len, avg_cpw = sentence_length_ratio(text)
        if avg_sent_len > 0:
            if (avg_sent_len < cfg.min_avg_sentence_len or
                    avg_sent_len > cfg.max_avg_sentence_len):
                return False
            if (avg_cpw < cfg.min_char_word_ratio or
                    avg_cpw > cfg.max_char_word_ratio):
                return False

    # 3. Script consistency
    sc = script_consistency(text, lang)
    if sc < cfg.min_script_consistency:
        return False

    # 4. Repetition
    rr = repetition_ratio(text)
    if rr < cfg.min_unique_5gram_ratio:
        return False

    return True


def filter_stats(
    text: str,
    lang: str,
    cfg: Optional[HeuristicConfig] = None,
) -> Dict[str, object]:
    """Return all filter scores for a document (for debugging / analysis)."""
    if cfg is None:
        cfg = DEFAULT_CONFIG

    sd = stopword_density(text, lang)
    sc = script_consistency(text, lang)
    rr = repetition_ratio(text)

    stats = {
        "stopword_density": sd,
        "script_consistency": sc,
        "repetition_ratio": rr,
        "passes": passes_heuristics(text, lang, cfg),
    }

    if lang == "en":
        stats["flesch_kincaid_grade"] = flesch_kincaid_grade(text)
    else:
        avg_sl, avg_cpw = sentence_length_ratio(text)
        stats["avg_sentence_length"] = avg_sl
        stats["avg_chars_per_word"] = avg_cpw

    return stats
