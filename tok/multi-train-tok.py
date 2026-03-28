#!/usr/bin/env python3
"""
Multi-language Byte-Level BPE Tokenizer
========================================
Script : train_tokenizer_multi.py
Model  : Byte-Level BPE
Data   : FineWeb-2 (language-specific) + FineWeb-edu (English)
Target : HuggingFace Hub → {HF_USER}/tokenizer-{LANG}-en

Supported languages
-------------------
  pl  Polish      nl  Dutch        es  Spanish     el  Greek
  fr  French      de  German       it  Italian     eu  Basque
  tr  Turkish     id  Indonesian   tl  Tagalog     fa  Persian
  hi  Hindi       ta  Tamil        sv  Swedish     ru  Russian
  ca  Catalan

Usage
-----
  python train_tokenizer_multi.py --lang pl
  python train_tokenizer_multi.py --lang fa --vocab-size 60000
  python train_tokenizer_multi.py --lang hi --hf-user MyOrg --sentences 3000000
  python train_tokenizer_multi.py --lang ta --no-push   # local only

Why Byte-Level BPE across all 17 languages?
--------------------------------------------
  - ByteLevel converts every input character to its raw UTF-8 byte sequence
    before tokenisation, so the 256-byte initial alphabet guarantees ZERO
    unknown tokens regardless of script (Cyrillic, Greek, Devanagari, Tamil …)
  - BPE merges then learn high-frequency byte-pair combinations, building up
    from individual bytes to full words/morphemes.
  - Works for space-delimited (European) and morphologically rich (Turkish,
    Finnish-type) languages alike.
  - Language-specific tuning lives entirely in the NORMALIZER layer (below),
    not in the BPE algorithm itself.

Key per-language normalisation decisions
-----------------------------------------
  ALL langs   : NFKC
    - Collapses Unicode Compatibility forms (full-width digits, presentation
      block glyphs, ligatures) into canonical base characters.
    - Safe for all 17 languages — does NOT remove meaningful diacritics.

  Persian (fa): NFKC + strip tatweel (U+0640)
    - Tatweel is a calligraphic letter-stretcher with zero linguistic content.
    - Arabic Presentation Forms (FE block) common in older Persian web text
      are normalised to canonical forms by NFKC.
    - ZWNJ (U+200C) is intentionally PRESERVED — it distinguishes word-internal
      morpheme boundaries in Persian (می‌روم vs میروم).

  All others  : NFKC only — diacritics in Polish, Turkish, Swedish, etc.
    are phonemically distinct and must NEVER be stripped.
"""

import os
import sys
import argparse
from pathlib import Path

from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    normalizers,
    processors,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import create_repo

# =====================================================
# LANGUAGE CONFIGS
# =====================================================
# fw2_name   : FineWeb-2 subset name (FLORES-200 code)
# normalizer : "nfkc" | "nfkc+tatweel"
# min_freq   : BPE min_frequency; lower for smaller corpora
# notes      : tokenisation decisions specific to this language
# test_cases : benchmark strings in (name → text) order

LANG_CONFIGS: dict = {

    # --------------------------------------------------
    # LATIN SCRIPT — WESTERN EUROPEAN
    # --------------------------------------------------

    "pl": {
        "name"      : "Polish",
        "fw2_name"  : "pol_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Heavily inflected (7 grammatical cases, grammatical gender).",
            "Diacritics ą ć ę ł ń ó ś ź ż are phonemically distinct — never strip.",
            "NFKC sufficient; no Presentation Form or compatibility block issues.",
            "BPE naturally learns common case suffixes (-owi, -ach, -ego …).",
        ],
        "test_cases": {
            "PL_Simple"    : "Cześć, jak się masz?",
            "PL_Diacritics": "Żółw skacze przez źródło w środę.",
            "PL_Inflection": "Przez nieporozumienie przyszedł do złego budynku.",
            "PL_Mixed"     : "Używamy Pythona do uczenia maszynowego.",
        },
    },

    "nl": {
        "name"      : "Dutch",
        "fw2_name"  : "nld_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Compound-heavy (stofzuigerzak, verantwoordelijkheid); BPE segments naturally.",
            "Diacritics rare (é ë for disambiguation only); NFKC sufficient.",
            "NFKC normalises typographic apostrophes common in Dutch contractions.",
        ],
        "test_cases": {
            "NL_Simple"  : "Hallo, hoe gaat het?",
            "NL_Compound": "De regeringsverantwoordelijkheid is groot.",
            "NL_Mixed"   : "We gebruiken machine learning voor taaltechnologie.",
        },
    },

    "es": {
        "name"      : "Spanish",
        "fw2_name"  : "spa_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Tildes (á é í ó ú ñ ü) are phonemically meaningful — never strip.",
            "Inverted punctuation (¿ ¡) handled naturally by ByteLevel.",
            "One of the largest corpora; generous data for sub-word learning.",
        ],
        "test_cases": {
            "ES_Simple" : "Hola, ¿cómo estás?",
            "ES_Accents": "El niño está jugando en el jardín bajo el cielo.",
            "ES_Mixed"  : "Usamos Python para el aprendizaje automático.",
        },
    },

    "fr": {
        "name"      : "French",
        "fw2_name"  : "fra_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Accents à â ç é è ê ë î ï ô ù û ü ÿ are meaningful.",
            "NFKC normalises curly quotes and typographic apostrophes (very common in FR).",
            "Elision (l'été, c'est) handled naturally — apostrophe separates tokens.",
        ],
        "test_cases": {
            "FR_Simple" : "Bonjour, comment allez-vous?",
            "FR_Accents": "L'été, nous préférons les crêpes avec du beurre.",
            "FR_Mixed"  : "Nous utilisons Python pour l'apprentissage automatique.",
        },
    },

    "de": {
        "name"      : "German",
        "fw2_name"  : "deu_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Umlauts ä ö ü and ß are distinct phonemes — never strip or fold.",
            "NFKC does NOT expand ß → ss (that is NFKD case-folding); ß is preserved.",
            "Long noun compounds tokenised into meaningful sub-words by BPE.",
            "Consider vocab_size 60k–80k for German to cover compound fragments.",
        ],
        "test_cases": {
            "DE_Simple"   : "Hallo, wie geht es Ihnen?",
            "DE_Umlauts"  : "Über die Größe des Brötchens lässt sich streiten.",
            "DE_Compound" : "Das Donaudampfschifffahrtsgesellschaftskapitänspatent.",
            "DE_Mixed"    : "Wir nutzen maschinelles Lernen für NLP-Aufgaben.",
        },
    },

    "it": {
        "name"      : "Italian",
        "fw2_name"  : "ita_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Accents à è é ì í ò ó ù are meaningful, mainly on final syllables.",
            "NFKC sufficient; no complex presentation form issues.",
        ],
        "test_cases": {
            "IT_Simple" : "Ciao, come stai?",
            "IT_Accents": "Però, è già tardi per andare al caffè.",
            "IT_Mixed"  : "Usiamo Python per l'apprendimento automatico.",
        },
    },

    "sv": {
        "name"      : "Swedish",
        "fw2_name"  : "swe_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Ring å and umlauts ä ö are distinct phonemes — never strip.",
            "Compound-heavy (like German); BPE handles via sub-word merges.",
            "NFKC sufficient.",
        ],
        "test_cases": {
            "SV_Simple"  : "Hej, hur mår du?",
            "SV_Umlauts" : "Åsa och Björn åker till Göteborg på lördag.",
            "SV_Compound": "Realisationsvinstbeskattning är ett långt ord.",
            "SV_Mixed"   : "Vi använder Python för maskininlärning.",
        },
    },

    "ca": {
        "name"      : "Catalan",
        "fw2_name"  : "cat_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Accents à è é í ï ó ò ú ü are meaningful.",
            "Geminated l·l (U+00B7 middle dot) is normalised correctly by NFKC.",
            "Lexically close to Spanish and French; shared sub-word tokens likely.",
        ],
        "test_cases": {
            "CA_Simple" : "Hola, com estàs?",
            "CA_Accents": "El col·legi és a l'avinguda principal del barri.",
            "CA_Mixed"  : "Fem servir Python per a l'aprenentatge automàtic.",
        },
    },

    # --------------------------------------------------
    # LATIN SCRIPT — AGGLUTINATIVE / ISOLATE
    # --------------------------------------------------

    "eu": {
        "name"      : "Basque",
        "fw2_name"  : "eus_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 1,   # lower-resource: min_freq=1 to preserve rare morphemes
        "notes": [
            "Language isolate; highly agglutinative with complex suffix stacking.",
            "BPE naturally learns frequent Basque suffixes (-rekin, -ean, -ko, -ren).",
            "min_freq=1 because corpus is smaller; avoids discarding valid morphemes.",
            "Code-switches heavily with Spanish in web data.",
        ],
        "test_cases": {
            "EU_Simple"       : "Kaixo, nola zaude?",
            "EU_Agglutinative": "Etxekoandrearekin hitz egin dugu atzo arratsaldean.",
            "EU_Mixed"        : "Python erabiliz hizkuntza-prozesatzea egiten dugu.",
        },
    },

    "tr": {
        "name"      : "Turkish",
        "fw2_name"  : "tur_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Highly agglutinative; one word can encode an entire clause.",
            "Dotted/dotless I: ı (U+0131) and İ (U+0130) vs i and I.",
            "NFKC does NOT merge these — correct, they are distinct phonemes.",
            "BPE learns common suffixes (-lar/-ler, -da/-de, -ın/-in, -sız/-siz).",
        ],
        "test_cases": {
            "TR_Simple"      : "Merhaba, nasılsınız?",
            "TR_Agglutinative": "Çekoslovakyalılaştıramadıklarımızdanmışsınız.",
            "TR_DottedI"     : "İstanbul ve Izmir büyük şehirlerdir.",
            "TR_Mixed"       : "Python kullanarak doğal dil işleme yapıyoruz.",
        },
    },

    "id": {
        "name"      : "Indonesian",
        "fw2_name"  : "ind_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Agglutinative; productive prefix/suffix morphology (me-, ber-, -kan, -an).",
            "Low diacritic usage; mostly ASCII-range Latin in modern text.",
            "NFKC handles any legacy characters in older web text.",
        ],
        "test_cases": {
            "ID_Simple"    : "Halo, apa kabar?",
            "ID_Morphology": "Mempertanggungjawabkan keputusan ini sangat penting.",
            "ID_Mixed"     : "Kami menggunakan Python untuk pembelajaran mesin.",
        },
    },

    "tl": {
        "name"      : "Tagalog",
        "fw2_name"  : "tgl_Latn",
        "script"    : "Latin",
        "normalizer": "nfkc",
        "min_freq"  : 1,   # lower-resource; keep rare affixes
        "notes": [
            "Agglutinative; focus-marking affixes (mag-, -in, -an, -um-) are central.",
            "Occasional Spanish loanword diacritics (á, é); NFKC sufficient.",
            "Heavy English code-switching in web data; bilingual sub-words expected.",
            "min_freq=1 due to smaller corpus size in FineWeb-2.",
        ],
        "test_cases": {
            "TL_Simple" : "Kamusta ka?",
            "TL_Affixes": "Pinaghandaan namin ang pagtatanghal para sa mga manonood.",
            "TL_Mixed"  : "Gumagamit kami ng Python para sa machine learning.",
        },
    },

    # --------------------------------------------------
    # NON-LATIN SCRIPTS — EUROPEAN
    # --------------------------------------------------

    "el": {
        "name"      : "Greek",
        "fw2_name"  : "ell_Grek",
        "script"    : "Greek",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Greek script (U+0370–U+03FF); each char = 2 bytes in UTF-8.",
            "NFKC collapses Greek Compatibility block (U+1F00–U+1FFF) polytonic",
            "  precomposed forms → monotonic equivalents used in modern Greek.",
            "256-byte initial alphabet means all Greek chars are covered from byte 0.",
            "Monotonic accent (single tonos) is preserved by NFKC — meaningful.",
        ],
        "test_cases": {
            "EL_Simple"   : "Γεια σου, πώς είσαι;",
            "EL_Polytonic": "Ἐν ἀρχῇ ἦν ὁ λόγος.",   # ancient Greek (polytonic)
            "EL_Mixed"    : "Χρησιμοποιούμε Python για μηχανική μάθηση.",
        },
    },

    "ru": {
        "name"      : "Russian",
        "fw2_name"  : "rus_Cyrl",
        "script"    : "Cyrillic",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Cyrillic script (U+0400–U+04FF); each char = 2 bytes in UTF-8.",
            "NFKC normalises rare Cyrillic compatibility characters.",
            "Stress marks (U+0301, combining accent) appear in dictionaries but",
            "  almost never in web text; NFKC leaves them unchanged (correct).",
            "Uses spaces between words; ByteLevel handles naturally.",
        ],
        "test_cases": {
            "RU_Simple"    : "Привет, как дела?",
            "RU_Morphology": "Непротивостоянию властям посвятили отдельную главу.",
            "RU_Mixed"     : "Мы используем Python для машинного обучения.",
        },
    },

    # --------------------------------------------------
    # NON-LATIN SCRIPTS — MIDDLE EAST / SOUTH ASIA
    # --------------------------------------------------

    "fa": {
        "name"      : "Persian",
        "fw2_name"  : "pes_Arab",   # Western Persian (Farsi), Arabic script
        "script"    : "Arabic",
        "normalizer": "nfkc+tatweel",
        "min_freq"  : 2,
        "notes": [
            "Uses Arabic script + Persian-specific chars پ (U+067E) چ (U+0686) ژ (U+0698) گ (U+06AF).",
            "Each char = 2 bytes in UTF-8; 256-byte alphabet covers all of them.",
            "NFKC collapses Arabic Presentation Forms (FE block) common in older Persian web text.",
            "STRIP tatweel U+0640 — purely calligraphic; never carries meaning.",
            "PRESERVE ZWNJ (U+200C): marks word-internal morpheme boundaries in Persian.",
            "  e.g. می‌روم (mí-ravam) vs میروم — ZWNJ is retained by NFKC.",
            "Persian uses spaces between words more consistently than Arabic.",
        ],
        "test_cases": {
            "FA_Simple"  : "سلام، حال شما چطور است؟",
            "FA_ZWNJ"    : "می‌روم به مدرسه هر روز صبح.",
            "FA_Tatweel" : "کتـــاب",   # tatweel stripped → same tokens as کتاب
            "FA_Mixed"   : "ما از Python برای یادگیری ماشین استفاده می‌کنیم.",
        },
    },

    "hi": {
        "name"      : "Hindi",
        "fw2_name"  : "hin_Deva",
        "script"    : "Devanagari",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Devanagari script (U+0900–U+097F); chars are 3 bytes in UTF-8.",
            "NFKC is critical: normalises composed/decomposed forms of matras and",
            "  the virama (U+094D halant) — multiple valid representations collapse to one.",
            "ByteLevel learns Devanagari char and syllable merges given enough data.",
            "Hindi uses spaces between words; prefix space convention is not needed.",
        ],
        "test_cases": {
            "HI_Simple"  : "नमस्ते, आप कैसे हैं?",
            "HI_Compound": "विश्वविद्यालय में प्रवेश परीक्षा कल होगी।",
            "HI_Mixed"   : "हम Python का उपयोग मशीन लर्निंग के लिए करते हैं।",
        },
    },

    "ta": {
        "name"      : "Tamil",
        "fw2_name"  : "tam_Taml",
        "script"    : "Tamil",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Tamil script (U+0B80–U+0BFF); chars are 3 bytes in UTF-8.",
            "NFKC normalises composed forms; Tamil has no Compatibility block issues.",
            "Agglutinative; complex suffix stacking handled well by BPE sub-words.",
            "Uses spaces between words.",
        ],
        "test_cases": {
            "TA_Simple"  : "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "TA_Compound": "தமிழ்நாட்டில் பல அழகான கோயில்கள் உள்ளன.",
            "TA_Mixed"   : "நாங்கள் Python ஐ இயந்திர கற்றலுக்கு பயன்படுத்துகிறோம்.",
        },
    },

}

# Languages with smaller FineWeb-2 corpora; reduce sentences if streaming stalls.
LOWER_RESOURCE_LANGS = {"eu", "tl", "ta"}

# =====================================================
# CLI
# =====================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Byte-Level BPE tokenizer for one of 17 languages.")
    p.add_argument("--lang",        required=True,  choices=sorted(LANG_CONFIGS),
                   help="ISO-639-1 language code.")
    p.add_argument("--hf-user",     default="RA-ALTA",
                   help="HuggingFace username / org (default: RA-ALTA).")
    p.add_argument("--vocab-size",  type=int, default=50_000,
                   help="BPE vocabulary size (default: 50000).")
    p.add_argument("--sentences",   type=int, default=2_000_000,
                   help="Total training sentences, split 50/50 lang/EN (default: 2M).")
    p.add_argument("--no-push",     action="store_true",
                   help="Skip HuggingFace Hub push.")
    p.add_argument("--out-dir",     default=None,
                   help="Local output directory (default: tokenizer-{lang}-local).")
    return p.parse_args()

# =====================================================
# 1. DATA GENERATOR
# =====================================================
def get_training_corpus(cfg: dict, n_sentences: int):
    """
    Yields alternating target-language and English sentences.
    Both datasets are streamed — no local disk required.
    """
    lang_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name=cfg["fw2_name"],
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    en_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    lang_iter = iter(lang_ds)
    en_iter   = iter(en_ds)

    yielded = 0
    for _ in range(n_sentences // 2):
        try:
            yield next(lang_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
            yielded += 2
        except StopIteration:
            print(f"⚠️  Data exhausted after {yielded:,} sentences.")
            break

# =====================================================
# 2. BUILD NORMALIZER
# =====================================================
def build_normalizer(norm_type: str):
    """
    Returns the tokenizers normalizer for the given type string.

    "nfkc"          → NFKC only (safe for all Latin/Cyrillic/Greek/Indic scripts)
    "nfkc+tatweel"  → NFKC then strip U+0640 (for Arabic-script languages: fa)
    """
    if norm_type == "nfkc":
        return normalizers.NFKC()
    elif norm_type == "nfkc+tatweel":
        return normalizers.Sequence([
            normalizers.NFKC(),
            # Tatweel (kashida) U+0640 — calligraphic letter-stretcher; no meaning.
            # Appears frequently in Persian/Arabic web text. Strip before training
            # so "کتـاب" and "کتاب" map to identical token sequences.
            normalizers.Replace(pattern="\u0640", content=""),
        ])
    else:
        raise ValueError(f"Unknown normalizer type: {norm_type!r}")

# =====================================================
# 3. BUILD TOKENIZER
# =====================================================
def build_tokenizer(cfg: dict) -> Tokenizer:
    """
    Assembles the full tokenizer pipeline.

    Pipeline
    --------
    Normalizer  → language-specific (see build_normalizer)
    Pre-tokenizer → ByteLevel(add_prefix_space=False)
      - Converts every character to its raw UTF-8 bytes before BPE
      - 256-byte alphabet guarantees zero unknown tokens for ALL scripts
      - add_prefix_space=False: avoids spurious leading Ġ for non-English text
    BPE model
    Post-processor → ByteLevel(trim_offsets=False)
      - Ensures offset mappings are consistent with ByteLevel encoding
    Decoder → ByteLevel
      - Converts byte sequences back to Unicode strings on decode
    """
    tokenizer = Tokenizer(models.BPE())

    tokenizer.normalizer    = build_normalizer(cfg["normalizer"])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder       = decoders.ByteLevel()

    return tokenizer

# =====================================================
# 4. TRAIN & PUSH
# =====================================================
def train_and_push(
    lang      : str,
    hf_user   : str,
    vocab_size: int,
    n_sentences: int,
    out_dir   : Path,
    push      : bool,
):
    cfg     = LANG_CONFIGS[lang]
    repo_id = f"{hf_user}/tokenizer-{lang}-en"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"\n{'='*60}")
    print(f" Language : {cfg['name']} ({lang})")
    print(f" Script   : {cfg['script']}")
    print(f" FW2 name : {cfg['fw2_name']}")
    print(f" Vocab    : {vocab_size:,}")
    print(f" Sentences: {n_sentences:,}")
    print(f" Repo     : {repo_id}")
    print(f"{'='*60}\n")

    # Print language-specific notes
    for note in cfg["notes"]:
        print(f"  ℹ  {note}")
    print()

    tokenizer = build_tokenizer(cfg)

    trainer = trainers.BpeTrainer(
        vocab_size       = vocab_size,
        min_frequency    = cfg["min_freq"],
        special_tokens   = ["<unk>", "<s>", "</s>", "<pad>"],
        # Include all 256 possible byte values in the initial alphabet.
        # This ensures that even rare scripts (Tamil, Devanagari, Cyrillic …)
        # have their component bytes in the vocab before any merges happen,
        # making OOV tokens structurally impossible.
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"🚀 Training Byte-Level BPE …")
    tokenizer.train_from_iterator(
        get_training_corpus(cfg, n_sentences),
        trainer=trainer,
    )

    # ---- Wrap in HF PreTrainedTokenizerFast ----
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object            = tokenizer,
        bos_token                   = "<s>",
        eos_token                   = "</s>",
        pad_token                   = "<pad>",
        unk_token                   = "<unk>",
        # ByteLevel decoder owns spacing; HF must not add its own cleanup
        clean_up_tokenization_spaces = False,
    )

    hf_tokenizer.save_pretrained(out_dir)
    print(f"✅ Tokenizer saved → {out_dir}")

    if push:
        if not hf_token:
            print("⚠️  HF_TOKEN not set — skipping Hub push.")
            return
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"✅ Pushed → https://huggingface.co/{repo_id}")

# =====================================================
# 5. BENCHMARK
# =====================================================
def run_benchmark(lang: str, out_dir: Path):
    """
    Tests tokenisation correctness and token-efficiency for:
      - Language-specific cases (diacritics, morphology, script stress tests)
      - English (verifies bilingual sub-word learning)
      - Mixed language + code-switching
      - Emoji (ByteLevel must handle 4-byte UTF-8 emoji)

    Checks:
      1. Round-trip losslessness  : decode(encode(text)).strip() == text.strip()
         For Persian + tatweel: decoded text will not contain tatweel (correct).
      2. Token-efficiency ratio   : tokens / chars  (lower = better compression)
    """
    cfg = LANG_CONFIGS[lang]
    tk  = AutoTokenizer.from_pretrained(out_dir)

    print(f"\n🧪 Benchmark — {cfg['name']} ({lang})\n" + "─" * 60)

    # Shared cases added to every language
    shared_cases = {
        "EN_Simple": "The quick brown fox jumps over the lazy dog.",
        "Emoji"    : "Learning is fun! 🚀🔥",
    }

    all_cases  = {**cfg["test_cases"], **shared_cases}
    all_passed = True

    for name, text in all_cases.items():
        tokens  = tk.encode(text)
        decoded = tk.decode(tokens)
        n_tok   = len(tokens)
        n_chr   = len(text)
        ratio   = n_tok / n_chr

        # For tatweel test: normaliser strips U+0640 so decoded won't contain it
        expected = text.replace("\u0640", "") if lang == "fa" else text
        ok       = decoded.strip() == expected.strip()
        status   = "✅" if ok else "❌ MISMATCH"
        if not ok:
            all_passed = False

        print(f"  [{name}]")
        print(f"    Input  : {text!r}")
        print(f"    Tokens : {n_tok}  (chars={n_chr}, tok/char={ratio:.2f})")
        print(f"    IDs    : {tokens[:8]}{'…' if n_tok > 8 else ''}")
        print(f"    Decoded: {decoded!r}  {status}")
        print()

    marker = "🎉 All checks passed." if all_passed else "⚠️  Some checks FAILED."
    print(marker)

# =====================================================
# 6. VOCAB INSPECTION
# =====================================================
def inspect_vocab(lang: str, out_dir: Path, n: int = 50):
    """
    Sample the learned vocabulary.
    For non-Latin scripts look for target-language tokens (not just bytes).
    For agglutinative languages (tr, eu, fi-like) look for suffix tokens.
    """
    cfg   = LANG_CONFIGS[lang]
    tk    = AutoTokenizer.from_pretrained(out_dir)
    vocab = tk.get_vocab()
    specials = {"<unk>", "<s>", "</s>", "<pad>"}

    print(f"\n🔍 Vocab sample — {cfg['name']} (first {n} non-special)\n" + "─" * 40)
    shown = 0
    for piece, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if piece in specials:
            continue
        print(f"  {idx:6d}  {piece!r}")
        shown += 1
        if shown >= n:
            break
    print(f"\nTotal vocab size: {len(vocab):,}")

# =====================================================
# MAIN
# =====================================================
def main():
    args    = parse_args()
    lang    = args.lang
    cfg     = LANG_CONFIGS[lang]
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"tokenizer-{lang}-local")
    out_dir.mkdir(exist_ok=True)

    if lang in LOWER_RESOURCE_LANGS and args.sentences > 1_000_000:
        print(
            f"⚠️  {cfg['name']} is lower-resource. "
            f"Consider --sentences 500000 to avoid long waits on exhausted streams."
        )

    train_and_push(
        lang        = lang,
        hf_user     = args.hf_user,
        vocab_size  = args.vocab_size,
        n_sentences = args.sentences,
        out_dir     = out_dir,
        push        = not args.no_push,
    )
    run_benchmark(lang, out_dir)
    inspect_vocab(lang, out_dir, n=50)


if __name__ == "__main__":
    main()
