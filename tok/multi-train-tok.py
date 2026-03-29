#!/usr/bin/env python3
"""
Multi-language Tokenizer Trainer
==================================
Script : train_tokenizer_multi.py
Data   : FineWeb-2 (language-specific) + FineWeb-edu (English)
Target : HuggingFace Hub → {HF_USER}/tokenizer-{LANG}-en

Security Note: 
`trust_remote_code` is explicitly set to False. The FineWeb datasets 
use the standard Parquet format and do not require loading scripts.
"""

import os
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
LANG_CONFIGS: dict = {

    # --------------------------------------------------
    # LATIN SCRIPT — WESTERN EUROPEAN
    # --------------------------------------------------
    "pl": {
        "name"      : "Polish",
        "fw2_name"  : "pol_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Heavily inflected (7 grammatical cases, grammatical gender).",
            "Diacritics ą ć ę ł ń ó ś ź ż are phonemically distinct — never strip.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Compound-heavy (stofzuigerzak, verantwoordelijkheid); BPE segments naturally.",
            "Diacritics rare (é ë for disambiguation only); NFKC sufficient.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Tildes (á é í ó ú ñ ü) are phonemically meaningful — never strip.",
            "Inverted punctuation (¿ ¡) handled naturally by ByteLevel.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Accents à â ç é è ê ë î ï ô ù û ü ÿ are meaningful.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Umlauts ä ö ü and ß are distinct phonemes — never strip or fold.",
            "Long noun compounds tokenised into meaningful sub-words by BPE.",
        ],
        "test_cases": {
            "DE_Simple"  : "Hallo, wie geht es Ihnen?",
            "DE_Umlauts" : "Über die Größe des Brötchens lässt sich streiten.",
            "DE_Compound": "Das Donaudampfschifffahrtsgesellschaftskapitänspatent.",
            "DE_Mixed"   : "Wir nutzen maschinelles Lernen für NLP-Aufgaben.",
        },
    },

    "it": {
        "name"      : "Italian",
        "fw2_name"  : "ita_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Accents à è é ì í ò ó ù are meaningful, mainly on final syllables.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Ring å and umlauts ä ö are distinct phonemes — never strip.",
            "Compound-heavy (like German); BPE handles via sub-word merges.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Accents à è é í ï ó ò ú ü are meaningful.",
            "Geminated l·l (U+00B7 middle dot) is normalised correctly by NFKC.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 1,
        "notes": [
            "Language isolate; highly agglutinative with complex suffix stacking.",
            "BPE naturally learns frequent Basque suffixes (-rekin, -ean, -ko, -ren).",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Highly agglutinative; one word can encode an entire clause.",
            "Dotted/dotless I (ı/İ vs i/I) preserved by NFKC.",
        ],
        "test_cases": {
            "TR_Simple"       : "Merhaba, nasılsınız?",
            "TR_Agglutinative": "Çekoslovakyalılaştıramadıklarımızdanmışsınız.",
            "TR_DottedI"      : "İstanbul ve Izmir büyük şehirlerdir.",
            "TR_Mixed"        : "Python kullanarak doğal dil işleme yapıyoruz.",
        },
    },

    "id": {
        "name"      : "Indonesian",
        "fw2_name"  : "ind_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Agglutinative; productive prefix/suffix morphology (me-, ber-, -kan, -an).",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 1,
        "notes": [
            "Agglutinative; focus-marking affixes (mag-, -in, -an, -um-) are central.",
            "Heavy English code-switching in web data.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Greek script (U+0370–U+03FF); 2 bytes in UTF-8.",
            "NFKC collapses polytonic precomposed forms → monotonic equivalents.",
        ],
        "test_cases": {
            "EL_Simple"   : "Γεια σου, πώς είσαι;",
            "EL_Polytonic": "Ἐν ἀρχῇ ἦν ὁ λόγος.",
            "EL_Mixed"    : "Χρησιμοποιούμε Python για μηχανική μάθηση.",
        },
    },

    "ru": {
        "name"      : "Russian",
        "fw2_name"  : "rus_Cyrl",
        "script"    : "Cyrillic",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Cyrillic script (U+0400–U+04FF); 2 bytes in UTF-8.",
            "NFKC handles rare Cyrillic compatibility characters.",
        ],
        "test_cases": {
            "RU_Simple"    : "Привет, как дела?",
            "RU_Morphology": "Непротивостоянию властям посвятили отдельную главу.",
            "RU_Mixed"     : "Мы используем Python для машинного обучения.",
        },
    },

    # --------------------------------------------------
    # NON-LATIN SCRIPTS — EAST ASIAN
    # --------------------------------------------------
    "zh": {
        "name"            : "Chinese",
        "fw2_name"        : "cmn_Hani",
        "script"          : "Han",
        "model"           : "bpe",
        "normalizer"      : "nfc",
        "add_prefix_space": True,
        "trim_offsets"    : True,
        "clean_up_spaces" : True,
        "min_freq"        : 2,
        "notes": [
            "Han script. NFC chosen over NFKC to avoid collapsing CJK variant characters.",
            "No word boundaries — BPE learns character n-gram merges.",
        ],
        "test_cases": {
            "ZH_Simple"  : "你好，我饿了。",
            "ZH_Long"    : "人工智能技术正在改变我们的生活方式。",
            "ZH_Mixed"   : "Hello 你好!",
            "ZH_Numbers" : "这本书的价格是299元。",
        },
    },

    # --------------------------------------------------
    # NON-LATIN SCRIPTS — MIDDLE EAST / SOUTH ASIA
    # --------------------------------------------------
    "ar": {
        "name"            : "Arabic",
        "fw2_name"        : "arb_Arab",
        "script"          : "Arabic",
        "model"           : "unigram",
        "normalizer"      : "nfkc+tatweel",
        "clean_up_spaces" : False,
        "shrinking_factor": 0.75,
        "max_piece_length": 20,
        "n_sub_iterations": 2,
        "notes": [
            "Modern Standard Arabic. Uses Unigram for probabilistic segmentation.",
            "NFKC + strip tatweel (U+0640). Alef-hamza and diacritics preserved.",
            "UnicodeScripts pre-tokenizer for clean Arabic-English code-switching.",
        ],
        "test_cases": {
            "AR_Simple"     : "مرحباً، أنا جائع.",
            "AR_Clitics"    : "وسيكتبونها في التقرير.",
            "AR_Diacritized": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            "AR_Tatweel"    : "كتـــاب",
            "AR_Numerals"   : "السعر هو ١٢٬٠٠٠ ريال.",
            "AR_EN_Mixed"   : "نستخدم Python لتطوير نماذج الذكاء الاصطناعي.",
        },
    },

    "fa": {
        "name"      : "Persian",
        "fw2_name"  : "pes_Arab",
        "script"    : "Arabic",
        "model"     : "bpe",
        "normalizer": "nfkc+tatweel",
        "min_freq"  : 2,
        "notes": [
            "Arabic script + Persian-specific chars (پ چ ژ گ).",
            "PRESERVE ZWNJ (U+200C) for word-internal morpheme boundaries.",
        ],
        "test_cases": {
            "FA_Simple" : "سلام، حال شما چطور است؟",
            "FA_ZWNJ"   : "می‌روم به مدرسه هر روز صبح.",
            "FA_Tatweel": "کتـــاب",
            "FA_Mixed"  : "ما از Python برای یادگیری ماشین استفاده می‌کنیم.",
        },
    },

    "hi": {
        "name"      : "Hindi",
        "fw2_name"  : "hin_Deva",
        "script"    : "Devanagari",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Devanagari script (U+0900–U+097F). Chars are 3 bytes in UTF-8.",
            "NFKC normalises composed/decomposed forms of matras and virama.",
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
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": [
            "Tamil script (U+0B80–U+0BFF). Chars are 3 bytes in UTF-8.",
            "Agglutinative; complex suffix stacking handled by BPE.",
        ],
        "test_cases": {
            "TA_Simple"  : "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "TA_Compound": "தமிழ்நாட்டில் பல அழகான கோயில்கள் உள்ளன.",
            "TA_Mixed"   : "நாங்கள் Python ஐ இயந்திர கற்றலுக்கு பயன்படுத்துகிறோம்.",
        },
    },
}

LOWER_RESOURCE_LANGS = {"eu", "tl", "ta"}

# =====================================================
# CLI
# =====================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tokenizer for one of 19 languages.")
    p.add_argument("--lang", required=True, choices=sorted(LANG_CONFIGS), help="ISO-639-1 code.")
    p.add_argument("--hf-user", default="Beetle-Data", help="HF username/org.")
    p.add_argument("--vocab-size", type=int, default=50_000, help="Vocab size.")
    p.add_argument("--sentences", type=int, default=2_000_000, help="Total sentences.")
    p.add_argument("--no-push", action="store_true", help="Skip HF Hub push.")
    p.add_argument("--out-dir", default=None, help="Local output dir.")
    return p.parse_args()

# =====================================================
# 1. DATA GENERATOR (NO REMOTE CODE)
# =====================================================
def get_training_corpus(cfg: dict, n_sentences: int):
    # Standard Parquet loading, no remote code needed.
    lang_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name=cfg["fw2_name"],
        split="train",
        streaming=True,
        trust_remote_code=False 
    )
    en_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
        trust_remote_code=False
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
    if norm_type == "nfc":
        return normalizers.NFC()
    elif norm_type == "nfkc":
        return normalizers.NFKC()
    elif norm_type == "nfkc+tatweel":
        return normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace(pattern="\u0640", content=""),
        ])
    else:
        raise ValueError(f"Unknown normalizer type: {norm_type!r}")

# =====================================================
# 3. BUILD TOKENIZER
# =====================================================
def build_tokenizer(cfg: dict) -> Tokenizer:
    model_type = cfg.get("model", "bpe")

    if model_type == "bpe":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer     = build_normalizer(cfg["normalizer"])
        tokenizer.pre_tokenizer  = pre_tokenizers.ByteLevel(
            add_prefix_space=cfg.get("add_prefix_space", False)
        )
        tokenizer.post_processor = processors.ByteLevel(
            trim_offsets=cfg.get("trim_offsets", False)
        )
        tokenizer.decoder        = decoders.ByteLevel()

    elif model_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer    = build_normalizer(cfg["normalizer"])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True),
        ])
        tokenizer.decoder = decoders.Metaspace()
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    return tokenizer

# =====================================================
# 4. BUILD TRAINER
# =====================================================
def build_trainer(cfg: dict, vocab_size: int):
    model_type = cfg.get("model", "bpe")

    if model_type == "bpe":
        return trainers.BpeTrainer(
            vocab_size       = vocab_size,
            min_frequency    = cfg["min_freq"],
            special_tokens   = ["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        )
    elif model_type == "unigram":
        return trainers.UnigramTrainer(
            vocab_size       = vocab_size,
            special_tokens   = ["<unk>", "<s>", "</s>", "<pad>"],
            unk_token        = "<unk>",
            shrinking_factor = cfg.get("shrinking_factor", 0.75),
            max_piece_length = cfg.get("max_piece_length", 16),
            n_sub_iterations = cfg.get("n_sub_iterations", 2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

# =====================================================
# 5. TRAIN & PUSH
# =====================================================
def train_and_push(lang, hf_user, vocab_size, n_sentences, out_dir, push):
    cfg = LANG_CONFIGS[lang]
    repo_id = f"{hf_user}/tokenizer-{lang}-en"
    hf_token = os.environ.get("HF_TOKEN")
    model_type = cfg.get("model", "bpe")

    print(f"\n🚀 Training {model_type.upper()} tokenizer for {cfg['name']}...")
    
    tokenizer = build_tokenizer(cfg)
    trainer   = build_trainer(cfg, vocab_size)

    tokenizer.train_from_iterator(
        get_training_corpus(cfg, n_sentences),
        trainer=trainer,
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token = "<s>",
        eos_token = "</s>",
        pad_token = "<pad>",
        unk_token = "<unk>",
        clean_up_tokenization_spaces = cfg.get("clean_up_spaces", False),
    )

    hf_tokenizer.save_pretrained(out_dir)
    print(f"✅ Saved to {out_dir}")

    if push:
        if not hf_token:
            print("⚠️ HF_TOKEN not set — skipping push.")
            return
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"✅ Pushed to https://huggingface.co/{repo_id}")

# =====================================================
# 6. BENCHMARK
# =====================================================
def run_benchmark(lang: str, out_dir: Path):
    cfg = LANG_CONFIGS[lang]
    tk  = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=False)

    print(f"\n🧪 Benchmark — {cfg['name']} ({lang})\n" + "─" * 60)
    shared_cases = {
        "EN_Simple": "The quick brown fox jumps over the lazy dog.",
        "Emoji"    : "Learning is fun! 🚀🔥",
    }
    all_cases = {**cfg["test_cases"], **shared_cases}
    strips_tatweel = cfg["normalizer"] == "nfkc+tatweel"

    for name, text in all_cases.items():
        tokens  = tk.encode(text)
        decoded = tk.decode(tokens)
        
        expected = text.replace("\u0640", "") if strips_tatweel else text
        ok = decoded.strip() == expected.strip()
        print(f"  [{name}] {'✅' if ok else '❌'} ({len(tokens)} tokens)")

# =====================================================
# 7. VOCAB INSPECTION
# =====================================================
def inspect_vocab(lang: str, out_dir: Path, n: int = 50):
    tk = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=False)
    vocab = tk.get_vocab()
    print(f"\n🔍 Vocab sample (size: {len(vocab):,})\n" + "─" * 40)
    for piece, idx in sorted(vocab.items(), key=lambda x: x[1])[:n]:
        print(f"  {idx:6d}  {piece!r}")

# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()
    lang = args.lang
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"tokenizer-{lang}-local")
    out_dir.mkdir(exist_ok=True)

    train_and_push(
        lang        = lang,
        hf_user     = args.hf_user,
        vocab_size  = args.vocab_size,
        n_sentences = args.sentences,
        out_dir     = out_dir,
        push        = not args.no_push,
    )
    run_benchmark(lang, out_dir)
    inspect_vocab(lang, out_dir)

if __name__ == "__main__":
    main()
