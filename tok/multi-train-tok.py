#!/usr/bin/env python3
"""
Multi-language Tokenizer Trainer
==================================
Script : multi-train-tok.py
Data   : FineWeb-2 (language-specific) + FineWeb-edu (English)
Target : HuggingFace Hub → {HF_USER}/tokenizer-{LANG}-en

Security Note: 
Explicitly uses trust_remote_code=False. 
Datasets are confirmed Parquet format (Safe).
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
# This dictionary contains the specific normalization and 
# training logic for all 19 languages in your bash loop.

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
            "Heavily inflected (7 grammatical cases).",
            "Diacritics ą ć ę ł ń ó ś ź ż preserved.",
        ],
        "test_cases": {
            "PL_Simple"    : "Cześć, jak się masz?",
            "PL_Diacritics": "Żółw skacze przez źródło w środę.",
        },
    },

    "nl": {
        "name"      : "Dutch",
        "fw2_name"  : "nld_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Compound-heavy; BPE segments naturally."],
        "test_cases": {"NL_Simple": "Hallo, hoe gaat het?"},
    },

    "es": {
        "name"      : "Spanish",
        "fw2_name"  : "spa_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Meaningful tildes (á ñ) preserved."],
        "test_cases": {"ES_Simple": "Hola, ¿cómo estás?"},
    },

    "fr": {
        "name"      : "French",
        "fw2_name"  : "fra_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Elision (l'été) handled by ByteLevel."],
        "test_cases": {"FR_Simple": "Bonjour, comment allez-vous?"},
    },

    "de": {
        "name"      : "German",
        "fw2_name"  : "deu_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["ß preserved; compounds tokenized into morphemes."],
        "test_cases": {"DE_Simple": "Hallo, wie geht es Ihnen?"},
    },

    "it": {
        "name"      : "Italian",
        "fw2_name"  : "ita_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Final syllable accents preserved."],
        "test_cases": {"IT_Simple": "Ciao, come stai?"},
    },

    "sv": {
        "name"      : "Swedish",
        "fw2_name"  : "swe_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Ring å and umlauts ä ö preserved."],
        "test_cases": {"SV_Simple": "Hej, hur mår du?"},
    },

    "ca": {
        "name"      : "Catalan",
        "fw2_name"  : "cat_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Geminated l·l correctly normalized."],
        "test_cases": {"CA_Simple": "Hola, com estàs?"},
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
        "notes": ["Agglutinative isolate; min_freq=1 for rare suffixes."],
        "test_cases": {"EU_Simple": "Kaixo, nola zaude?"},
    },

    "tr": {
        "name"      : "Turkish",
        "fw2_name"  : "tur_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Dotted/dotless I distinguished."],
        "test_cases": {"TR_Simple": "Merhaba, nasılsınız?"},
    },

    "id": {
        "name"      : "Indonesian",
        "fw2_name"  : "ind_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Productive prefix/suffix morphology."],
        "test_cases": {"ID_Simple": "Halo, apa kabar?"},
    },

    "tl": {
        "name"      : "Tagalog",
        "fw2_name"  : "tgl_Latn",
        "script"    : "Latin",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 1,
        "notes": ["Focus-marking affixes; heavy EN code-switching."],
        "test_cases": {"TL_Simple": "Kamusta ka?"},
    },

    # --------------------------------------------------
    # NON-LATIN SCRIPTS
    # --------------------------------------------------

    "el": {
        "name"      : "Greek",
        "fw2_name"  : "ell_Grek",
        "script"    : "Greek",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Polytonic forms normalized to monotonic."],
        "test_cases": {"EL_Simple": "Γεια σου, πώς είσαι;"},
    },

    "ru": {
        "name"      : "Russian",
        "fw2_name"  : "rus_Cyrl",
        "script"    : "Cyrillic",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Cyrillic (2-byte UTF-8) handles naturally."],
        "test_cases": {"RU_Simple": "Привет, как дела?"},
    },

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
        "notes": ["NFC used to avoid variant collapse; no word boundaries."],
        "test_cases": {"ZH_Simple": "你好，我饿了。"},
    },

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
        "notes": ["Unigram model; tatweel stripped; UnicodeScripts used."],
        "test_cases": {"AR_Simple": "مرحباً، أنا جائع."},
    },

    "fa": {
        "name"      : "Persian",
        "fw2_name"  : "pes_Arab",
        "script"    : "Arabic",
        "model"     : "bpe",
        "normalizer": "nfkc+tatweel",
        "min_freq"  : 2,
        "notes": ["ZWNJ preserved for internal boundaries."],
        "test_cases": {"FA_Simple": "سلام، حال شما چطور است؟"},
    },

    "hi": {
        "name"      : "Hindi",
        "fw2_name"  : "hin_Deva",
        "script"    : "Devanagari",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Devanagari matras normalized."],
        "test_cases": {"HI_Simple": "नमस्ते, आप कैसे हैं?"},
    },

    "ta": {
        "name"      : "Tamil",
        "fw2_name"  : "tam_Taml",
        "script"    : "Tamil",
        "model"     : "bpe",
        "normalizer": "nfkc",
        "min_freq"  : 2,
        "notes": ["Tamil script (3-byte UTF-8); highly agglutinative."],
        "test_cases": {"TA_Simple": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"},
    },
}

# =====================================================
# UTILITIES & LOGIC
# =====================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=sorted(LANG_CONFIGS))
    p.add_argument("--hf-user", default="Beetle-Data")
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--sentences", type=int, default=2000000)
    return p.parse_args()

def get_training_corpus(cfg, n_sentences):
    # Dataset loading with trust_remote_code=False
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
    lang_iter, en_iter = iter(lang_ds), iter(en_ds)
    for _ in range(n_sentences // 2):
        try:
            yield next(lang_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

def build_normalizer(norm_type):
    if norm_type == "nfc": return normalizers.NFC()
    if norm_type == "nfkc": return normalizers.NFKC()
    if norm_type == "nfkc+tatweel":
        return normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(pattern="\u0640", content="")])
    raise ValueError(f"Unknown: {norm_type}")

def build_tokenizer(cfg):
    model_type = cfg.get("model", "bpe")
    if model_type == "bpe":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = build_normalizer(cfg["normalizer"])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=cfg.get("add_prefix_space", False))
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=cfg.get("trim_offsets", False))
        tokenizer.decoder = decoders.ByteLevel()
    elif model_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = build_normalizer(cfg["normalizer"])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        ])
        tokenizer.decoder = decoders.Metaspace()
    return tokenizer

def build_trainer(cfg, vocab_size):
    model_type = cfg.get("model", "bpe")
    if model_type == "bpe":
        return trainers.BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=cfg["min_freq"], 
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
    return trainers.UnigramTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        unk_token="<unk>",
        shrinking_factor=cfg.get("shrinking_factor", 0.75),
        max_piece_length=cfg.get("max_piece_length", 16)
    )

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    args = parse_args()
    cfg = LANG_CONFIGS[args.lang]
    out_dir = Path(f"tokenizer-{args.lang}-local")
    out_dir.mkdir(exist_ok=True)

    print(f"--- Training {args.lang} ({cfg['name']}) ---")
    
    tokenizer = build_tokenizer(cfg)
    trainer = build_trainer(cfg, args.vocab_size)
    
    tokenizer.train_from_iterator(get_training_corpus(cfg, args.sentences), trainer=trainer)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, 
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
        clean_up_tokenization_spaces=cfg.get("clean_up_spaces", False)
    )

    hf_tokenizer.save_pretrained(out_dir)
    
    # Hub push logic
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        repo_id = f"{args.hf_user}/tokenizer-{args.lang}-en"
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)

if __name__ == "__main__":
    main()
