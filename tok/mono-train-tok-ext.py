#!/usr/bin/env python3
"""
Monolingual Tokenizer Trainer — Extended Languages (ar, ja, ko)
================================================================

Companion to tok/mono-train-tok.py (which covers en, nl, de, zh).
This script adds monolingual tokenizers for Arabic, Japanese, Korean.

Why a separate script?
  - mono-train-tok.py uses ByteLevel BPE (good for whitespace-rich Latin/CJK
    where word-shape matters). For ar/ja/ko, Unigram with Metaspace +
    UnicodeScripts pre-tokenization gives substantially better compression
    and morpheme alignment (ar is morphologically rich; ja/ko have no
    inter-word whitespace at all — ByteLevel produces noisy byte-pair merges).
  - The patterns here mirror tok/ar-en-tok.py and tok/ja-en-tok.py which
    are the proven recipes for these languages in the bilingual setting.

Usage:
    python tok/mono-train-tok-ext.py --lang ar --hf-user Beetle-Data \\
        --vocab-size 50000 --sentences 2000000 --repo-suffix mono

Output:
    Local:   tokenizer-{lang}-{suffix}-local/
    Hub:     {hf_user}/tokenizer-{lang}-{suffix}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
    from tokenizers import (
        Tokenizer, models, trainers, pre_tokenizers,
        decoders, normalizers,
    )
    from transformers import PreTrainedTokenizerFast
    from huggingface_hub import create_repo
except ImportError as e:
    print(f"Error: missing dependency — {e}", file=sys.stderr)
    print("Run: pip install datasets tokenizers transformers tqdm huggingface_hub",
          file=sys.stderr)
    sys.exit(1)


# =====================================================
# LANGUAGE CONFIGS
# =====================================================
# Each config drives:
#   - which FineWeb-2 subset to stream
#   - which model class (Unigram / BPE)
#   - which normalizer / pre-tokenizer / decoder pipeline
#   - trainer hyperparameters (max_piece_length, shrinking_factor)
LANG_CONFIGS = {
    "ar": {
        "name": "Arabic",
        "fw2_name": "arb_Arab",         # Modern Standard Arabic
        "model": "unigram",
        "norm": "nfkc_tatweel",          # NFKC + strip kashida (U+0640)
        "pretok": "scripts_metaspace",   # UnicodeScripts + Metaspace
        "max_piece_length": 20,          # arabic clitic-laden words can be long
        "shrinking_factor": 0.75,
        "min_freq": 2,
    },
    "ja": {
        "name": "Japanese",
        "fw2_name": "jpn_Jpan",
        "model": "unigram",
        "norm": "nfkc",
        "pretok": "scripts_metaspace",   # Japanese has no whitespace; UnicodeScripts
                                         # splits Han/Hiragana/Katakana/Latin boundaries
        "max_piece_length": 16,
        "shrinking_factor": 0.75,
        "min_freq": 2,
    },
    "ko": {
        "name": "Korean",
        "fw2_name": "kor_Hang",
        "model": "unigram",
        "norm": "nfkc",
        "pretok": "scripts_metaspace",   # Korean has whitespace but Hangul jamo
                                         # composition benefits from script-aware splits
        "max_piece_length": 16,
        "shrinking_factor": 0.75,
        "min_freq": 2,
    },
}


# =====================================================
# DATA STREAMER
# =====================================================
def get_training_corpus(cfg: dict, n_sentences: int):
    print(f"Initializing stream ({cfg['name']}, {n_sentences:,} sentences)...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name=cfg["fw2_name"],
            split="train",
            streaming=True,
        )
        it = iter(ds)
        for i in range(n_sentences):
            try:
                yield next(it)["text"].replace("\n", " ")
            except StopIteration:
                print(f"\nReached end of dataset early at sentence {i}")
                break
    except Exception as e:
        print(f"\nDataset connection error: {e}", file=sys.stderr)
        sys.exit(1)


# =====================================================
# BUILD TOKENIZER
# =====================================================
def _build_normalizer(cfg: dict):
    if cfg["norm"] == "nfkc_tatweel":
        return normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace(pattern="\u0640", content=""),  # strip kashida
        ])
    if cfg["norm"] == "nfkc":
        return normalizers.NFKC()
    if cfg["norm"] == "nfc":
        return normalizers.NFC()
    raise ValueError(f"Unknown norm: {cfg['norm']}")


def _build_pretokenizer(cfg: dict):
    if cfg["pretok"] == "scripts_metaspace":
        return pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True),
        ])
    raise ValueError(f"Unknown pretok: {cfg['pretok']}")


def build_tokenizer(cfg: dict) -> Tokenizer:
    if cfg["model"] == "unigram":
        tok = Tokenizer(models.Unigram())
    elif cfg["model"] == "bpe":
        tok = Tokenizer(models.BPE())
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    tok.normalizer = _build_normalizer(cfg)
    tok.pre_tokenizer = _build_pretokenizer(cfg)
    tok.decoder = decoders.Metaspace()
    return tok


def build_trainer(cfg: dict, vocab_size: int):
    if cfg["model"] == "unigram":
        return trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            unk_token="<unk>",
            shrinking_factor=cfg.get("shrinking_factor", 0.75),
            max_piece_length=cfg.get("max_piece_length", 16),
            n_sub_iterations=2,
        )
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=cfg.get("min_freq", 2),
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )


# =====================================================
# MAIN PIPELINE
# =====================================================
def train_and_push(
    lang: str,
    hf_user: str = "Beetle-Data",
    vocab_size: int = 50_000,
    n_sentences: int = 2_000_000,
    repo_suffix: str = "mono",
) -> "PreTrainedTokenizerFast":
    if lang not in LANG_CONFIGS:
        raise ValueError(
            f"Unknown language: {lang}. Available: {sorted(LANG_CONFIGS)}"
        )
    cfg = LANG_CONFIGS[lang]
    out_dir = Path(f"tokenizer-{lang}-{repo_suffix}-local")
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- Preparing {cfg['name']} ({lang}) monolingual pipeline ---")
    print(f"  model={cfg['model']}  vocab_size={vocab_size:,}  "
          f"sentences={n_sentences:,}")
    sys.stdout.flush()

    tokenizer = build_tokenizer(cfg)
    trainer = build_trainer(cfg, vocab_size)

    progress_iterator = tqdm(
        get_training_corpus(cfg, n_sentences),
        total=n_sentences,
        desc=f"Streaming & Training {lang}",
        unit=" sentences",
        colour="cyan",
    )

    t0 = time.perf_counter()
    tokenizer.train_from_iterator(progress_iterator, trainer=trainer)
    elapsed = time.perf_counter() - t0
    print(f"\nTraining finished in {elapsed:,.1f}s. "
          f"Final vocab size: {tokenizer.get_vocab_size():,}")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.save_pretrained(out_dir)
    print(f"Local files saved to: {out_dir}")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        repo_id = f"{hf_user}/tokenizer-{lang}-{repo_suffix}"
        print(f"Pushing to Hub: {repo_id}...")
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"Pushed: {repo_id}")
    else:
        print("Skipping Hub push (HF_TOKEN not set).")

    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a monolingual tokenizer for ar/ja/ko."
    )
    parser.add_argument("--lang", required=True,
                        choices=sorted(LANG_CONFIGS),
                        help="Language code: ar, ja, or ko")
    parser.add_argument("--hf-user", default="Beetle-Data")
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--sentences", type=int, default=2_000_000)
    parser.add_argument("--repo-suffix", default="mono")
    args = parser.parse_args()

    train_and_push(
        lang=args.lang,
        hf_user=args.hf_user,
        vocab_size=args.vocab_size,
        n_sentences=args.sentences,
        repo_suffix=args.repo_suffix,
    )


if __name__ == "__main__":
    main()
