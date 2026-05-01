#!/usr/bin/env python3
"""
4-way Multilingual Unigram Tokenizer: English + Arabic + Japanese + Korean
==========================================================================
Model:    Unigram (SentencePiece-style)
Data:     FineWeb-edu (en) + FineWeb-2 (arb_Arab + jpn_Jpan + kor_Hang)
Sampling: balanced, 25% per language
Target:   HuggingFace Hub → {HF_USER}/tokenizer-en-ar-ja-ko-multi

Why Unigram (not ByteLevel BPE)?
  - Three of the four languages (ar/ja/ko) have either no inter-word
    whitespace (ja) or morphologically rich/agglutinative structure
    (ar/ko) that ByteLevel BPE handles poorly.
  - With UnicodeScripts pre-tokenization, script boundaries (Latin / Arabic /
    Han / Hiragana / Katakana / Hangul) are kept clean, and Metaspace gives
    lossless round-trip including English whitespace.
  - This avoids the well-known "ByteLevel-on-CJK" pathology where the Rust
    BPE trainer's merge-table finalization explodes on long whitespace-free
    multi-byte runs.

Usage:
    python tok/multi-en-ar-ja-ko-tok.py \\
        [--hf-user Beetle-Data] [--vocab-size 50000] \\
        [--sentences 2000000] [--no-push]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    sys.exit(1)


HF_USER_DEFAULT = "Beetle-Data"
TAG = "en-ar-ja-ko-multi"


def get_training_corpus(n_sentences: int):
    """Round-robin over en/ar/ja/ko streams, 25% each."""
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu",
                         split="train", streaming=True)
    ar_ds = load_dataset("HuggingFaceFW/fineweb-2",
                         name="arb_Arab", split="train", streaming=True)
    ja_ds = load_dataset("HuggingFaceFW/fineweb-2",
                         name="jpn_Jpan", split="train", streaming=True)
    ko_ds = load_dataset("HuggingFaceFW/fineweb-2",
                         name="kor_Hang", split="train", streaming=True)

    streams = [iter(en_ds), iter(ar_ds), iter(ja_ds), iter(ko_ds)]
    per_stream = n_sentences // 4

    for _ in range(per_stream):
        for it in streams:
            try:
                yield next(it)["text"].replace("\n", " ")
            except StopIteration:
                # If one stream ends early, continue with the others; the
                # downstream sampler will fall short of the target but still
                # produce a usable tokenizer.
                pass


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        # Strip Arabic kashida (U+0640) — purely cosmetic, never meaningful
        normalizers.Replace(pattern="\u0640", content=""),
    ])
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True),
    ])
    tok.decoder = decoders.Metaspace()
    return tok


def train_and_push(hf_user: str, vocab_size: int, sentences: int,
                   no_push: bool):
    repo_id = f"{hf_user}/tokenizer-{TAG}"
    out_dir = Path(f"tokenizer-{TAG}-local")
    out_dir.mkdir(exist_ok=True)

    print(f"🚀 Training 4-way Unigram tokenizer (vocab={vocab_size:,}) "
          f"→ {repo_id} ({sentences:,} sentences, balanced 25%/lang)…")

    tokenizer = build_tokenizer()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        unk_token="<unk>",
        shrinking_factor=0.75,
        # Arabic clitic-laden words can be long; 20 is a safe upper bound.
        max_piece_length=20,
        n_sub_iterations=2,
    )

    print("📚 Streaming en + ar + ja + ko (FineWeb-edu + FineWeb-2)…")
    tokenizer.train_from_iterator(get_training_corpus(sentences),
                                  trainer=trainer)
    print(f"✅ Final vocab: {tokenizer.get_vocab_size():,}")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.save_pretrained(out_dir)
    print(f"💾 Local: {out_dir}")

    hf_token = os.environ.get("HF_TOKEN")
    if no_push or not hf_token:
        if not hf_token:
            print("Skipping Hub push (HF_TOKEN not set).")
        return

    print(f"⬆️  Pushing to Hub: {repo_id}…")
    create_repo(repo_id, exist_ok=True, token=hf_token)
    hf_tokenizer.push_to_hub(repo_id, token=hf_token)
    print(f"✅ Pushed: {repo_id}")


def main():
    p = argparse.ArgumentParser(
        description="Train 4-way Unigram tokenizer (en+ar+ja+ko)."
    )
    p.add_argument("--hf-user", default=HF_USER_DEFAULT)
    p.add_argument("--vocab-size", type=int, default=50_000)
    p.add_argument("--sentences", type=int, default=2_000_000)
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()
    train_and_push(args.hf_user, args.vocab_size, args.sentences, args.no_push)


if __name__ == "__main__":
    main()
