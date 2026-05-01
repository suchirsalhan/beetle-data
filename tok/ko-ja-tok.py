#!/usr/bin/env python3
"""
Korean + Japanese Bilingual Unigram Tokenizer
=============================================
Model:    Unigram (SentencePiece-style)
Data:     FineWeb-2 (kor_Hang) + FineWeb-2 (jpn_Jpan)
Target:   HuggingFace Hub → {HF_USER}/tokenizer-ko-ja

Why Unigram?
  - Both Korean and Japanese have non-trivial sub-word structure with
    little or no helpful whitespace cues for sub-word merges.
  - ByteLevel BPE on Hangul / Han / Hiragana / Katakana wastes vocab on
    multi-byte UTF-8 sequences; Unigram with UnicodeScripts pre-tokenization
    keeps script boundaries clean and learns morpheme-aligned pieces.
  - The Metaspace pre-tokenizer (▁) gives lossless round-trip even with
    inconsistent whitespace (Japanese has none; Korean has spaces).

Usage:
    python tok/ko-ja-tok.py [--hf-user Beetle-Data] [--vocab-size 50000] \\
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
LANG_PAIR = "ko-ja"


def get_training_corpus(n_sentences: int):
    """Interleave Korean and Japanese 50/50."""
    ko_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="kor_Hang",
        split="train",
        streaming=True,
    )
    ja_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="jpn_Jpan",
        split="train",
        streaming=True,
    )
    ko_iter, ja_iter = iter(ko_ds), iter(ja_ds)
    half = n_sentences // 2
    for _ in range(half):
        try:
            yield next(ko_iter)["text"].replace("\n", " ")
            yield next(ja_iter)["text"].replace("\n", " ")
        except StopIteration:
            break


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),  # split Han / Hiragana / Katakana / Hangul / Latin
        pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True),
    ])
    tok.decoder = decoders.Metaspace()
    return tok


def train_and_push(hf_user: str, vocab_size: int, sentences: int,
                   no_push: bool):
    repo_id = f"{hf_user}/tokenizer-{LANG_PAIR}"
    out_dir = Path("tokenizer-local-temp-ko-ja")
    out_dir.mkdir(exist_ok=True)

    print(f"🚀 Training Unigram tokenizer for ko-ja  (vocab={vocab_size:,}) "
          f"→ {repo_id} ({sentences:,} sentences)…")

    tokenizer = build_tokenizer()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        unk_token="<unk>",
        shrinking_factor=0.75,
        # Korean Hangul syllables + Japanese Han characters can compose into
        # mid-length tokens; 16 covers most morpheme units.
        max_piece_length=16,
        n_sub_iterations=2,
    )

    print("📚 Streaming training data (FineWeb-2 kor_Hang + jpn_Jpan)…")
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
    p = argparse.ArgumentParser(description="Train ko-ja Unigram tokenizer.")
    p.add_argument("--hf-user", default=HF_USER_DEFAULT)
    p.add_argument("--vocab-size", type=int, default=50_000)
    p.add_argument("--sentences", type=int, default=2_000_000)
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()
    train_and_push(args.hf_user, args.vocab_size, args.sentences, args.no_push)


if __name__ == "__main__":
    main()
