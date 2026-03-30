#!/usr/bin/env python3
"""
Japanese-English Tokenizer Trainer with tqdm progress
=====================================================
Tracks streaming progress while training tokenizer on
Japanese (FineWeb-2) + English (FineWeb-edu)
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from transformers import PreTrainedTokenizerFast
from huggingface_hub import create_repo
from tqdm import tqdm


# =====================================================
# CONFIG
# =====================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train a Japanese-English tokenizer.")
    p.add_argument("--hf-user", default="Beetle-Data", help="HF username / org.")
    p.add_argument("--vocab-size", type=int, default=50000, help="Vocab size.")
    p.add_argument("--sentences", type=int, default=2000000, help="Total training sentences.")
    p.add_argument("--no-push", action="store_true", help="Skip HF Hub push.")
    return p.parse_args()


# =====================================================
# 1. DATA GENERATOR
# =====================================================
def get_training_corpus(n_sentences):
    """
    Yields Japanese (FineWeb-2) and English (FineWeb-edu) sentences.
    Ensures exactly n_sentences (or slightly less if streams end).
    """
    ja_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="jpn_Jpan",
        split="train",
        streaming=True
    )
    en_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True
    )

    ja_iter = iter(ja_ds)
    en_iter = iter(en_ds)

    count = 0
    half = n_sentences // 2

    for _ in range(half):
        try:
            ja_text = next(ja_iter)["text"].replace("\n", " ")
            yield ja_text
            count += 1

            if count >= n_sentences:
                break

            en_text = next(en_iter)["text"].replace("\n", " ")
            yield en_text
            count += 1

            if count >= n_sentences:
                break

        except StopIteration:
            break


# =====================================================
# 2. BUILD TOKENIZER
# =====================================================
def build_tokenizer():
    tokenizer = Tokenizer(models.Unigram())

    tokenizer.normalizer = normalizers.NFC()

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Metaspace(replacement="▁")  # FIXED
    ])

    tokenizer.decoder = decoders.Metaspace(replacement="▁")  # FIXED

    return tokenizer


# =====================================================
# 3. BUILD TRAINER
# =====================================================
def build_trainer(vocab_size):
    return trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        unk_token="<unk>",
        shrinking_factor=0.75,
        max_piece_length=16,
        n_sub_iterations=2
    )


# =====================================================
# 4. TRAIN & PUSH
# =====================================================
def train_and_push():
    args = parse_args()

    out_dir = Path("tokenizer-ja-en-local")
    out_dir.mkdir(exist_ok=True)

    repo_id = f"{args.hf_user}/tokenizer-ja-en"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"\n🚀 Training Japanese-English Unigram tokenizer")
    print(f"   • Vocab size: {args.vocab_size:,}")
    print(f"   • Sentences:  {args.sentences:,}\n")

    tokenizer = build_tokenizer()
    trainer = build_trainer(args.vocab_size)

    # tqdm wrapper
    corpus_iterator = tqdm(
        get_training_corpus(args.sentences),
        total=args.sentences,
        desc="Streaming sentences",
        unit="sent"
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    # Wrap in HF tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=True
    )

    hf_tokenizer.save_pretrained(out_dir)
    print(f"\n✅ Tokenizer saved to {out_dir}")

    # =====================================================
    # TEST
    # =====================================================
    print("\n🧪 Running decode test...")
    test_str = "こんにちは、世界！ Today is a good day."

    tokens = hf_tokenizer.encode(test_str)
    decoded = hf_tokenizer.decode(tokens)

    print(f"Input:    {test_str}")
    print(f"Tokens:   {tokens[:20]}...")
    print(f"Decoded:  {decoded}")

    # =====================================================
    # PUSH TO HUB
    # =====================================================
    if args.no_push:
        print("\n⏭️ Skipping push (--no-push enabled)")
        return

    if not hf_token:
        print("\n⚠️ HF_TOKEN not set — skipping push.")
        return

    print(f"\n📤 Pushing to Hugging Face Hub: {repo_id}")

    create_repo(repo_id, exist_ok=True, token=hf_token)
    hf_tokenizer.push_to_hub(repo_id, token=hf_token)

    print(f"✅ Pushed to https://huggingface.co/{repo_id}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    train_and_push()
