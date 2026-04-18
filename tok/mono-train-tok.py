#!/usr/bin/env python3
"""
Monolingual Tokenizer Trainer (BPE)
====================================
Features:
- Streaming FineWeb-2 / FineWeb-edu (Zero disk usage)
- Language-specific normalization (NFC / NFKC)
- ByteLevel BPE pre-tokenization
- Real-time progress tracking via tqdm
- Pushes to HuggingFace Hub as Beetle-Data/tokenizer-{lang}-mono

Supports: en, nl, de, zh
Default: 2,000,000 sentences (matches total data volume of the bilingual tokenizers).
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
    from tokenizers import (
        Tokenizer, models, trainers, pre_tokenizers,
        decoders, normalizers, processors
    )
    from transformers import PreTrainedTokenizerFast
    from huggingface_hub import create_repo
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Run: pip install datasets tokenizers transformers tqdm huggingface_hub")
    sys.exit(1)

# =====================================================
# 1. LANGUAGE CONFIGS
# =====================================================
# source: "fineweb-edu" uses HuggingFaceFW/fineweb-edu (English only).
# source: "fineweb-2"   uses HuggingFaceFW/fineweb-2  with a specific `name`.
LANG_CONFIGS = {
    "en": {
        "name": "English",
        "source": "fineweb-edu",
        "norm": "nfkc",
        "min_freq": 2,
    },
    "nl": {
        "name": "Dutch",
        "source": "fineweb-2",
        "fw2_name": "nld_Latn",
        "norm": "nfkc",
        "min_freq": 2,
    },
    "de": {
        "name": "German",
        "source": "fineweb-2",
        "fw2_name": "deu_Latn",
        "norm": "nfkc",
        "min_freq": 2,
    },
    "zh": {
        "name": "Chinese",
        "source": "fineweb-2",
        "fw2_name": "cmn_Hani",
        "norm": "nfc",
        "add_prefix_space": True,
        "trim_offsets": True,
        "clean_up_spaces": True,
        "min_freq": 2,
    },
}


# =====================================================
# 2. DATA STREAMER
# =====================================================
def get_training_corpus(cfg, n_sentences):
    """Yield text from a single-language stream."""
    print(f"Initializing stream ({cfg['name']}, {n_sentences:,} sentences)...")

    try:
        if cfg["source"] == "fineweb-edu":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        else:  # fineweb-2
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
        print(f"\nDataset connection error: {e}")
        sys.exit(1)


# =====================================================
# 3. BUILD COMPONENTS
# =====================================================
def build_tokenizer(cfg):
    if cfg["norm"] == "nfc":
        norm = normalizers.NFC()
    else:  # nfkc
        norm = normalizers.NFKC()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = norm
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=cfg.get("add_prefix_space", False)
    )
    tokenizer.post_processor = processors.ByteLevel(
        trim_offsets=cfg.get("trim_offsets", False)
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def build_trainer(cfg, vocab_size):
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=cfg.get("min_freq", 2),
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )


# =====================================================
# 4. MAIN PIPELINE
# =====================================================
def train_and_push(
    lang: str,
    hf_user: str = "Beetle-Data",
    vocab_size: int = 50_000,
    n_sentences: int = 2_000_000,
    repo_suffix: str = "mono",
) -> "PreTrainedTokenizerFast":
    """Train a monolingual BPE tokenizer and optionally push to HF Hub.

    Repo ID: {hf_user}/tokenizer-{lang}-{repo_suffix}
    """
    if lang not in LANG_CONFIGS:
        raise ValueError(f"Unknown language: {lang}. Available: {sorted(LANG_CONFIGS)}")

    cfg = LANG_CONFIGS[lang]
    out_dir = Path(f"tokenizer-{lang}-{repo_suffix}-local")
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- Preparing {cfg['name']} ({lang}) monolingual pipeline ---")
    tokenizer = build_tokenizer(cfg)
    trainer = build_trainer(cfg, vocab_size)

    print("Training BPE model...")
    progress_iterator = tqdm(
        get_training_corpus(cfg, n_sentences),
        total=n_sentences,
        desc="Streaming & Training",
        unit=" sentences",
        colour="cyan",
    )
    tokenizer.train_from_iterator(progress_iterator, trainer=trainer)

    print("\nTraining finished. Saving files...")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=cfg.get("clean_up_spaces", False),
    )

    hf_tokenizer.save_pretrained(out_dir)
    print(f"Local files saved to: {out_dir}")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        repo_id = f"{hf_user}/tokenizer-{lang}-{repo_suffix}"
        print(f"Pushing to Hub: {repo_id}...")
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"Pushed successfully: {repo_id}")
    else:
        print("Skipping Hub push (HF_TOKEN not found in environment).")

    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a monolingual BPE tokenizer.")
    parser.add_argument("--lang", required=True, choices=sorted(LANG_CONFIGS),
                        help="Language code: en, nl, de, zh")
    parser.add_argument("--hf-user", default="Beetle-Data")
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--sentences", type=int, default=2_000_000,
                        help="Total sentences to stream (default 2M).")
    parser.add_argument("--repo-suffix", default="mono",
                        help="Repo naming: tokenizer-{lang}-{suffix} (default 'mono').")
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
