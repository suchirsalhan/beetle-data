#!/usr/bin/env python3
"""
Multi-language Tokenizer Trainer (With Progress Tracking)
========================================================
Script : multi-train-tok.py
Data   : FineWeb-2 (Streaming) + FineWeb-edu (Streaming)
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

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
    "pl": {"name": "Polish", "fw2_name": "pol_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "nl": {"name": "Dutch", "fw2_name": "nld_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "es": {"name": "Spanish", "fw2_name": "spa_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "fr": {"name": "French", "fw2_name": "fra_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "de": {"name": "German", "fw2_name": "deu_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "it": {"name": "Italian", "fw2_name": "ita_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "sv": {"name": "Swedish", "fw2_name": "swe_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "ca": {"name": "Catalan", "fw2_name": "cat_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "eu": {"name": "Basque", "fw2_name": "eus_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 1},
    "tr": {"name": "Turkish", "fw2_name": "tur_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "id": {"name": "Indonesian", "fw2_name": "ind_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "tl": {"name": "Tagalog", "fw2_name": "tgl_Latn", "model": "bpe", "normalizer": "nfkc", "min_freq": 1},
    "el": {"name": "Greek", "fw2_name": "ell_Grek", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "ru": {"name": "Russian", "fw2_name": "rus_Cyrl", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "zh": {
        "name": "Chinese", "fw2_name": "cmn_Hani", "model": "bpe", "normalizer": "nfc", 
        "add_prefix_space": True, "trim_offsets": True, "clean_up_spaces": True, "min_freq": 2
    },
    "ar": {
        "name": "Arabic", "fw2_name": "arb_Arab", "model": "unigram", "normalizer": "nfkc+tatweel", 
        "clean_up_spaces": False, "shrinking_factor": 0.75, "max_piece_length": 20
    },
    "fa": {"name": "Persian", "fw2_name": "pes_Arab", "model": "bpe", "normalizer": "nfkc+tatweel", "min_freq": 2},
    "hi": {"name": "Hindi", "fw2_name": "hin_Deva", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
    "ta": {"name": "Tamil", "fw2_name": "tam_Taml", "model": "bpe", "normalizer": "nfkc", "min_freq": 2},
}

# =====================================================
# DATA GENERATOR (STREAMING)
# =====================================================
def get_training_corpus(cfg, n_sentences):
    """Yields text from FineWeb datasets without loading into memory."""
    lang_ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg["fw2_name"], split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    lang_iter, en_iter = iter(lang_ds), iter(en_ds)
    
    for _ in range(n_sentences // 2):
        try:
            # Yield Target Language
            yield next(lang_iter)["text"].replace("\n", " ")
            # Yield English (Bilingual support)
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

# =====================================================
# TOKENIZER COMPONENTS
# =====================================================
def build_tokenizer(cfg):
    model_type = cfg.get("model", "bpe")
    
    # 1. Normalization
    if cfg["normalizer"] == "nfc": 
        norm = normalizers.NFC()
    elif cfg["normalizer"] == "nfkc": 
        norm = normalizers.NFKC()
    else: # nfkc + tatweel for Arabic/Persian
        norm = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(pattern="\u0640", content="")])

    # 2. Model & Pre-tokenization logic
    if model_type == "bpe":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = norm
        # ByteLevel pre-tokenization is standard for BPE to handle all UTF-8 bytes
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=cfg.get("add_prefix_space", False))
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=cfg.get("trim_offsets", False))
        tokenizer.decoder = decoders.ByteLevel()
    else:
        # Unigram (Arabic specific setup)
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = norm
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(), # Splits Arabic from Latin/Digits
            pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        ])
        tokenizer.decoder = decoders.Metaspace()
        
    return tokenizer

def build_trainer(cfg, vocab_size):
    if cfg.get("model", "bpe") == "bpe":
        return trainers.BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=cfg.get("min_freq", 2), 
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
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=sorted(LANG_CONFIGS), help="Language code (e.g., pl)")
    p.add_argument("--hf-user", default="Beetle-Data", help="HF Username for upload")
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--sentences", type=int, default=1_000_000, help="Total sentences to stream")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = LANG_CONFIGS[args.lang]
    out_dir = Path(f"tokenizer-{args.lang}-local")
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- 🛠️  Preparing Pipeline for {cfg['name']} ---")
    tokenizer = build_tokenizer(cfg)
    trainer = build_trainer(cfg, args.vocab_size)
    
    # --- The "Anti-Stuck" Logging Strategy ---
    print(f"🚀 Starting training from stream ({args.sentences:,} sentences)...")
    
    # We wrap the generator in tqdm so we see progress while train_from_iterator consumes it
    corpus = get_training_corpus(cfg, args.sentences)
    
    tokenizer.train_from_iterator(
        tqdm(corpus, total=args.sentences, desc="Tokenizing & Counting", unit="sents", colour="green"),
        trainer=trainer
    )

    print(f"\n✅ Training Complete. Wrapping in PreTrainedTokenizerFast...")
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=cfg.get("clean_up_spaces", False)
    )
    
    hf_tokenizer.save_pretrained(out_dir)
    print(f"📦 Saved locally to: {out_dir}")

    # --- Push to Hub ---
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        repo_id = f"{args.hf_user}/tokenizer-{args.lang}-en"
        print(f"📤 Pushing to HuggingFace Hub: {repo_id}")
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"🎉 Successfully uploaded!")
    else:
        print("💡 Tip: Set HF_TOKEN environment variable to auto-upload to Hub.")

if __name__ == "__main__":
    main()
