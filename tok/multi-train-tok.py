#!/usr/bin/env python3
"""
Multi-language Tokenizer Trainer (BPE & Unigram)
================================================
Features:
- Streaming FineWeb-2 & FineWeb-edu (Zero disk usage)
- Language-specific normalization (NFC/NFKC/Tatweel)
- Custom Pre-tokenization (ByteLevel for BPE, UnicodeScripts for Unigram)
- Real-time progress tracking via tqdm
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Ensure all dependencies are available
try:
    from datasets import load_dataset
    from tokenizers import (
        Tokenizer, models, trainers, pre_tokenizers, 
        decoders, normalizers, processors
    )
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    from huggingface_hub import create_repo
except ImportError as e:
    print(f"❌ Error: Missing dependency. {e}")
    print("Run: pip install datasets tokenizers transformers tqdm huggingface_hub")
    sys.exit(1)

# =====================================================
# 1. LANGUAGE CONFIGS
# =====================================================
LANG_CONFIGS = {
    "pl": {"name": "Polish", "fw2_name": "pol_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "nl": {"name": "Dutch", "fw2_name": "nld_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "es": {"name": "Spanish", "fw2_name": "spa_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "fr": {"name": "French", "fw2_name": "fra_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "de": {"name": "German", "fw2_name": "deu_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "it": {"name": "Italian", "fw2_name": "ita_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "sv": {"name": "Swedish", "fw2_name": "swe_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "ca": {"name": "Catalan", "fw2_name": "cat_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "eu": {"name": "Basque", "fw2_name": "eus_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 1},
    "tr": {"name": "Turkish", "fw2_name": "tur_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "id": {"name": "Indonesian", "fw2_name": "ind_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "tl": {"name": "Tagalog", "fw2_name": "tgl_Latn", "model": "bpe", "norm": "nfkc", "min_freq": 1},
    "el": {"name": "Greek", "fw2_name": "ell_Grek", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "ru": {"name": "Russian", "fw2_name": "rus_Cyrl", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "zh": {
        "name": "Chinese", "fw2_name": "cmn_Hani", "model": "bpe", "norm": "nfc", 
        "add_prefix_space": True, "trim_offsets": True, "clean_up_spaces": True, "min_freq": 2
    },
    "ar": {
        "name": "Arabic", "fw2_name": "arb_Arab", "model": "unigram", "norm": "nfkc+tatweel", 
        "clean_up_spaces": False, "shrinking_factor": 0.75, "max_piece_length": 20
    },
    "fa": {"name": "Persian", "fw2_name": "pes_Arab", "model": "bpe", "norm": "nfkc+tatweel", "min_freq": 2},
    "hi": {"name": "Hindi", "fw2_name": "hin_Deva", "model": "bpe", "norm": "nfkc", "min_freq": 2},
    "ta": {"name": "Tamil", "fw2_name": "tam_Taml", "model": "bpe", "norm": "nfkc", "min_freq": 2},
}

# =====================================================
# 2. DATA STREAMER
# =====================================================
def get_training_corpus(cfg, n_sentences):
    """Yields text from FineWeb-2 and FineWeb-edu."""
    print(f"📡 Initializing connection to HuggingFace (Streaming {n_sentences:,} sentences)...")
    
    try:
        lang_ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg["fw2_name"], split="train", streaming=True)
        en_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        
        lang_iter, en_iter = iter(lang_ds), iter(en_ds)
        
        for i in range(n_sentences // 2):
            try:
                yield next(lang_iter)["text"].replace("\n", " ")
                yield next(en_iter)["text"].replace("\n", " ")
            except StopIteration:
                print(f"\n⚠️ Reached end of dataset early at sentence {i*2}")
                break
    except Exception as e:
        print(f"\n❌ Dataset Connection Error: {e}")
        sys.exit(1)

# =====================================================
# 3. BUILD COMPONENTS
# =====================================================
def build_tokenizer(cfg):
    model_type = cfg.get("model", "bpe")
    
    # Normalization logic
    if cfg["norm"] == "nfc": 
        norm = normalizers.NFC()
    elif cfg["norm"] == "nfkc": 
        norm = normalizers.NFKC()
    else: # nfkc+tatweel
        norm = normalizers.Sequence([
            normalizers.NFKC(), 
            normalizers.Replace(pattern="\u0640", content="")
        ])

    if model_type == "bpe":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = norm
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=cfg.get("add_prefix_space", False))
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=cfg.get("trim_offsets", False))
        tokenizer.decoder = decoders.ByteLevel()
    else: # Unigram (Arabic)
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = norm
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
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
# 4. MAIN PIPELINE
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=sorted(LANG_CONFIGS))
    parser.add_argument("--hf-user", default="RA-ALTA")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--sentences", type=int, default=1000000)
    args = parser.parse_args()

    cfg = LANG_CONFIGS[args.lang]
    out_dir = Path(f"tokenizer-{args.lang}-local")
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- 🛠️ Preparing {cfg['name']} ({args.lang}) Pipeline ---")
    tokenizer = build_tokenizer(cfg)
    trainer = build_trainer(cfg, args.vocab_size)
    
    # Progress Tracking
    print(f"🚀 Training {cfg['model'].upper()} model...")
    
    # We wrap the generator in tqdm so train_from_iterator displays a progress bar
    # 'total' is required for the percentage and ETA to work
    progress_iterator = tqdm(
        get_training_corpus(cfg, args.sentences), 
        total=args.sentences, 
        desc="Streaming & Training", 
        unit=" sentences",
        colour="cyan"
    )

    tokenizer.train_from_iterator(progress_iterator, trainer=trainer)

    print(f"\n✅ Training finished. Saving files...")
    
    # Convert to Transformers Fast Tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=cfg.get("clean_up_spaces", False)
    )
    
    hf_tokenizer.save_pretrained(out_dir)
    print(f"📁 Local files saved to: {out_dir}")

    # Hub Push
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        repo_id = f"{args.hf_user}/tokenizer-{args.lang}-en"
        print(f"📤 Pushing to Hub: {repo_id}...")
        create_repo(repo_id, exist_ok=True, token=hf_token)
        hf_tokenizer.push_to_hub(repo_id, token=hf_token)
        print(f"✨ Pushed successfully!")
    else:
        print("⏭️ Skipping Hub push (HF_TOKEN not found in environment).")

if __name__ == "__main__":
    # Force flushing of stdout for immediate terminal feedback
    main()
