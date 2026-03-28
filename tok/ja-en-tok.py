#!/usr/bin/env python3
"""
Japanese + English Unigram SentencePiece Tokenizer
====================================================
Script:   train_tokenizer_ja.py
Model:    Unigram (SentencePiece-style)
Data:     FineWeb-2 (jpn_Jpan) + FineWeb-edu (English)
Target:   HuggingFace Hub → {HF_USER}/tokenizer-ja-en

Why Unigram over Byte-Level BPE for Japanese?
  - Japanese has no word-delimiting spaces → ByteLevel prefix-space is meaningless
  - CJK characters are 3 bytes in UTF-8 → byte-level BPE fragments badly
  - Unigram + UnicodeScripts naturally segments across script boundaries
  - NFKC normalization handles full-width variants (１２３→123, ａ→a, etc.)
"""

import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    normalizers,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import create_repo

# =====================================================
# CONFIG
# =====================================================
HF_USER             = "Beetle-Data"
HF_TOKEN            = os.environ.get("HF_TOKEN")
LANG                = "ja"
VOCAB_SIZE          = 50_000
BOOTSTRAP_SENTENCES = 500_000   # total sentences (split 50/50 ja/en)
REPO_ID             = f"{HF_USER}/tokenizer-{LANG}-en"
OUT_DIR             = Path("tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. DATA GENERATOR
# =====================================================
def get_training_corpus():
    """
    Yields alternating Japanese and English sentences.

    Japanese source : FineWeb-2  → jpn_Jpan  (change from cmn_Hani used for zh)
    English source  : FineWeb-edu
    Newlines are collapsed to spaces so each yield is one training line.
    """
    ja_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="jpn_Jpan",          # ← jpn_Jpan for Japanese (was cmn_Hani for zh)
        split="train",
        streaming=True,
    )
    en_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    ja_iter, en_iter = iter(ja_ds), iter(en_ds)

    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            yield next(ja_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

# =====================================================
# 2. BUILD TOKENIZER
# =====================================================
def build_tokenizer() -> Tokenizer:
    """
    Constructs a Unigram tokenizer tuned for Japanese + English.

    Pipeline
    --------
    1. Normalizer  : NFKC
       - Collapses full-width/half-width variants common in Japanese web text
         e.g. １２３ → 123, ａｂｃ → abc, ｶﾅ (half-width katakana) → カナ
       - Also handles standard Unicode compatibility decomposition

    2. Pre-tokenizer: Sequence([UnicodeScripts, Metaspace])
       a) UnicodeScripts
          - Splits on script-boundary transitions
            e.g. "東京でAI研究" → ["東京で", "AI", "研究"]
          - Gives the Unigram model natural segmentation hints
            without needing MeCab or other morphological analysers
          - Handles Hiragana / Katakana / Kanji / Latin / digits independently
       b) Metaspace (replacement="▁", add_prefix_space=True)
          - SentencePiece-style space encoding: marks word beginnings with ▁
          - Allows lossless round-trip for English (which uses spaces)
          - Does not harm Japanese (no spaces to mark, ▁ only prepended once)

    3. Model: Unigram
       - Probabilistic; trained to find the most likely segmentation
       - Handles open-vocabulary gracefully via sub-word fallback
       - No explicit merge table (unlike BPE), so generalises better to
         unseen kanji compound sequences

    4. Decoder: Metaspace
       - Reverses ▁ markers back to spaces on decode

    Why NOT ByteLevel here?
       - Each CJK char = 3 raw UTF-8 bytes → fragments to up to 3 tokens per char
       - With vocab_size=50k the model wastes capacity on 3-byte kanji triplets
       - UnicodeScripts + Unigram produces single tokens per common kanji/kana
    """
    tokenizer = Tokenizer(models.Unigram())

    # --- Normalizer ---
    tokenizer.normalizer = normalizers.NFKC()

    # --- Pre-tokenizer ---
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),        # script-boundary splits
        pre_tokenizers.Metaspace(               # space-as-▁ (SentencePiece style)
            replacement="▁",
            add_prefix_space=True,
        ),
    ])

    # --- Decoder ---
    tokenizer.decoder = decoders.Metaspace()

    return tokenizer

# =====================================================
# 3. TRAIN & PUSH
# =====================================================
def train_and_push():
    print(f"🚀 Training Unigram tokenizer for '{LANG}+en'  (vocab={VOCAB_SIZE:,})…")

    tokenizer = build_tokenizer()

    trainer = trainers.UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        unk_token="<unk>",
        # shrinking_factor controls aggressiveness of vocab pruning each round
        # default 0.75 is fine; lower = faster but less optimal vocab
        shrinking_factor=0.75,
        # max_piece_length caps sub-word length; 16 covers longest Japanese compounds
        max_piece_length=16,
        # n_sub_iterations: EM iterations per pruning step; 2 is standard
        n_sub_iterations=2,
    )

    print("📚 Streaming training data…")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # ---- Wrap in HF PreTrainedTokenizerFast ----
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        # Do NOT set add_prefix_space here — Metaspace pre-tokenizer handles it
        clean_up_tokenization_spaces=False,  # Metaspace decoder owns spacing
    )

    # Save locally
    hf_tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ Tokenizer saved → {OUT_DIR}")

    # Push to Hub
    if HF_TOKEN:
        create_repo(REPO_ID, exist_ok=True, token=HF_TOKEN)
        hf_tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"✅ Pushed → https://huggingface.co/{REPO_ID}")
    else:
        print("⚠️  HF_TOKEN not set — skipping Hub push.")

# =====================================================
# 4. BENCHMARK
# =====================================================
def run_benchmark():
    """
    Spot-checks tokenization quality across script types.
    Verifies:
      - Token counts are reasonable (not fragmenting CJK to bytes)
      - Round-trip decode is lossless
      - Script boundaries are respected
    """
    print("\n🧪 Benchmark\n" + "─" * 60)
    tk = AutoTokenizer.from_pretrained(OUT_DIR)

    test_cases = {
        # Japanese
        "JA_Simple"      : "私は学生です。",                        # I am a student.
        "JA_Compound"    : "東京大学の研究者が新しいAIモデルを発表した。",  # compound nouns + katakana
        "JA_Mixed_Script": "彼女はPythonでディープラーニングを勉強している。",  # kanji+katakana+Latin
        "JA_FullWidth"   : "価格は１２，０００円です。",              # full-width digits (NFKC test)
        "JA_HalfKana"    : "ｺﾝﾆﾁﾊ",                              # half-width katakana (NFKC test)
        # English
        "EN_Simple"      : "The quick brown fox jumps over the lazy dog.",
        # Mixed
        "Mixed_JA_EN"    : "Hello！　今日はいい天気ですね。",
        # Emoji / special
        "Emoji"          : "学習は楽しい！ 🚀🔥",
    }

    all_passed = True
    for name, text in test_cases.items():
        tokens   = tk.encode(text)
        decoded  = tk.decode(tokens)
        n_tokens = len(tokens)
        n_chars  = len(text)
        ratio    = n_tokens / n_chars

        status = "✅" if decoded.strip() == text.strip() else "❌ MISMATCH"
        if decoded.strip() != text.strip():
            all_passed = False

        print(f"[{name}]")
        print(f"  Input  : {text!r}")
        print(f"  Tokens : {n_tokens}  (chars={n_chars}, tok/char={ratio:.2f})")
        print(f"  IDs    : {tokens[:10]}{'…' if n_tokens > 10 else ''}")
        print(f"  Decoded: {decoded!r}  {status}")
        print()

    if all_passed:
        print("🎉 All round-trip checks passed.")
    else:
        print("⚠️  Some round-trip checks FAILED — review normalizer/decoder settings.")

# =====================================================
# 5. VOCAB INSPECTION (optional helper)
# =====================================================
def inspect_vocab(n: int = 40):
    """Print a sample of the learned vocabulary to sanity-check coverage."""
    tk = AutoTokenizer.from_pretrained(OUT_DIR)
    vocab = tk.get_vocab()
    print(f"\n🔍 Vocab sample (first {n} non-special tokens)\n" + "─" * 40)
    skipped_special = 0
    shown = 0
    for piece, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if piece in ("<unk>", "<s>", "</s>", "<pad>"):
            skipped_special += 1
            continue
        print(f"  {idx:6d}  {piece!r}")
        shown += 1
        if shown >= n:
            break
    print(f"\nTotal vocab size: {len(vocab):,}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    train_and_push()
    run_benchmark()
    inspect_vocab(n=40)
