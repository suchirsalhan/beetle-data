#!/usr/bin/env python3
"""
Arabic + English Unigram SentencePiece Tokenizer
=================================================
Script:   train_tokenizer_ar.py
Model:    Unigram (SentencePiece-style)
Data:     FineWeb-2 (arb_Arab) + FineWeb-edu (English)
Target:   HuggingFace Hub → {HF_USER}/tokenizer-ar-en

Why Unigram over Byte-Level BPE for Arabic?
  - Arabic is morphologically very rich (clitics, root-and-pattern system)
    e.g. "وَسَيَكْتُبُونَهَا" = one word encoding ~6 morphemes
  - Byte-Level BPE treats Arabic letters (2 bytes each in UTF-8) as raw bytes,
    wasting vocab slots on byte-pair noise instead of meaningful sub-words
  - Unigram's probabilistic model handles the morphological complexity better:
    it finds globally optimal segmentations rather than greedy left-to-right merges
  - Arabic DOES use spaces (unlike Japanese), but morphological clitics attach
    without spaces, so sub-word granularity still matters a lot

Arabic-specific normalization choices (see build_tokenizer):
  - NFKC     : collapses Arabic Presentation Forms (U+FE70–FEFF) → canonical forms
  - Tatweel  : strip U+0640 (kashida / letter-stretcher) — purely cosmetic in text
  - Alef     : we intentionally do NOT normalize أ/إ/آ → ا
                (hamza position carries meaning; AraBERT ablations show ~1pt drop)
  - Tashkeel : we intentionally do NOT strip diacritics (harakat)
                (they are rare in web text; stripping harms diacritized religious text)
"""

import os
import re
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
HF_USER             = "RA-ALTA"
HF_TOKEN            = os.environ.get("HF_TOKEN")
LANG                = "ar"
VOCAB_SIZE          = 50_000
BOOTSTRAP_SENTENCES = 2_000_000   # total sentences (split 50/50 ar/en)
REPO_ID             = f"{HF_USER}/tokenizer-{LANG}-en"
OUT_DIR             = Path("tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. DATA GENERATOR
# =====================================================
def get_training_corpus():
    """
    Yields alternating Arabic and English sentences.

    Arabic source  : FineWeb-2 → arb_Arab
                     (Modern Standard Arabic, Arabic script)
                     Replaces the previous uonlp/CulturaX source — FineWeb-2
                     is quality-filtered and more consistent with FineWeb-edu.
    English source : FineWeb-edu

    Newlines collapsed so each yield is a single training line.
    """
    ar_ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="arb_Arab",          # Modern Standard Arabic
        split="train",
        streaming=True,
    )
    en_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    ar_iter, en_iter = iter(ar_ds), iter(en_ds)

    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            yield next(ar_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

# =====================================================
# 2. BUILD TOKENIZER
# =====================================================
def build_tokenizer() -> Tokenizer:
    """
    Constructs a Unigram tokenizer tuned for Arabic + English.

    Pipeline
    --------
    1. Normalizer: Sequence([NFKC, Strip Tatweel])

       a) NFKC
          - Decomposes and recomposes with compatibility equivalence
          - Collapses Arabic Presentation Forms (isolated/initial/medial/final
            letter glyphs in the FE block) → their canonical Unicode base forms
            e.g. ﻋ (U+FEAB) → ع (U+0639)
          - Also handles full-width Latin/digits in mixed Arabic-English content

       b) Strip Tatweel (U+0640)
          - Kashida is a calligraphic stretcher; it carries zero linguistic info
          - Common in low-quality web text and OCR output
          - Remove it so "كتـــاب" and "كتاب" map to the same tokens

    2. Pre-tokenizer: Sequence([UnicodeScripts, Metaspace])

       a) UnicodeScripts
          - Splits at script-boundary transitions
            e.g. "برنامج Python للذكاء الاصطناعي"
                 → ["برنامج ", "Python", " للذكاء الاصطناعي"]
          - Crucial for Arabic-English mixed text (code-switching is very common
            in Arabic web data)
          - Keeps Arabic tokens together, Latin tokens together

       b) Metaspace (replacement="▁", add_prefix_space=True)
          - SentencePiece-style space encoding
          - Arabic uses spaces between words, so ▁ correctly marks word starts
          - Enables lossless round-trip for both Arabic and English
          - Better than ByteLevel: no byte-fragment noise on Arabic letters

    3. Model: Unigram
       - Probabilistic segmentation; well-suited to morphologically rich languages
       - Finds globally optimal splits instead of greedy BPE merges
       - Arabic clitics (و / ب / ك / ل / ال prefixes, ه / ها / هم suffixes)
         are learned as sub-word units naturally

    4. Decoder: Metaspace
       - Reverses ▁ markers → spaces on decode

    Why NOT ByteLevel here?
       - Arabic letters = 2 bytes each in UTF-8 (U+0600–U+06FF range)
       - ByteLevel BPE would learn byte-pair merges like (0xD8, 0xB9) → ع
         instead of learning that ع is a common letter and علم is a common root
       - Results in ~2× more tokens per Arabic word vs Unigram
    """
    tokenizer = Tokenizer(models.Unigram())

    # --- Normalizer ---
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        # Strip tatweel (U+0640) — purely cosmetic, never meaningful
        normalizers.Replace(pattern="\u0640", content=""),
    ])

    # --- Pre-tokenizer ---
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),     # split Arabic / Latin / digits
        pre_tokenizers.Metaspace(            # space-as-▁
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
        # shrinking_factor: fraction of vocab kept each EM pruning round
        # 0.75 (default) is safe; lower = faster but slightly less optimal
        shrinking_factor=0.75,
        # max_piece_length: Arabic words can be long due to attached clitics
        # e.g. "وَسَيَكْتُبُونَهَا" = 18 chars; 20 covers edge cases
        max_piece_length=20,
        # n_sub_iterations: EM steps per pruning round; 2 is standard
        n_sub_iterations=2,
    )

    print("📚 Streaming training data (FineWeb-2 arb_Arab + FineWeb-edu)…")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # ---- Wrap in HF PreTrainedTokenizerFast ----
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,  # Metaspace decoder owns spacing
    )

    hf_tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ Tokenizer saved → {OUT_DIR}")

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
    Spot-checks tokenization across Arabic script varieties and mixed content.

    Key checks:
      - Pure Arabic: basic sentence, clitic-heavy, diacritized
      - Mixed Arabic-English (code-switching)
      - Tatweel stripping (normalization test)
      - Arabic numerals vs Eastern Arabic-Indic numerals
      - Round-trip losslessness for all cases
    """
    print("\n🧪 Benchmark\n" + "─" * 60)
    tk = AutoTokenizer.from_pretrained(OUT_DIR)

    test_cases = {
        # Basic Arabic
        "AR_Simple"        : "مرحباً، أنا جائع.",
        # Clitic-heavy: "and-will-write-it-they" = one word
        "AR_Clitics"       : "وسيكتبونها في التقرير.",
        # Diacritized text (Quranic-style)
        "AR_Diacritized"   : "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        # Tatweel (kashida) — should normalise to same as without
        "AR_Tatweel"       : "كتـــاب",
        # Eastern Arabic-Indic numerals
        "AR_Numerals"      : "السعر هو ١٢٬٠٠٠ ريال.",
        # Code-switching: Arabic + English + Latin script
        "AR_EN_Mixed"      : "نستخدم Python لتطوير نماذج الذكاء الاصطناعي.",
        # English
        "EN_Simple"        : "The quick brown fox jumps over the lazy dog.",
        # Emoji
        "Emoji"            : "التعلم ممتع! 🚀🔥",
    }

    all_passed = True
    for name, text in test_cases.items():
        tokens  = tk.encode(text)
        decoded = tk.decode(tokens)
        n_tok   = len(tokens)
        n_chr   = len(text)
        ratio   = n_tok / n_chr

        # Tatweel test: decoded won't contain tatweel (normalised away)
        # so compare against normalised input
        normalised_input = text.replace("\u0640", "")
        check_text = normalised_input if name == "AR_Tatweel" else text
        ok = decoded.strip() == check_text.strip()
        status = "✅" if ok else "❌ MISMATCH"
        if not ok:
            all_passed = False

        print(f"[{name}]")
        print(f"  Input  : {text!r}")
        print(f"  Tokens : {n_tok}  (chars={n_chr}, tok/char={ratio:.2f})")
        print(f"  IDs    : {tokens[:10]}{'…' if n_tok > 10 else ''}")
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
    """
    Print a sample of the learned vocabulary.
    Useful to verify:
      - Arabic sub-words appear (not raw byte sequences)
      - Common prefixes (ال، وال، بال، في، من) are their own tokens
      - Arabic and Latin tokens coexist cleanly
    """
    tk = AutoTokenizer.from_pretrained(OUT_DIR)
    vocab = tk.get_vocab()
    specials = {"<unk>", "<s>", "</s>", "<pad>"}

    print(f"\n🔍 Vocab sample (first {n} non-special tokens)\n" + "─" * 40)
    shown = 0
    for piece, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if piece in specials:
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
