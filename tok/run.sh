#!/usr/bin/env bash

set -e

# =========================
# CONFIG
# =========================
HF_USER="Beetle-Data"
VOCAB_SIZE=50000
SENTENCES=2000000

# All languages from your config
LANGS=(
  pl nl es el fr de it eu tr id tl fa hi ta sv ru ca ar zh
)

LOG_DIR="logs_tok"
mkdir -p "$LOG_DIR"

# =========================
# RUN MULTI-LANG TRAINING
# =========================
for LANG in "${LANGS[@]}"; do
  echo "======================================="
  echo "🚀 Training tokenizer for $LANG"
  echo "======================================="

  python multi-train-tok.py \
    --lang "$LANG" \
    --hf-user "$HF_USER" \
    --vocab-size "$VOCAB_SIZE" \
    --sentences "$SENTENCES" \
    > "$LOG_DIR/${LANG}.log" 2>&1

  echo "✅ Finished $LANG"
done

# =========================
# JAPANESE (separate script)
# =========================
echo "======================================="
echo "🚀 Training tokenizer for JA"
echo "======================================="

python ja-en-tok.py \
  --hf-user "$HF_USER" \
  --vocab-size "$VOCAB_SIZE" \
  --sentences "$SENTENCES" \
  > "$LOG_DIR/ja.log" 2>&1

echo "🎉 ALL DONE"
