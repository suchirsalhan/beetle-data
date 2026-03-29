#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# =========================
# PATH FIX
# =========================
# This finds the directory where THIS script (run.sh) is located
# so it can find the python files in the same folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# CONFIG
# =========================
HF_USER="Beetle-Data"
VOCAB_SIZE=50000
SENTENCES=2000000

# All languages
LANGS=(
  pl nl es el fr de it eu tr id tl fa hi ta sv ru ca ar zh
)

LOG_DIR="$SCRIPT_DIR/logs_tok"
mkdir -p "$LOG_DIR"

# =========================
# RUN MULTI-LANG TRAINING
# =========================
for LANG in "${LANGS[@]}"; do
  echo "===================================================="
  echo "🚀 STARTING PIPELINE FOR: $LANG"
  echo "===================================================="

  # Use $SCRIPT_DIR/ to point to the python file correctly
  python3 -u "$SCRIPT_DIR/multi-train-tok.py" \
    --lang "$LANG" \
    --hf-user "$HF_USER" \
    --vocab-size "$VOCAB_SIZE" \
    --sentences "$SENTENCES" \
    2>&1 | tee "$LOG_DIR/${LANG}.log"

  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Successfully finished $LANG"
  else
    echo "❌ ERROR: Training failed for $LANG. Check $LOG_DIR/${LANG}.log"
    exit 1
  fi
  echo ""
done

# =========================
# JAPANESE (separate script)
# =========================
if [ -f "$SCRIPT_DIR/ja-en-tok.py" ]; then
  echo "===================================================="
  echo "🚀 STARTING PIPELINE FOR: JA (Japanese)"
  echo "===================================================="

  python3 -u "$SCRIPT_DIR/ja-en-tok.py" \
    --hf-user "$HF_USER" \
    --vocab-size "$VOCAB_SIZE" \
    --sentences "$SENTENCES" \
    2>&1 | tee "$LOG_DIR/ja.log"

  echo "✅ Successfully finished JA"
fi

echo "🎉 ALL DONE"
