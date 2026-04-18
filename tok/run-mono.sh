#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# CONFIG
# =========================
HF_USER="Beetle-Data"
VOCAB_SIZE=50000
SENTENCES=2000000       # Match the total data volume of the bilingual tokenizers
REPO_SUFFIX="mono"

LANGS=(en nl de zh)

LOG_DIR="$SCRIPT_DIR/logs_tok_mono"
mkdir -p "$LOG_DIR"

# =========================
# RUN MONOLINGUAL TRAINING
# =========================
for LANG in "${LANGS[@]}"; do
  echo "===================================================="
  echo "STARTING MONOLINGUAL PIPELINE FOR: $LANG"
  echo "===================================================="

  python3 -u "$SCRIPT_DIR/mono-train-tok.py" \
    --lang "$LANG" \
    --hf-user "$HF_USER" \
    --vocab-size "$VOCAB_SIZE" \
    --sentences "$SENTENCES" \
    --repo-suffix "$REPO_SUFFIX" \
    2>&1 | tee "$LOG_DIR/${LANG}.log"

  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Successfully finished $LANG"
  else
    echo "ERROR: Training failed for $LANG. Check $LOG_DIR/${LANG}.log"
    exit 1
  fi
  echo ""
done

echo "ALL DONE"
