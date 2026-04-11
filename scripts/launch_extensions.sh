#!/bin/bash
# =============================================================================
# launch_extensions.sh — Pipeline for extension languages.
#
# Runs the same decontamination + pretokenization pipeline as core languages,
# but for the 8 extension languages (Urdu, Bengali, Czech, Gujarati, Thai,
# Vietnamese, Korean, Danish). These are lower priority and run separately.
#
# All outputs are pushed to HF org: Beetle-Data
#
# Usage:
#   bash scripts/launch_extensions.sh
#   bash scripts/launch_extensions.sh --lang ur bn  # specific languages
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BEETLE_DATA="${PROJECT_ROOT}/beetle-data"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
NUM_WORKERS="${NUM_WORKERS:-24}"

EXTENSION_LANGS=("ur" "bn" "cs" "gu" "th" "vi" "ko" "da")

cd "$BEETLE_DATA"

echo "============================================================"
echo "Beetle-Data Pipeline: Extension Languages"
echo "  Languages: ${EXTENSION_LANGS[*]}"
echo "  Output dir: $OUTPUT_DIR"
echo "  HF org:     $HF_USER"
echo "============================================================"

# If specific languages passed via --lang, use those; otherwise all extensions
if [[ "$*" == *"--lang"* ]]; then
    python -m pipeline.run_pipeline \
        --project-root "$PROJECT_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --hf-user "$HF_USER" \
        --num-workers "$NUM_WORKERS" \
        "$@"
else
    python -m pipeline.run_pipeline \
        --project-root "$PROJECT_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --hf-user "$HF_USER" \
        --num-workers "$NUM_WORKERS" \
        --lang "${EXTENSION_LANGS[@]}" \
        "$@"
fi

echo ""
echo "============================================================"
echo "Extension pipeline complete."
echo "  Datasets uploaded to: https://huggingface.co/$HF_USER"
echo "============================================================"
