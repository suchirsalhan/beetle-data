#!/bin/bash
# =============================================================================
# launch_low_resource.sh — Pipeline for low-resource languages.
#
# Runs the decontamination + pretokenization pipeline for low-resource
# languages. These may not reach full 24B/12B/8B token counts.
#
# European: Hungarian, Bulgarian, Croatian, Ukrainian, Slovenian
# African:  Somali, Amharic, Yoruba, Wolof
#
# All outputs are pushed to HF org: Beetle-Data
#
# Usage:
#   bash scripts/launch_low_resource.sh
#   bash scripts/launch_low_resource.sh --lang hu bg  # specific languages
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BEETLE_DATA="${PROJECT_ROOT}/beetle-data"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
NUM_WORKERS="${NUM_WORKERS:-24}"

LOW_RESOURCE_LANGS=("hu" "bg" "hr" "uk" "sl" "so" "am" "yo" "wo")

cd "$BEETLE_DATA"

echo "============================================================"
echo "Beetle-Data Pipeline: Low-Resource Languages"
echo "  Languages: ${LOW_RESOURCE_LANGS[*]}"
echo "  Output dir: $OUTPUT_DIR"
echo "  HF org:     $HF_USER"
echo "  NOTE: These languages may not reach full token targets."
echo "============================================================"

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
        --lang "${LOW_RESOURCE_LANGS[@]}" \
        "$@"
fi

echo ""
echo "============================================================"
echo "Low-resource pipeline complete."
echo "  Datasets uploaded to: https://huggingface.co/$HF_USER"
echo "============================================================"
