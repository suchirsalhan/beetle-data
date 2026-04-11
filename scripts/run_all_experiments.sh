#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Complete Beetle-Data pipeline for all language tiers.
#
# Runs tokenizer training, decontamination, and pretokenization for all
# languages across three tiers (core, extension, low-resource).
# All outputs are pushed to HF org: Beetle-Data
#
# Usage:
#   # Full pipeline (all tiers, sequential)
#   bash scripts/run_all_experiments.sh
#
#   # Specific tier only
#   bash scripts/run_all_experiments.sh --tier core
#   bash scripts/run_all_experiments.sh --tier extension
#   bash scripts/run_all_experiments.sh --tier low_resource
#
#   # Tokenizers only (all languages)
#   bash scripts/run_all_experiments.sh --tokenizers-only
#
#   # Skip tokenizer training (data pipeline only)
#   bash scripts/run_all_experiments.sh --skip-tokenizers
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BEETLE_DATA="${PROJECT_ROOT}/beetle-data"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
NUM_WORKERS="${NUM_WORKERS:-24}"

cd "$BEETLE_DATA"

# ── Parse arguments ──────────────────────────────────────────────────────────
TIER="all"
TOKENIZERS_ONLY=false
SKIP_TOKENIZERS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier) TIER="$2"; shift 2 ;;
        --tokenizers-only) TOKENIZERS_ONLY=true; shift ;;
        --skip-tokenizers) SKIP_TOKENIZERS=true; shift ;;
        *) shift ;;
    esac
done

CORE_LANGS=(pl nl es el ja fr zh de it eu tr id tl fa hi ta sv ru ca ar)
EXTENSION_LANGS=(ur bn cs gu th vi ko da)
LOW_RESOURCE_LANGS=(hu bg hr uk sl so am yo wo)

echo "============================================================"
echo "Beetle-Data: Run All Experiments"
echo "  Tier:        $TIER"
echo "  Output dir:  $OUTPUT_DIR"
echo "  HF org:      $HF_USER"
echo "============================================================"

# ── Step 1: Train tokenizers ────────────────────────────────────────────────
train_tokenizers() {
    local langs=("$@")
    echo ""
    echo "── Training tokenizers for ${#langs[@]} languages ──"
    for lang in "${langs[@]}"; do
        echo "[tok] Training tokenizer for $lang..."
        python3 tok/multi-train-tok.py \
            --lang "$lang" \
            --hf-user "$HF_USER" \
            --vocab-size 50000 \
            --sentences 2000000 || echo "[tok] WARNING: $lang tokenizer failed"
    done
}

if [[ "$SKIP_TOKENIZERS" == "false" ]]; then
    if [[ "$TIER" == "all" || "$TIER" == "core" ]]; then
        train_tokenizers "${CORE_LANGS[@]}"
    fi
    if [[ "$TIER" == "all" || "$TIER" == "extension" ]]; then
        train_tokenizers "${EXTENSION_LANGS[@]}"
    fi
    if [[ "$TIER" == "all" || "$TIER" == "low_resource" ]]; then
        train_tokenizers "${LOW_RESOURCE_LANGS[@]}"
    fi
fi

if [[ "$TOKENIZERS_ONLY" == "true" ]]; then
    echo "Tokenizer training complete. Exiting (--tokenizers-only)."
    exit 0
fi

# ── Step 2: Build benchmark index (one-time) ────────────────────────────────
INDEX_PATH="${OUTPUT_DIR}/benchmark_13gram.pkl"
if [[ ! -f "$INDEX_PATH" ]]; then
    echo ""
    echo "── Building 13-gram benchmark index ──"
    python -m pipeline.benchmark_index \
        --output "$INDEX_PATH" \
        --project-root "$PROJECT_ROOT"
fi

# ── Step 3: Run data pipeline per tier ───────────────────────────────────────
run_pipeline_for_langs() {
    local tier_name="$1"
    shift
    local langs=("$@")
    echo ""
    echo "── Running pipeline for $tier_name languages: ${langs[*]} ──"
    python -m pipeline.run_pipeline \
        --project-root "$PROJECT_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --hf-user "$HF_USER" \
        --num-workers "$NUM_WORKERS" \
        --lang "${langs[@]}"
}

if [[ "$TIER" == "all" || "$TIER" == "core" ]]; then
    run_pipeline_for_langs "core" "${CORE_LANGS[@]}"
fi

if [[ "$TIER" == "all" || "$TIER" == "extension" ]]; then
    run_pipeline_for_langs "extension" "${EXTENSION_LANGS[@]}"
fi

if [[ "$TIER" == "all" || "$TIER" == "low_resource" ]]; then
    run_pipeline_for_langs "low_resource" "${LOW_RESOURCE_LANGS[@]}"
fi

# ── Step 4: Verify output ───────────────────────────────────────────────────
echo ""
echo "── Verifying output ──"
python scripts/verify_output.py --output-dir "$OUTPUT_DIR" --hf-user "$HF_USER" --quick

echo ""
echo "============================================================"
echo "All Beetle-Data experiments complete."
echo "  Datasets: https://huggingface.co/$HF_USER"
echo "============================================================"
