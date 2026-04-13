#!/bin/bash
# =============================================================================
# launch_held_out.sh — STEP 2: Stream held-out evaluation data (all languages).
#
# Run on ONE node after STEP 1 (static pipeline) completes across all 4 nodes.
# Streams FineWeb-2 documents BEYOND the training cutoff for each language,
# producing a held-out Parquet dataset for learning curve analysis.
#
# The number of documents to skip is read from the Stage 2 stats file
# ({lang}_stats.json), guaranteeing no overlap with training data.
#
# Usage:
#   bash scripts/launch_held_out.sh
#
#   # Override number of held-out docs per language (default: 10,000)
#   N_DOCS=50000 bash scripts/launch_held_out.sh
#
#   # Custom output directory
#   OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_held_out.sh
#
# Output: pipeline_output/held_out/{lang}/*.parquet
#         pipeline_output/held_out/{lang}/{lang}_held_out_stats.json
#
# Requires: Stage 2 stats files must exist for all languages.
#           Run STEP 1 first on all 4 nodes.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
N_DOCS="${N_DOCS:-10000}"

# All 20 core languages (same order as NODE_ASSIGNMENTS in config.py)
CORE_LANGS=(fr de es zh ja nl it ru pl tr tl hi ta eu ar sv el ca fa id)

cd "$BEETLE_DATA"

if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

echo "============================================================"
echo "Beetle-Data Pipeline: STEP 2 — Held-Out Data Streaming"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Docs per lang: $N_DOCS"
echo "  Languages:    ${CORE_LANGS[*]}"
echo "============================================================"
echo ""

FAILED_LANGS=()

for lang in "${CORE_LANGS[@]}"; do
    STATS_FILE="$OUTPUT_DIR/decontaminated/$lang/${lang}_stats.json"
    if [ ! -f "$STATS_FILE" ]; then
        echo "WARNING: Stats file not found for '$lang': $STATS_FILE"
        echo "         Run STEP 1 for language '$lang' before streaming held-out data."
        FAILED_LANGS+=("$lang")
        continue
    fi

    echo ">>> Streaming held-out data for: $lang"
    python -m pipeline.stream_held_out \
        --lang "$lang" \
        --output-dir "$OUTPUT_DIR" \
        --n-docs "$N_DOCS"
    echo ""
done

echo "============================================================"
echo "STEP 2 complete."
if [ ${#FAILED_LANGS[@]} -gt 0 ]; then
    echo "  Skipped (missing Stage 2 stats): ${FAILED_LANGS[*]}"
fi
echo "  Held-out data: $OUTPUT_DIR/held_out/"
echo "============================================================"
