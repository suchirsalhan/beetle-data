#!/bin/bash
# =============================================================================
# launch_held_out_babybabel.sh — Stream held-out BabyBabel (100M ∖ 50M)
# for Beetle-HumanScale AoA evaluation.
#
# For bilingual training we use Beetle-Data/BabyBabel-{lang}-50M; held-out
# comes from Beetle-Data/BabyBabel-{lang}-100M minus the 50M subset (dedup
# by document-text hash — the two HF datasets are independently curated).
#
# Usage:
#   bash scripts/launch_held_out_babybabel.sh
#
#   # Override default language list
#   LANGS="deu nld zho" bash scripts/launch_held_out_babybabel.sh
#
#   # Cap held-out doc count per language (smoke test)
#   MAX_DOCS=5000 bash scripts/launch_held_out_babybabel.sh
#
#   # Custom output directory
#   OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_held_out_babybabel.sh
#
# Output: pipeline_output/held_out_babybabel/{lang}/*.parquet
#         pipeline_output/held_out_babybabel/{lang}/{lang}_held_out_stats.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
LANGS="${LANGS:-deu nld zho}"
MAX_DOCS="${MAX_DOCS:-}"

cd "$BEETLE_DATA"

if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

echo "============================================================"
echo "Beetle-Data: BabyBabel Held-Out Streaming (for AoA eval)"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Languages:     $LANGS"
if [ -n "$MAX_DOCS" ]; then
    echo "  Max docs/lang: $MAX_DOCS"
fi
echo "============================================================"
echo ""

FAILED_LANGS=()

for lang in $LANGS; do
    echo ">>> Streaming held-out BabyBabel for: $lang"
    args=(--lang "$lang" --output-dir "$OUTPUT_DIR")
    if [ -n "$MAX_DOCS" ]; then
        args+=(--max-docs "$MAX_DOCS")
    fi
    if ! python -m pipeline.stream_held_out_babybabel "${args[@]}"; then
        echo "ERROR: held-out streaming failed for '$lang'"
        FAILED_LANGS+=("$lang")
    fi
    echo ""
done

echo "============================================================"
echo "BabyBabel held-out streaming complete."
if [ ${#FAILED_LANGS[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED_LANGS[*]}"
    exit 1
fi
echo "  Held-out data: $OUTPUT_DIR/held_out_babybabel/"
echo "============================================================"
