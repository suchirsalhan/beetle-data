#!/bin/bash
# =============================================================================
# launch_push_held_out_babybabel.sh — Publish held-out BabyBabel shards to HF.
#
# Pushes the output of launch_held_out_babybabel.sh (per-language parquet +
# stats) to Beetle-HumanScale as one dataset repo per language:
#   https://huggingface.co/datasets/Beetle-HumanScale/BabyBabel-{lang}-held-out
#
# Usage:
#   bash scripts/launch_push_held_out_babybabel.sh
#
#   # Override language list
#   LANGS="zho nld eng deu" bash scripts/launch_push_held_out_babybabel.sh
#
#   # Custom source directory (must contain held_out_babybabel/{lang}/*.parquet)
#   OUTPUT_DIR=/root/beetle-data/pipeline_output \
#       bash scripts/launch_push_held_out_babybabel.sh
#
#   # Override destination org
#   ORG=Beetle-Data bash scripts/launch_push_held_out_babybabel.sh
#
#   # Preview without uploading
#   DRY_RUN=1 bash scripts/launch_push_held_out_babybabel.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
LANGS="${LANGS:-zho nld eng deu}"
ORG="${ORG:-Beetle-HumanScale}"
DRY_RUN="${DRY_RUN:-}"

cd "$BEETLE_DATA"

if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

echo "============================================================"
echo "Beetle-Data: Push held-out BabyBabel → HF"
echo "  Source dir: $OUTPUT_DIR/held_out_babybabel/"
echo "  Languages:  $LANGS"
echo "  Target org: $ORG"
if [ -n "$DRY_RUN" ]; then
    echo "  DRY RUN:    enabled"
fi
echo "============================================================"
echo ""

args=(--langs $LANGS --output-dir "$OUTPUT_DIR" --org "$ORG")
if [ -n "$DRY_RUN" ]; then
    args+=(--dry-run)
fi

python -m pipeline.push_held_out_babybabel "${args[@]}"
status=$?

echo ""
echo "============================================================"
if [ $status -ne 0 ]; then
    echo "Push FAILED (exit=$status). See log above for failing languages."
    exit $status
fi
echo "Push complete. Datasets:"
for lang in $LANGS; do
    echo "  https://huggingface.co/datasets/${ORG}/BabyBabel-${lang}-held-out"
done
echo "============================================================"
