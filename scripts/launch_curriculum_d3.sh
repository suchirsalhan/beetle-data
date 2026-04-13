#!/bin/bash
# =============================================================================
# launch_curriculum_d3.sh — Curriculum STEP 4: Stages D + 3 per language.
#
# Runs scoring + indexing (Stage D) and pretokenization (Stage 3) for the
# specified languages. Designed to be called once per node with a disjoint
# language set — exactly the same command structure as STEP 1:
#
#   Node 0:  bash scripts/launch_curriculum_d3.sh --lang fr de es zh ja
#   Node 1:  bash scripts/launch_curriculum_d3.sh --lang nl it ru pl tr
#   Node 2:  bash scripts/launch_curriculum_d3.sh --lang tl hi ta eu ar
#   Node 3:  bash scripts/launch_curriculum_d3.sh --lang sv el ca fa id
#
# Prerequisites (downloaded automatically from HuggingFace if not local):
#   1. Raw decontaminated Parquet shards (Beetle-Data/{lang}-raw-28B)
#      — uploaded by STEP 1 when run with --curriculum-prep flag.
#   2. Student model artifacts (Beetle-Data/beetlestream-student-model)
#      — uploaded by launch_curriculum_abc.sh (STEP 3).
#   3. Feature Parquet / annotation data (Beetle-Data/beetlestream-annotations)
#      — uploaded by launch_curriculum_abc.sh (STEP 3).
#
# Outputs per language (uploaded to HuggingFace):
#   Beetle-Data/{lang}-indexed-28B   — Hive-partitioned Parquet, topic-indexed
#   Beetle-Data/{lang}-curriculum-28B — Pretokenized Arrow (quality/difficulty/topic_id)
#
# Token targets (from config.py):
#   Stream:  28B clean tokens per language (from raw Parquet)
#   Train:   24B tokens per bilingual pair (12B L1 + 12B EN), cut at Stage 3
#
# Compute: ~12 hrs wall-clock, 96 A100-hrs per node (8 GPUs × 12 hrs)
#
# Usage:
#   bash scripts/launch_curriculum_d3.sh --lang fr de es zh ja
#   OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_curriculum_d3.sh --lang fr de es zh ja
#
# Environment:
#   OUTPUT_DIR  Base output directory (default: pipeline_output)
#   HF_USER     HuggingFace org (default: Beetle-Data)
#   HF_SUFFIX   Dataset suffix (default: 28B)
#   HF_TOKEN    HuggingFace API token
#   NUM_WORKERS Multiprocessing workers (default: 24)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
HF_SUFFIX="${HF_SUFFIX:-28B}"
NUM_WORKERS="${NUM_WORKERS:-24}"
CONFIG="configs/beetlestream_curriculum.yaml"
LANGS=()

# Parse --lang arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LANGS+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hf-user)
            HF_USER="$2"
            shift 2
            ;;
        --hf-suffix)
            HF_SUFFIX="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ ${#LANGS[@]} -eq 0 ]; then
    echo "ERROR: No languages specified. Use --lang fr de es zh ja"
    exit 1
fi

cd "$BEETLE_DATA"

if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

echo "============================================================"
echo "BeetleStream v2: Curriculum STEP 4 — Stages D + 3"
echo "  Languages:    ${LANGS[*]}"
echo "  Output dir:   $OUTPUT_DIR"
echo "  HF org:       $HF_USER (suffix: $HF_SUFFIX)"
echo "  Workers:      $NUM_WORKERS"
echo "  Compute:      ~12 hrs wall-clock, 96 A100-hrs per node"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────
# Download Stage A-C artifacts from HuggingFace if not local.
# These are shared across all language nodes.
# ─────────────────────────────────────────────────────────────
download_from_hf() {
    local repo_id="$1"
    local local_dir="$2"
    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "  Already present locally: $local_dir — skipping download"
        return
    fi
    echo "  Downloading $repo_id → $local_dir ..."
    python - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$repo_id",
    repo_type="dataset",
    local_dir="$local_dir",
    token=os.environ.get("HF_TOKEN", "") or None,
)
print("  Download complete.")
PYEOF
}

echo ">>> Downloading Stage A-C artifacts from HuggingFace..."
download_from_hf "$HF_USER/beetlestream-student-model" "$OUTPUT_DIR/student_model"
download_from_hf "$HF_USER/beetlestream-annotations"   "$OUTPUT_DIR/annotations"
echo ""

# ─────────────────────────────────────────────────────────────
# Process each language sequentially on this node.
# ─────────────────────────────────────────────────────────────
for lang in "${LANGS[@]}"; do
    echo "============================================================"
    echo "Language: $lang"
    echo "============================================================"

    # Download raw decontaminated Parquet from HF if not local
    RAW_PARQUET_DIR="$OUTPUT_DIR/decontaminated/$lang"
    if [ ! -d "$RAW_PARQUET_DIR" ] || [ -z "$(ls "$RAW_PARQUET_DIR"/*.parquet 2>/dev/null)" ]; then
        echo ">>> Downloading raw Parquet for $lang from HF..."
        download_from_hf "$HF_USER/$lang-raw-$HF_SUFFIX" "$RAW_PARQUET_DIR"
    else
        echo ">>> Raw Parquet for $lang already local — skipping download"
    fi

    # Stage D: Score + index
    # Applies heuristic filters (stopword density, readability, script consistency,
    # repetition ratio), embeds surviving documents with the student model,
    # predicts quality/difficulty, clusters into 200 topics via k-means, and writes
    # Hive-partitioned Parquet shards.
    echo ">>> Stage D: Scoring and indexing $lang..."
    python -m pipeline.run_pipeline --stage D \
        --lang "$lang" \
        --output-dir "$OUTPUT_DIR" \
        --beetlestream-config "$CONFIG" \
        --hf-user "$HF_USER" \
        --num-workers "$NUM_WORKERS" \
        --skip-disk-check

    # Stage 3: Pretokenize from indexed shards (curriculum mode)
    # Produces Arrow dataset with quality, difficulty, and topic_id columns
    # alongside input_ids. Stops at TARGET_TOKENS_PER_LANG (24B) per pair.
    echo ">>> Stage 3: Pretokenizing curriculum data for $lang..."
    python -m pipeline.run_pipeline --stage 3 \
        --lang "$lang" \
        --output-dir "$OUTPUT_DIR" \
        --hf-user "$HF_USER" \
        --num-workers "$NUM_WORKERS" \
        --stream-mode curriculum \
        --dataset-suffix "$HF_SUFFIX" \
        --skip-disk-check

    # Clean up local raw Parquet and indexed shards after upload
    echo ">>> Cleaning up local files for $lang..."
    rm -rf "$RAW_PARQUET_DIR" || true
    rm -rf "$OUTPUT_DIR/indexed/$lang" || true
    echo ""
done

echo "============================================================"
echo "Curriculum STEP 4 complete for: ${LANGS[*]}"
echo "  Indexed shards:  https://huggingface.co/$HF_USER (indexed-$HF_SUFFIX)"
echo "  Curriculum Arrow: https://huggingface.co/$HF_USER (curriculum-$HF_SUFFIX)"
echo "============================================================"
