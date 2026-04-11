#!/bin/bash
# =============================================================================
# launch_decontaminate.sh — SLURM launcher for full storage-optimized pipeline.
#
# Runs the complete pipeline (Stages 1-3) per-node with storage-aware processing:
# each language is decontaminated, pretokenized, uploaded to HF, then cleaned up
# locally. Peak disk: ~270 GB per node.
#
# Usage:
#   sbatch scripts/launch_decontaminate.sh
#
# Environment variables:
#   OUTPUT_DIR    Base output directory (default: /mnt/ssd-3/beetle-data)
#   PROJECT_ROOT  PHD project root (default: /path/to/PHD)
#   HF_USER       HuggingFace org (default: Beetle-Data)
#   NUM_WORKERS   CPU workers per stage (default: 24)
# =============================================================================

#SBATCH --job-name=beetle-pipeline
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pipeline_%j_%t.out
#SBATCH --error=logs/pipeline_%j_%t.err

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/PHD}"
BEETLE_DATA="${PROJECT_ROOT}/beetle-data"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ssd-3/beetle-data}"
INDEX_PATH="${OUTPUT_DIR}/benchmark_13gram.pkl"
HF_USER="${HF_USER:-Beetle-Data}"
NUM_WORKERS="${NUM_WORKERS:-24}"

mkdir -p logs

# ── Activate environment ────────────────────────────────────────────────────
cd "$BEETLE_DATA"
source venvs/demo/bin/activate 2>/dev/null || true

export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true

NODE_ID=${SLURM_NODEID:-0}

# ── Stage 1: Build benchmark index (node 0 only) ───────────────────────────
if [ "$NODE_ID" -eq 0 ]; then
    if [ ! -f "$INDEX_PATH" ]; then
        echo "[Node $NODE_ID] Building benchmark index..."
        python -m pipeline.benchmark_index \
            --output "$INDEX_PATH" \
            --project-root "$PROJECT_ROOT"
    fi
fi

# Wait for index
echo "[Node $NODE_ID] Waiting for benchmark index..."
while [ ! -f "$INDEX_PATH" ]; do sleep 5; done

# ── Stages 2+3: Storage-optimized per-language processing ──────────────────
echo "[Node $NODE_ID] Starting storage-optimized pipeline..."
python -m pipeline.run_pipeline \
    --stage 2 3 \
    --node-id "$NODE_ID" \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --index "$INDEX_PATH" \
    --hf-user "$HF_USER" \
    --num-workers "$NUM_WORKERS"

echo "[Node $NODE_ID] Pipeline complete."
