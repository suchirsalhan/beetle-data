#!/bin/bash
# =============================================================================
# launch_pretokenize.sh — SLURM launcher for Stage 3 across 4 nodes.
#
# Reads decontaminated Parquet from Stage 2 and pretokenizes into Arrow
# datasets compatible with beetlelm's PretokenizedMultilingualDataset.
#
# Each node processes its assigned languages (both L1 and EN sides).
#
# Usage:
#   sbatch scripts/launch_pretokenize.sh
#
# Prerequisites:
#   - Stage 2 must have completed (decontaminated Parquet shards exist)
#   - conda/venv with beetle-data requirements activated
#   - HF_TOKEN set for tokenizer download
# =============================================================================

#SBATCH --job-name=beetle-pretokenize
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pretokenize_%j_%t.out
#SBATCH --error=logs/pretokenize_%j_%t.err

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$BEETLE_DATA/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
NUM_WORKERS="${NUM_WORKERS:-24}"
SEQ_LEN="${SEQ_LEN:-512}"

mkdir -p logs

# ── Activate environment ────────────────────────────────────────────────────
cd "$BEETLE_DATA"
source venvs/demo/bin/activate 2>/dev/null || true

export HF_HOME="${HF_HOME:-/scratch/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true

# ── Stage 3: Pretokenize ───────────────────────────────────────────────────
NODE_ID=${SLURM_NODEID:-0}

echo "[Node $NODE_ID] Starting pretokenization for assigned languages..."
python -m pipeline.pretokenize_arrow \
    --node-id "$NODE_ID" \
    --output-dir "$OUTPUT_DIR" \
    --hf-user "$HF_USER" \
    --num-workers "$NUM_WORKERS" \
    --seq-len "$SEQ_LEN"

echo "[Node $NODE_ID] Pretokenization complete."
