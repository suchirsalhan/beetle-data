#!/bin/bash
# =============================================================================
# launch_pretokenize_babybabel.sh — Pretokenize BabyBabel human-scale data.
#
# Tokenizes 9 BabyLM-community datasets with bilingual tokenizers from
# Beetle-HumanScale and pushes Arrow datasets to Beetle-Data on HuggingFace.
#
# Human-scale data is small (~50M tokens/lang), so a single node suffices.
# All 36 pairs (72 Arrow datasets) complete in under an hour.
#
# Usage:
#   # All 36 pairs (default)
#   sbatch scripts/launch_pretokenize_babybabel.sh
#
#   # Pilot only (3 pairs: nld-eng, zho-eng, zho-nld)
#   MODE=pilot sbatch scripts/launch_pretokenize_babybabel.sh
#
#   # Single pair
#   L1=eng L2=nld sbatch scripts/launch_pretokenize_babybabel.sh
#
#   # Skip HuggingFace upload
#   NO_UPLOAD=1 sbatch scripts/launch_pretokenize_babybabel.sh
#
# Prerequisites:
#   - Bilingual tokenizers on Beetle-HumanScale (run beetlelm/script/run_tokenizers.sh first)
#   - HF_TOKEN set for tokenizer download and dataset upload
# =============================================================================

#SBATCH --job-name=beetle-pretok-babybabel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pretok_babybabel_%j.out
#SBATCH --error=logs/pretok_babybabel_%j.err

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output/babybabel}"
TARGET="${TARGET:-50M}"
MODE="${MODE:-all}"      # "all", "pilot", or "pair"
L1="${L1:-}"
L2="${L2:-}"
NO_UPLOAD="${NO_UPLOAD:-0}"

mkdir -p logs

# ── Activate environment ────────────────────────────────────────────────────
cd "$BEETLE_DATA"
source venvs/demo/bin/activate 2>/dev/null || true

export HF_HOME="${HF_HOME:-/scratch/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true

# ── Build CLI args ──────────────────────────────────────────────────────────
CLI_ARGS="--target $TARGET --output-dir $OUTPUT_DIR"

if [ "$NO_UPLOAD" = "1" ]; then
    CLI_ARGS="$CLI_ARGS --no-upload"
fi

if [ -n "$L1" ] && [ -n "$L2" ]; then
    CLI_ARGS="--pair $L1 $L2 $CLI_ARGS"
elif [ "$MODE" = "pilot" ]; then
    CLI_ARGS="--pilot $CLI_ARGS"
else
    CLI_ARGS="--all $CLI_ARGS"
fi

# ── Run pretokenization ────────────────────────────────────────────────────
echo "Starting BabyBabel pretokenization (mode=$MODE, target=$TARGET)"
echo "  Output: $OUTPUT_DIR"
echo "  Upload: $([ "$NO_UPLOAD" = "1" ] && echo "disabled" || echo "enabled")"

python -m pipeline.pretokenize_babybabel $CLI_ARGS

echo "BabyBabel pretokenization complete."
