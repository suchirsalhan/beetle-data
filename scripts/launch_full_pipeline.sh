#!/bin/bash
# =============================================================================
# launch_full_pipeline.sh — Storage-optimized pipeline on a single node.
#
# Processes languages one at a time: decontaminate → pretokenize → upload to HF
# → delete local files. Peak disk usage stays under 300 GB.
#
# Usage:
#   # All 19 languages (default)
#   bash scripts/launch_full_pipeline.sh
#
#   # Specific languages
#   bash scripts/launch_full_pipeline.sh --lang pl nl es
#
#   # Without HF upload (keep local files — needs more disk)
#   bash scripts/launch_full_pipeline.sh --lang pl --no-upload
#
#   # Custom output directory (e.g., on SSD mount)
#   OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_full_pipeline.sh
# =============================================================================

set -euo pipefail

# Resolve BEETLE_DATA to the directory containing this script's parent
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$BEETLE_DATA/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
NUM_WORKERS="${NUM_WORKERS:-24}"

cd "$BEETLE_DATA"

# Activate virtual environment if it exists
if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: 'python' not found in PATH. Activate a virtual environment first."
    exit 1
fi

# Verify critical dependencies before starting long-running jobs
python -c "
import datasets, pyarrow, transformers, tokenizers
print(f'  datasets={datasets.__version__}  pyarrow={pyarrow.__version__}  transformers={transformers.__version__}')
" || { echo "ERROR: Missing Python dependencies. Run: pip install -r requirements.txt"; exit 1; }

echo "============================================================"
echo "Beetle-Data Pipeline: Storage-Optimized"
echo "  Beetle-Data:  $BEETLE_DATA"
echo "  Project root: $PROJECT_ROOT"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Workers:      $NUM_WORKERS"
echo "  Peak disk:    ~270 GB (with upload+cleanup)"
echo "============================================================"

# Pass all arguments through to the Python orchestrator
python -m pipeline.run_pipeline \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --hf-user "$HF_USER" \
    --num-workers "$NUM_WORKERS" \
    "$@"

echo ""
echo "============================================================"
echo "Pipeline complete."
echo "  Datasets uploaded to: https://huggingface.co/$HF_USER"
echo "============================================================"
