#!/bin/bash
# =============================================================================
# launch_full_pipeline.sh — STEP 1: Static pipeline on a single node.
#
# Runs Stages 1+2+3 (benchmark index → decontaminate → pretokenize) for the
# specified languages. Designed to be called once per node with a disjoint
# language set, giving 4 parallel nodes for 20 core languages:
#
#   Node 0:  bash scripts/launch_full_pipeline.sh --lang fr de es zh ja
#   Node 1:  bash scripts/launch_full_pipeline.sh --lang nl it ru pl tr
#   Node 2:  bash scripts/launch_full_pipeline.sh --lang tl hi ta eu ar
#   Node 3:  bash scripts/launch_full_pipeline.sh --lang sv el ca fa id
#
# Token targets (change in pipeline/config.py to adjust):
#   Stream:  28B clean tokens per language (≈ 21.5B whitespace words)
#   Train:   24B tokens per bilingual pair (12B L1 + 12B EN)
#
# Stage 1 (benchmark index) is built once by whichever node runs first.
# All subsequent nodes skip Stage 1 if the index file already exists.
# Requires a shared filesystem (e.g. /mnt/ssd-3) so all nodes see the index.
#
# Storage-optimized: processes languages one at a time, uploads each Arrow
# dataset to HuggingFace, then deletes local files. Peak disk: ~320 GB/node.
#
# Options:
#   --lang <codes>       Space-separated language codes to process (required)
#   --no-upload          Keep local files, skip HF upload (needs ~3.5 TB/node)
#   --curriculum-prep    Also upload raw Parquet shards to HF after Stage 2
#                        (enables later curriculum Stage D+3 without shared FS)
#   OUTPUT_DIR=<path>    Base output directory (default: pipeline_output)
#   HF_USER=<org>        HuggingFace organization (default: Beetle-Data)
#   NUM_WORKERS=<n>      Multiprocessing workers (default: 24)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$BEETLE_DATA/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
# NUM_WORKERS defaults to empty — PipelineConfig auto-detects min(cpu_count - 4, 64).
# Set the env var (e.g. NUM_WORKERS=48) to override.
NUM_WORKERS="${NUM_WORKERS:-}"

cd "$BEETLE_DATA"

# Activate virtual environment if present
if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

if ! command -v python &> /dev/null; then
    echo "ERROR: 'python' not found. Activate a virtual environment first."
    exit 1
fi

python -c "
import datasets, pyarrow, transformers, tokenizers
print(f'  datasets={datasets.__version__}  pyarrow={pyarrow.__version__}  transformers={transformers.__version__}')
" || { echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"; exit 1; }

echo "============================================================"
echo "Beetle-Data Pipeline: STEP 1 — Static Pipeline (Stages 1+2+3)"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Workers:        ${NUM_WORKERS:-auto (min(cpu_count - 4, 64))}"
echo "  Stream target:  28B clean tokens per language side"
echo "  Train target:   24B tokens per bilingual pair (12B L1 + 12B EN)"
echo "  Peak disk:      ~320 GB (with HF upload + cleanup)"
echo "============================================================"
echo ""
echo "Note: Stage 1 (benchmark index) is skipped if already built."
echo "      Delete $OUTPUT_DIR/benchmark_13gram.pkl to force rebuild."
echo ""

# Pass all arguments through to the Python orchestrator.
# --skip-disk-check suppresses the pre-flight disk check (adjust via
# max_local_disk_gb in PipelineConfig if you need a tighter guard).
# Only forward --num-workers when the env var is explicitly set; otherwise
# PipelineConfig auto-detects based on cpu_count().
EXTRA_ARGS=()
if [ -n "$NUM_WORKERS" ]; then
    EXTRA_ARGS+=(--num-workers "$NUM_WORKERS")
fi

# Supervisor retry loop: Stage 2 fails fast on stream / host-RAM errors
# (see pipeline/decontaminate_stream.py). On exit non-zero we restart in a
# fresh process; the per-language checkpoint resumes at the last committed
# shard boundary, and partial Parquet shards are already uploaded to HF via
# the periodic upload hook — so no work is lost across restarts.
MAX_SUPERVISOR_ATTEMPTS="${MAX_SUPERVISOR_ATTEMPTS:-8}"
RESTART_BACKOFF_SEC="${RESTART_BACKOFF_SEC:-30}"

# Disable `set -e` for the loop body so a non-zero exit triggers a retry
# instead of killing the script.
set +e
attempt=1
exit_code=0
while [ "$attempt" -le "$MAX_SUPERVISOR_ATTEMPTS" ]; do
    echo "------------------------------------------------------------"
    echo "Supervisor attempt $attempt / $MAX_SUPERVISOR_ATTEMPTS"
    echo "------------------------------------------------------------"

    python -m pipeline.run_pipeline \
        --project-root "$PROJECT_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --hf-user "$HF_USER" \
        --skip-disk-check \
        "${EXTRA_ARGS[@]}" \
        "$@"
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        break
    fi

    echo "Pipeline exited with code $exit_code; resuming from checkpoint in ${RESTART_BACKOFF_SEC}s (attempt $attempt)"
    sleep "$RESTART_BACKOFF_SEC"
    attempt=$((attempt + 1))
done
set -e

if [ "$exit_code" -ne 0 ]; then
    echo "Pipeline failed after $MAX_SUPERVISOR_ATTEMPTS supervisor attempts (last exit code: $exit_code)" >&2
    exit "$exit_code"
fi

echo ""
echo "============================================================"
echo "STEP 1 complete."
echo "  Pretokenized datasets: https://huggingface.co/$HF_USER"
echo "============================================================"
