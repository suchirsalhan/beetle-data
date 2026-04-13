#!/bin/bash
# =============================================================================
# launch_curriculum_abc.sh — Curriculum STEP 3: Stages A + B + C on one node.
#
# Runs the teacher annotation → feature transform → student model training
# stages of BeetleStream v2 on a SINGLE node with 8 A100-80GB GPUs.
#
# Prerequisite: STEP 1 (static pipeline) must have completed for all languages,
# and the raw decontaminated Parquet shards must be accessible either:
#   (a) On a shared filesystem at $OUTPUT_DIR/decontaminated/, OR
#   (b) On HuggingFace at Beetle-Data/{lang}-raw-28B
#       (uploaded automatically when --curriculum-prep was passed to STEP 1)
#
# After this script completes, two HuggingFace datasets are created:
#   Beetle-Data/beetlestream-annotations  — teacher annotation JSONL + feature Parquet
#   Beetle-Data/beetlestream-student-model — trained sklearn student model + embedder config
#
# These are needed by launch_curriculum_d3.sh (STEP 4) which can run later
# on 4 nodes without requiring a shared filesystem.
#
# Usage:
#   bash scripts/launch_curriculum_abc.sh
#
#   # Use smaller 8B teacher (faster, lower annotation quality)
#   bash scripts/launch_curriculum_abc.sh --teacher-model meta-llama/Meta-Llama-3-8B-Instruct
#
#   # Custom output directory (must match STEP 1)
#   OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_curriculum_abc.sh
#
# Environment:
#   OUTPUT_DIR      Base output directory (default: pipeline_output)
#   HF_TOKEN        HuggingFace API token (required for HF upload)
#   VLLM_PORT       vLLM server port (default: 8000)
#   TEACHER_MODEL   Override teacher model (default: Llama-3-70B-Instruct)
#   HF_USER         HuggingFace org (default: Beetle-Data)
#
# Compute: ~8 hrs wall-clock, 58 A100-hrs (1 node × 8 GPUs × 7.25 hrs for Stage A)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BEETLE_DATA="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BEETLE_DATA}/pipeline_output}"
HF_USER="${HF_USER:-Beetle-Data}"
VLLM_PORT="${VLLM_PORT:-8000}"
TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Meta-Llama-3-70B-Instruct}"
CONFIG="configs/beetlestream_curriculum.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher-model)
            TEACHER_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hf-user)
            HF_USER="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

cd "$BEETLE_DATA"

if [ -f "venvs/demo/bin/activate" ]; then
    source venvs/demo/bin/activate
fi

echo "============================================================"
echo "BeetleStream v2: Curriculum STEP 3 — Stages A + B + C"
echo "  Teacher model:  $TEACHER_MODEL"
echo "  vLLM port:      $VLLM_PORT"
echo "  Output dir:     $OUTPUT_DIR"
echo "  HF org:         $HF_USER"
echo "  Compute:        ~8 hrs wall-clock, 58 A100-hrs"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────
# Stage A: Teacher annotation (vLLM, 7 hrs)
# Reservoir-samples 500K documents across all languages and
# annotates with pedagogical rubric (quality, difficulty,
# vocabulary, engagement, topic) using Llama-3-70B-Instruct.
# Calibration examples from KidLM-corpus + CLC-L1-CEFR.
# ─────────────────────────────────────────────────────────────
echo ">>> Stage A: Launching vLLM server (tensor-parallel-size=8)..."
python -m vllm.entrypoints.openai.api_server \
    --model "$TEACHER_MODEL" \
    --tensor-parallel-size 8 \
    --port "$VLLM_PORT" &
VLLM_PID=$!

echo "    Waiting for vLLM server on port $VLLM_PORT..."
for i in $(seq 1 180); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "    vLLM ready after ${i}s"
        break
    fi
    sleep 1
done

echo ">>> Stage A: Running teacher annotation..."
python -m pipeline.run_pipeline --stage A \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config "$CONFIG" \
    --teacher-model "$TEACHER_MODEL" \
    --vllm-url "http://localhost:$VLLM_PORT/v1" \
    --skip-disk-check

echo ">>> Stage A: Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo ""

# ─────────────────────────────────────────────────────────────
# Stage B: Feature transform (10 min, CPU)
# Converts annotation JSONL to structured feature Parquet.
# Extracts numeric quality (0-5), difficulty (1-3), engagement (0-1).
# ─────────────────────────────────────────────────────────────
echo ">>> Stage B: Feature transform..."
python -m pipeline.run_pipeline --stage B \
    --output-dir "$OUTPUT_DIR" \
    --skip-disk-check
echo ""

# ─────────────────────────────────────────────────────────────
# Stage C: Student model training (10 min, 1 GPU)
# Trains lightweight sklearn regressors on multilingual-e5-base
# embeddings (768-dim) to approximate teacher pedagogical scores.
# ─────────────────────────────────────────────────────────────
echo ">>> Stage C: Student model training..."
python -m pipeline.run_pipeline --stage C \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config "$CONFIG" \
    --skip-disk-check
echo ""

# ─────────────────────────────────────────────────────────────
# Upload Stage A-C outputs to HuggingFace.
# This allows Stage D+3 nodes (STEP 4) to download them without
# needing a shared filesystem.
# ─────────────────────────────────────────────────────────────
echo ">>> Uploading Stage A-C outputs to HuggingFace..."
python - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

output_dir = os.environ.get("OUTPUT_DIR", "pipeline_output")
hf_user    = os.environ.get("HF_USER", "Beetle-Data")
hf_token   = os.environ.get("HF_TOKEN", "")
api        = HfApi(token=hf_token or None)

def upload_dir(local_dir: str, repo_id: str) -> None:
    p = Path(local_dir)
    if not p.exists():
        print(f"  WARNING: {local_dir} not found — skipping upload")
        return
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(p),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded {local_dir} → {repo_id}")

upload_dir(f"{output_dir}/annotations",    f"{hf_user}/beetlestream-annotations")
upload_dir(f"{output_dir}/features",       f"{hf_user}/beetlestream-annotations")
upload_dir(f"{output_dir}/student_model",  f"{hf_user}/beetlestream-student-model")
print("Upload complete.")
PYEOF

echo ""
echo "============================================================"
echo "Curriculum STEP 3 complete."
echo "  Annotations:    https://huggingface.co/$HF_USER/beetlestream-annotations"
echo "  Student model:  https://huggingface.co/$HF_USER/beetlestream-student-model"
echo ""
echo "Next: Run STEP 4 (Stages D+3) on 4 nodes:"
echo "  bash scripts/launch_curriculum_d3.sh --lang fr de es zh ja"
echo "  bash scripts/launch_curriculum_d3.sh --lang nl it ru pl tr"
echo "  bash scripts/launch_curriculum_d3.sh --lang tl hi ta eu ar"
echo "  bash scripts/launch_curriculum_d3.sh --lang sv el ca fa id"
echo "============================================================"
