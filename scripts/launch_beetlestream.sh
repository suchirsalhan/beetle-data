#!/bin/bash
# ============================================================================
# launch_beetlestream.sh — BeetleStream v2: Full Curriculum Pipeline
#
# Runs all stages (1, 2, A, B, C, D, 3) end-to-end for curriculum mode.
# Supports multi-node execution via SLURM_NODEID or --node-id.
#
# Usage:
#   # Full pipeline on a single node (all languages)
#   bash scripts/launch_beetlestream.sh
#
#   # Specific node in SLURM cluster
#   bash scripts/launch_beetlestream.sh --node-id 0
#
#   # Specific languages
#   bash scripts/launch_beetlestream.sh --lang fr de es zh ja
#
#   # With 8B teacher model (faster, lower quality)
#   bash scripts/launch_beetlestream.sh --teacher-model meta-llama/Meta-Llama-3-8B-Instruct
#
# Environment:
#   OUTPUT_DIR    — Base output directory (default: pipeline_output)
#   HF_TOKEN      — HuggingFace API token
#   VLLM_PORT     — vLLM server port (default: 8000)
#
# Requires:
#   - 8 A100-80GB GPUs per node
#   - ~4 TB disk on shared SSD (e.g., /mnt/ssd-3)
#   - vLLM installed (pip install vllm)
# ============================================================================

set -euo pipefail

# Defaults
OUTPUT_DIR="${OUTPUT_DIR:-pipeline_output}"
VLLM_PORT="${VLLM_PORT:-8000}"
TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Meta-Llama-3-70B-Instruct}"
CONFIG="configs/beetlestream_curriculum.yaml"
NODE_ID="${SLURM_NODEID:-${1:-}}"

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-id)
            NODE_ID="$2"
            shift 2
            ;;
        --lang)
            shift
            LANG_ARGS="--lang"
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LANG_ARGS="$LANG_ARGS $1"
                shift
            done
            EXTRA_ARGS="$EXTRA_ARGS $LANG_ARGS"
            ;;
        --teacher-model)
            TEACHER_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

NODE_ARGS=""
if [[ -n "$NODE_ID" ]]; then
    NODE_ARGS="--node-id $NODE_ID"
fi

echo "============================================================"
echo "BeetleStream v2: Pedagogical Multilingual Curriculum Pipeline"
echo "============================================================"
echo "Output dir:    $OUTPUT_DIR"
echo "Teacher model: $TEACHER_MODEL"
echo "vLLM port:     $VLLM_PORT"
echo "Node ID:       ${NODE_ID:-auto}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────
# Step 1: Build benchmark index (only on node 0 or if no node specified)
# ─────────────────────────────────────────────────────────────
if [[ -z "$NODE_ID" || "$NODE_ID" == "0" ]]; then
    echo ""
    echo ">>> Step 1: Building benchmark 13-gram index..."
    python -m pipeline.run_pipeline --stage 1 \
        --project-root . --output-dir "$OUTPUT_DIR" --skip-disk-check
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Decontaminate all languages
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Decontaminating languages..."
python -m pipeline.run_pipeline --stage 2 \
    $NODE_ARGS $EXTRA_ARGS \
    --index "$OUTPUT_DIR/benchmark_13gram.pkl" \
    --output-dir "$OUTPUT_DIR" --project-root . --skip-disk-check

# Also decontaminate English if on node 0
if [[ -z "$NODE_ID" || "$NODE_ID" == "0" ]]; then
    EN_DIR="$OUTPUT_DIR/decontaminated/en"
    if [[ ! -d "$EN_DIR" ]] || [[ -z "$(ls -A "$EN_DIR"/*.parquet 2>/dev/null)" ]]; then
        echo ">>> Decontaminating English..."
        python -m pipeline.run_pipeline --stage 2 --lang en \
            --index "$OUTPUT_DIR/benchmark_13gram.pkl" \
            --output-dir "$OUTPUT_DIR" --project-root . --skip-disk-check
    fi
fi

# ─────────────────────────────────────────────────────────────
# Steps 3-5: Teacher annotation + Feature transform + Student model
# (only on node 0 — other nodes wait)
# ─────────────────────────────────────────────────────────────
if [[ -z "$NODE_ID" || "$NODE_ID" == "0" ]]; then
    echo ""
    echo ">>> Step 3: Launching vLLM and running teacher annotation..."
    echo "    Model: $TEACHER_MODEL"

    # Launch vLLM in background
    python -m vllm.entrypoints.openai.api_server \
        --model "$TEACHER_MODEL" \
        --tensor-parallel-size 8 \
        --port "$VLLM_PORT" &
    VLLM_PID=$!

    # Wait for vLLM to be ready
    echo "    Waiting for vLLM server on port $VLLM_PORT..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "    vLLM ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Run Stage A
    python -m pipeline.run_pipeline --stage A \
        --output-dir "$OUTPUT_DIR" \
        --beetlestream-config "$CONFIG" \
        --teacher-model "$TEACHER_MODEL" \
        --vllm-url "http://localhost:$VLLM_PORT/v1"

    # Stop vLLM
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true

    echo ""
    echo ">>> Step 4: Feature transform..."
    python -m pipeline.run_pipeline --stage B --output-dir "$OUTPUT_DIR"

    echo ""
    echo ">>> Step 5: Training student model..."
    python -m pipeline.run_pipeline --stage C \
        --output-dir "$OUTPUT_DIR" \
        --beetlestream-config "$CONFIG"
fi

# ─────────────────────────────────────────────────────────────
# Step 6: Score + index all languages (all nodes)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 6: Scoring and indexing languages..."
python -m pipeline.run_pipeline --stage D \
    $NODE_ARGS $EXTRA_ARGS \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config "$CONFIG"

# ─────────────────────────────────────────────────────────────
# Step 7: Pretokenize indexed shards to Arrow
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 7: Pretokenizing indexed shards..."
python -m pipeline.run_pipeline --stage 3 \
    $NODE_ARGS $EXTRA_ARGS \
    --output-dir "$OUTPUT_DIR" \
    --stream-mode curriculum

echo ""
echo "============================================================"
echo "BeetleStream v2 pipeline complete!"
echo "Indexed shards: $OUTPUT_DIR/indexed/"
echo "Pretokenized:   $OUTPUT_DIR/pretokenized/"
echo "============================================================"
