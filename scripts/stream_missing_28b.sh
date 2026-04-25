#!/usr/bin/env bash
# ============================================================================
# Stream + decontaminate + pretokenize Beetle-Data/{lang}-28B and
# Beetle-Data/en-for-{lang}-28B for any subset of langs.
#
# Drives pipeline.run_pipeline (Stages 1-3) and pushes to HuggingFace.
# Stages 1+2+3 are CPU/RAM-bound (no GPUs needed). Pretokenization
# auto-trains Beetle-Data/tokenizer-{lang}-en if missing.
#
# Default LANGS reflect the audit gap: zho, spa, jpn are absent (or only
# present as raw) on Beetle-Data org. Override LANGS to do other langs.
#
# Env:
#   LANGS              space-separated FineWeb-2 codes (NOT 3-letter
#                      ISO), default "zh es ja"
#   OUTPUT_DIR         working dir for parquet/arrow shards,
#                      default "pipeline_output"
#   SKIP_STAGE_1       1 to skip benchmark index build (use existing
#                      ${OUTPUT_DIR}/benchmark_13gram.pkl), default 0
#   SKIP_EN_STAGE_2    1 to skip English Stage 2 (assume already done),
#                      default 0
#   EXTRA_STAGE_ARGS   extra args appended to every run_pipeline call,
#                      e.g. "--no-upload" for a local-only dry run
#   HF_TOKEN           required for HF push (warned if unset)
#
# Usage:
#   bash scripts/stream_missing_28b.sh
#   LANGS="zh" bash scripts/stream_missing_28b.sh
#   SKIP_STAGE_1=1 SKIP_EN_STAGE_2=1 LANGS="es ja" \
#       bash scripts/stream_missing_28b.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-pipeline_output}"
SKIP_STAGE_1="${SKIP_STAGE_1:-0}"
SKIP_EN_STAGE_2="${SKIP_EN_STAGE_2:-0}"
LOG_DIR="${PROJECT_ROOT}/logs/stream_missing_28b"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Push every N raw shards from Stage 2 (default 5 in cfg). Lower → more
# frequent HF commits → better resume granularity at the cost of more API
# calls. 2 ≈ every ~100K clean docs.
export BEETLE_SHARDS_PER_UPLOAD="${BEETLE_SHARDS_PER_UPLOAD:-2}"

# Per-invocation wallclock budget (seconds). On a 12 h SLURM slot, leaving
# ~20 min headroom for SLURM teardown + any final flush is safe. Each
# `python -m pipeline.run_pipeline` invocation is wrapped with `timeout`;
# exit 124 is treated as a soft stop so the user can re-run the script.
WALLCLOCK_BUDGET_SEC="${WALLCLOCK_BUDGET_SEC:-42000}"

# Track elapsed wallclock across all invocations in this script run so the
# last language doesn't get more than WALLCLOCK_BUDGET_SEC total.
SCRIPT_START_TS="${SECONDS}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

read -ra LANGS <<<"${LANGS:-zh es ja}"
read -ra EXTRA_STAGE_ARGS <<<"${EXTRA_STAGE_ARGS:-}"

INDEX_PATH="${OUTPUT_DIR}/benchmark_13gram.pkl"

remaining_budget() {
  # Seconds left in the script-wide wallclock budget. Floors at 60 to give
  # subprocesses a chance to start; the caller decides whether to skip.
  local elapsed=$(( SECONDS - SCRIPT_START_TS ))
  local remaining=$(( WALLCLOCK_BUDGET_SEC - elapsed ))
  if (( remaining < 60 )); then
    echo 60
  else
    echo "${remaining}"
  fi
}

# Run a pipeline.run_pipeline invocation under `timeout`. Exit code 124
# (SIGTERM-on-timeout) is treated as a soft stop: the underlying stage's
# checkpoint logic flushes in-flight state, and the user can re-run this
# script to resume. Any other non-zero exit is propagated.
run_with_timeout() {
  local log="$1"
  shift
  local budget
  budget="$(remaining_budget)"
  echo "  wallclock budget for this stage: ${budget}s"
  set +e
  timeout --signal=TERM --kill-after=60s "${budget}s" \
      python -m pipeline.run_pipeline "$@" 2>&1 | tee "${log}"
  local rc="${PIPESTATUS[0]}"
  set -e
  if (( rc == 124 )); then
    echo "  stage hit wallclock budget (exit 124) — treating as soft stop;"
    echo "  re-run this script to resume from checkpoint."
    return 0
  fi
  return "${rc}"
}

run_pipeline() {
  local stage="$1"
  local lang="$2"
  local log="$3"
  shift 3
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] stage=${stage} lang=${lang}"
  echo "  log: ${log}"
  echo "============================================================"
  run_with_timeout "${log}" \
      --stage "${stage}" \
      --lang "${lang}" \
      --project-root . \
      --output-dir "${OUTPUT_DIR}" \
      --skip-disk-check \
      "$@" \
      "${EXTRA_STAGE_ARGS[@]}"
}

# Stage 1 — build 13-gram benchmark index (one-time, ~5 min).
if [[ "${SKIP_STAGE_1}" != "1" ]]; then
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] stage=1 (benchmark index)"
  echo "============================================================"
  run_with_timeout "${LOG_DIR}/stage1.log" \
      --stage 1 \
      --project-root . \
      --output-dir "${OUTPUT_DIR}" \
      --skip-disk-check \
      "${EXTRA_STAGE_ARGS[@]}"
else
  echo "skipping Stage 1 (SKIP_STAGE_1=1) — expecting ${INDEX_PATH}"
fi

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "ERROR: ${INDEX_PATH} missing — cannot run Stage 2." >&2
  exit 1
fi

# Stage 2 — decontaminate. English stream is shared across all L1s,
# so do it once unless told to skip. `--upload-raw-parquet` engages the
# per-shard HF push path keyed by BEETLE_SHARDS_PER_UPLOAD so the job
# can be killed at the wallclock boundary and resume from checkpoint.
if [[ "${SKIP_EN_STAGE_2}" != "1" ]]; then
  run_pipeline 2 en "${LOG_DIR}/stage2_en.log" \
      --index "${INDEX_PATH}" --upload-raw-parquet
else
  echo "skipping English Stage 2 (SKIP_EN_STAGE_2=1)"
fi

for LANG in "${LANGS[@]}"; do
  run_pipeline 2 "${LANG}" "${LOG_DIR}/stage2_${LANG}.log" \
      --index "${INDEX_PATH}" --upload-raw-parquet
done

# Stage 3 — pretokenize (auto-trains tokenizer if missing) and push to HF.
for LANG in "${LANGS[@]}"; do
  run_pipeline 3 "${LANG}" "${LOG_DIR}/stage3_${LANG}.log"
done

echo "============================================================"
echo "[$(date -u +%FT%TZ)] stream_missing_28b.sh: DONE"
echo "  langs: ${LANGS[*]}"
echo "  HF outputs: Beetle-Data/{${LANGS[*]// /,}}-28B"
echo "             Beetle-Data/en-for-{${LANGS[*]// /,}}-28B"
echo "============================================================"
