#!/bin/bash
# ============================================================================
# stream_missing_28b_sl2.sh — Cambridge CSD3 / Wilkes3 SL2 SLURM wrapper
# around scripts/stream_missing_28b.sh, sized to complete Chinese 28B
# (Stage 1 + Stage 2 en + Stage 2 zh + Stage 3 zh) in a SINGLE job.
#
# SL2 limits respected:
#   - <= 64 GPUs in use at any one time (this job uses 4 — 1 ampere node)
#   - <= 36 h wallclock per job
#
# Compute shape:
#   pipeline.run_pipeline Stages 1-3 are CPU/RAM-bound (no GPUs needed).
#   The 4 A100s on the ampere node sit idle; we want the node for its
#   128 CPU threads + ~1 TB RAM. No torchrun / srun fan-out is needed.
#
# Single-job guarantee:
#   - WALLCLOCK_BUDGET_SEC = 35 h (cumulative across stages, enforced by
#     the wrapped script's `timeout`); ~3-8 h is the realistic runtime,
#     so the budget absorbs HF push retries and transient network blips.
#   - In-job supervisor loop retries up to MAX_ATTEMPTS times on
#     non-zero exits; each retry resumes from per-shard atomic
#     checkpoints (decontaminate_stream.py, pretokenize_arrow.py).
#
# Usage:
#   sbatch scripts/stream_missing_28b_sl2.sh
#
#   # override LANGS or other env if needed:
#   sbatch --export=ALL,LANGS="zh",OUTPUT_DIR="$HPC_WORK/beetle-data" \
#          scripts/stream_missing_28b_sl2.sh
# ============================================================================
#SBATCH -J zh-28b-stream
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --chdir=/rds/user/sas245/hpc-work/beetle-data
#SBATCH -o /rds/user/sas245/hpc-work/beetle-data/logs/stream_missing_28b/zh-sl2-%j.out
#SBATCH -e /rds/user/sas245/hpc-work/beetle-data/logs/stream_missing_28b/zh-sl2-%j.err
# NOTE: --chdir / -o / -e use absolute /rds paths because /home/<crsid> is
# read-only on Wilkes3 compute nodes, and SBATCH directives don't expand env
# vars. Submit via scripts/submit_stream_missing_28b_sl2.sh to override these
# with CRSid-agnostic ${HPC_WORK} paths.

set -euo pipefail

# ── Modules (Wilkes3 ampere baseline) ───────────────────────────────────────
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

JOBID=${SLURM_JOB_ID:-local}
echo "JobID:   $JOBID"
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "PWD:     $(pwd)"
echo "Node:    ${SLURM_JOB_NODELIST:-n/a}"

# ── Project root ────────────────────────────────────────────────────────────
# Under SLURM, ${BASH_SOURCE[0]} points at /var/spool/slurm/slurmd/jobNNN/slurm_script
# (a read-only copy), NOT at scripts/stream_missing_28b_sl2.sh. So derive
# BEETLE_DATA from SLURM_SUBMIT_DIR (set by sbatch) or the current PWD (set by
# --chdir). Only fall back to BASH_SOURCE when running outside SLURM.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    BEETLE_DATA="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    BEETLE_DATA="$(pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BEETLE_DATA="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${BEETLE_DATA}"

# Defence-in-depth: if BEETLE_DATA is read-only (e.g. /home/<user>/... on
# Wilkes3 compute nodes, or the SLURM spool dir), switch to writable RDS.
WRITE_TEST="$(mktemp -u .beetle_write_test.XXXXXX)"
if ! ( : > "${WRITE_TEST}" 2>/dev/null ); then
    for candidate in \
        "${HPC_WORK:-}/beetle-data" \
        "/rds/user/${USER}/hpc-work/beetle-data" \
        "/rds/user/${SLURM_JOB_USER:-$USER}/hpc-work/beetle-data"
    do
        if [ -n "${candidate}" ] && [ -d "${candidate}" ]; then
            echo "WARN: ${BEETLE_DATA} is read-only; switching to ${candidate}" >&2
            BEETLE_DATA="${candidate}"
            cd "${BEETLE_DATA}"
            break
        fi
    done
    if ! ( : > "${WRITE_TEST}" 2>/dev/null ); then
        echo "ERROR: no writable beetle-data dir found (tried HPC_WORK, /rds/user/${USER})" >&2
        exit 1
    fi
fi
rm -f "${WRITE_TEST}"

mkdir -p logs/stream_missing_28b

# ── venv: prefer repo convention, fall back to user template layout ─────────
if   [ -f venvs/demo/bin/activate ]; then
    # shellcheck disable=SC1091
    source venvs/demo/bin/activate
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "ERROR: no venv found (expected venvs/demo or .venv)" >&2
    exit 1
fi
echo "Python:  $(which python)  ($(python --version 2>&1))"

# ── Pipeline env ────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export OUTPUT_DIR="${OUTPUT_DIR:-${HPC_WORK:-$PWD}/beetle-data/pipeline_output}"
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.cache/huggingface}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set — required for HF push." >&2
    echo "Export it before sbatch (e.g. in ~/.bashrc) or pass via" >&2
    echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" $0" >&2
    exit 1
fi

# 35 h cumulative budget shared across all stages by stream_missing_28b.sh's
# internal `timeout` wrapper. Leaves ~1 h SLURM-slot headroom for HF flush,
# supervisor retries, and SLURM teardown.
export WALLCLOCK_BUDGET_SEC="${WALLCLOCK_BUDGET_SEC:-126000}"
export BEETLE_SHARDS_PER_UPLOAD="${BEETLE_SHARDS_PER_UPLOAD:-2}"
export LANGS="${LANGS:-zh}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "LANGS=${LANGS}"
echo "WALLCLOCK_BUDGET_SEC=${WALLCLOCK_BUDGET_SEC}"
echo "BEETLE_SHARDS_PER_UPLOAD=${BEETLE_SHARDS_PER_UPLOAD}"

# ── In-job supervisor: retry on transient HF / network failures ─────────────
# Each retry resumes from per-shard atomic checkpoints, so it is cheap.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
RESTART_BACKOFF_SEC="${RESTART_BACKOFF_SEC:-30}"

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
    attempt=$(( attempt + 1 ))
    echo "================ attempt ${attempt}/${MAX_ATTEMPTS} ================"
    set +e
    bash scripts/stream_missing_28b.sh
    rc=$?
    set -e
    if (( rc == 0 )); then
        echo "stream_missing_28b.sh succeeded on attempt ${attempt}"
        exit 0
    fi
    echo "stream_missing_28b.sh failed (rc=${rc}); sleeping ${RESTART_BACKOFF_SEC}s before retry"
    sleep "${RESTART_BACKOFF_SEC}"
    # After the first attempt, the index exists — skip the 5-min rebuild.
    # Stage 2 en / zh and Stage 3 zh resume cheaply via their own
    # checkpoints + idempotent HF-push paths.
    export SKIP_STAGE_1=1
done

echo "FATAL: gave up after ${MAX_ATTEMPTS} attempts" >&2
exit 1
