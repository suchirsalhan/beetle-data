#!/bin/bash
# ============================================================================
# submit_stream_missing_28b_sl2.sh — CRSid-agnostic submitter for
# stream_missing_28b_sl2.sh.
#
# Resolves $HPC_WORK at submission time, pre-creates the SLURM log directory
# (slurmd opens -o/-e *before* the script runs, so this must exist), and
# overrides the in-script SBATCH --chdir / -o / -e with the current user's
# RDS path. /home/<crsid>/... is read-only on Wilkes3 compute nodes, so all
# writes must go to ${HPC_WORK}.
#
# Usage:
#   ./scripts/submit_stream_missing_28b_sl2.sh
#   ./scripts/submit_stream_missing_28b_sl2.sh --time=00:30:00 \
#       --export=ALL,HF_TOKEN="$HF_TOKEN",WALLCLOCK_BUDGET_SEC=600,LANGS=zh
#
# Any extra args are forwarded to sbatch.
# ============================================================================
set -euo pipefail

: "${HPC_WORK:?HPC_WORK not set — are you on a CSD3 login node?}"
: "${HF_TOKEN:?HF_TOKEN not set — export it before submitting}"

PROJECT="${HPC_WORK}/beetle-data"
LOGDIR="${PROJECT}/logs/stream_missing_28b"

if [ ! -d "${PROJECT}" ]; then
    echo "ERROR: ${PROJECT} does not exist" >&2
    exit 1
fi

mkdir -p "${LOGDIR}"

exec sbatch \
    --chdir="${PROJECT}" \
    -o "${LOGDIR}/zh-sl2-%j.out" \
    -e "${LOGDIR}/zh-sl2-%j.err" \
    --export=ALL,HF_TOKEN="${HF_TOKEN}" \
    "$@" \
    "${PROJECT}/scripts/stream_missing_28b_sl2.sh"
