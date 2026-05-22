#!/bin/bash
#SBATCH --job-name=regulonado-arrow-bench
#SBATCH --output=logs/regulonado-arrow-bench-%j.out
#SBATCH --error=logs/regulonado-arrow-bench-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --partition=long
#SBATCH --account=default

set -euo pipefail

# Override these at submit time, e.g.
#   TRACKS=100 SAMPLES=100 STAGE=true sbatch tests/bench_arrow_build.sh
TRACKS="${TRACKS:-50}"
SAMPLES="${SAMPLES:-50}"
STAGE="${STAGE:-false}"
PROFILE="${PROFILE:-true}"
N_EXTRACT_THREADS="${N_EXTRACT_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
SIGNAL_SAMPLE_CHUNK="${SIGNAL_SAMPLE_CHUNK:-8}"
SIGNAL_TRACK_CHUNK="${SIGNAL_TRACK_CHUNK:-128}"
ARROW_BATCH_SIZE="${ARROW_BATCH_SIZE:-8}"
ARROW_COMPRESSION="${ARROW_COMPRESSION:-zstd}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

source "$REPO_DIR/.venv/bin/activate"

mkdir -p "$REPO_DIR/logs"

echo "Node              : $(hostname)"
echo "SLURM_TMPDIR      : ${SLURM_TMPDIR:-not set}"
echo "Tracks            : $TRACKS"
echo "Samples           : $SAMPLES"
echo "Stage             : $STAGE"
echo "Extract threads   : $N_EXTRACT_THREADS"
echo "Signal sample chunk: $SIGNAL_SAMPLE_CHUNK"
echo "Signal track chunk : $SIGNAL_TRACK_CHUNK"
echo "Arrow batch size   : $ARROW_BATCH_SIZE"
echo "Arrow compression  : $ARROW_COMPRESSION"
echo ""

ARGS=(
    --tracks "$TRACKS"
    --samples "$SAMPLES"
    --extract-threads "$N_EXTRACT_THREADS"
    --signal-sample-chunk "$SIGNAL_SAMPLE_CHUNK"
    --signal-track-chunk "$SIGNAL_TRACK_CHUNK"
    --arrow-batch-size "$ARROW_BATCH_SIZE"
    --arrow-compression "$ARROW_COMPRESSION"
)

[[ "$STAGE" == "true" ]] && ARGS+=(--stage)
[[ "$PROFILE" == "true" ]] && ARGS+=(--profile)

python "$SCRIPT_DIR/bench_arrow_build.py" "${ARGS[@]}"
