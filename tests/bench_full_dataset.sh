#!/bin/bash
#SBATCH --job-name=regulonado-bench
#SBATCH --output=logs/regulonado-bench-%j.out
#SBATCH --error=logs/regulonado-bench-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --partition=long
#SBATCH --account=default

set -euo pipefail

# ---------------------------------------------------------------------------
# Optional overrides (set as env vars before sbatch or on the command line)
# ---------------------------------------------------------------------------
ONLY="${ONLY:-}"                  # scale | staging | threads | "" (all)
NO_STAGING="${NO_STAGING:-false}" # set true if no NVMe scratch available
SKIP_PYTHON="${SKIP_PYTHON:-true}" # Python batched strategy chokes at 2299 tracks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

source "$REPO_DIR/.venv/bin/activate"

mkdir -p "$REPO_DIR/logs"

echo "Node        : $(hostname)"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-not set}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-?}"
echo "Scratch     : ${SLURM_TMPDIR:-/tmp}"
echo ""

ARGS=()
[[ -n "$ONLY" ]]                       && ARGS+=("--only" "$ONLY")
[[ "$NO_STAGING"  == "true" ]]         && ARGS+=("--no-staging")
[[ "$SKIP_PYTHON" == "true" ]]         && ARGS+=("--skip-python")

python "$SCRIPT_DIR/bench_full_dataset.py" "${ARGS[@]}"
