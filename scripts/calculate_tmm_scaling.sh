#!/bin/bash
#SBATCH --job-name=regulonado-tmm-scaling
#SBATCH --output=logs/regulonado-tmm-scaling-%j.out
#SBATCH --error=logs/regulonado-tmm-scaling-%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --account=default

set -euo pipefail

if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

# --- required inputs ---------------------------------------------------------
# Path to a regulonado_metadata.json written by `regulonado build`.
# The dataset splits (train/, validation/, test/) must live in the same directory.
# Run calculate_original_scaling.sh first to produce scale_factors.parquet.
METADATA="${METADATA:-}"

if [[ -z "$METADATA" ]]; then
    echo "ERROR: METADATA env var must point to a regulonado_metadata.json" >&2
    exit 1
fi

# Scale-factors parquet to read library sizes from and update in-place.
# Defaults to scale_factors.parquet next to the metadata file.
SCALE_FACTORS="${SCALE_FACTORS:-$(dirname "$METADATA")/scale_factors.parquet}"

# Which dataset split to use for TMM estimation.
SPLIT="${SPLIT:-train}"

# edgeR trimming defaults.  Rarely need changing.
TRIM_M="${TRIM_M:-0.3}"
TRIM_A="${TRIM_A:-0.05}"
MIN_COUNT="${MIN_COUNT:-1.0}"

ENRICH="${ENRICH:-true}"   # write updated scale_factor / tmm_factor back to metadata JSON

# ---------------------------------------------------------------------------

source "$REPO_DIR/.venv/bin/activate"

mkdir -p "$REPO_DIR/logs"

echo "Node          : $(hostname)"
echo "Repo          : $REPO_DIR"
echo "Metadata      : $METADATA"
echo "Scale factors : $SCALE_FACTORS"
echo "Split         : $SPLIT"
echo "Trim M / A    : $TRIM_M / $TRIM_A"
echo ""

python -m regulonado calculate-tmm-scaling \
    "$METADATA" \
    --scale-factors "$SCALE_FACTORS" \
    --split "$SPLIT" \
    --trim-m "$TRIM_M" \
    --trim-a "$TRIM_A" \
    --min-count "$MIN_COUNT"

if [[ "$ENRICH" == "true" ]]; then
    echo ""
    echo "Enriching metadata JSON with updated scale_factor / tmm_factor ..."
    python -m regulonado enrich-metadata "$METADATA" "$SCALE_FACTORS" \
        --field scale_factor \
        --field tmm_factor \
        --field clip_soft \
        --field clip_hard
fi
