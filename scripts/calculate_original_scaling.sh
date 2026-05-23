#!/bin/bash
#SBATCH --job-name=regulonado-calc-scaling
#SBATCH --output=logs/regulonado-calc-scaling-%j.out
#SBATCH --error=logs/regulonado-calc-scaling-%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
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
METADATA="${METADATA:-}"

if [[ -z "$METADATA" ]]; then
    echo "ERROR: METADATA env var must point to a regulonado_metadata.json" >&2
    exit 1
fi

# Output path — defaults to scale_factors.parquet next to the metadata file.
OUTPUT="${OUTPUT:-$(dirname "$METADATA")/scale_factors.parquet}"
FORMAT="${FORMAT:-parquet}"
MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

# Clip thresholds are derived from library_size by default.  Override here if
# your data is consistently deeper/shallower than a typical 30-50 M-read run.
# Units: raw counts per 1 M mapped reads (applied after scale_factor).
#   clip_soft=7  → soft clip at 7 * (library_size / 1e6) raw counts per bin
#   clip_hard=16 → hard cap at 16 * (library_size / 1e6) raw counts per bin
# These defaults replicate the existing fallback thresholds (~348 / 796) for a
# ~50 M-read library.  Increase for deeper libraries, decrease for shallow ones.
ENRICH="${ENRICH:-true}"   # write scale_factor/clip_soft/clip_hard back to metadata JSON

# ---------------------------------------------------------------------------

source "$REPO_DIR/.venv/bin/activate"

mkdir -p "$REPO_DIR/logs"

echo "Node     : $(hostname)"
echo "Repo     : $REPO_DIR"
echo "Metadata : $METADATA"
echo "Output   : $OUTPUT"
echo "Format   : $FORMAT"
echo "Workers  : $MAX_WORKERS"
echo ""

python -m regulonado calculate-original-scaling \
    "$METADATA" \
    --output "$OUTPUT" \
    --format "$FORMAT" \
    --workers "$MAX_WORKERS"

if [[ "$ENRICH" == "true" ]]; then
    echo ""
    echo "Enriching metadata JSON with scale_factor / clip_soft / clip_hard ..."
    python -m regulonado enrich-metadata "$METADATA" "$OUTPUT"
fi
