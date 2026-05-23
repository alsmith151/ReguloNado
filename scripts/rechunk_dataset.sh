#!/bin/bash
#SBATCH --job-name=rechunk-dataset
#SBATCH --output=logs/rechunk-dataset-%j.out
#SBATCH --error=logs/rechunk-dataset-%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=long

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

SRC="${SRC:-/home/a/asmith/project_milne_group/software/Regulonado/dataset/2026-05-21-regulonado-v2-compressed}"
DST="${DST:-/home/a/asmith/project_milne_group/software/Regulonado/dataset/2026-05-21-regulonado-v2-rechunked}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
LEVEL="${LEVEL:-3}"
WORKERS="${WORKERS:-8}"

echo "Node     : $(hostname)"
echo "Src      : $SRC"
echo "Dst      : $DST"
echo "Batch sz : $MAX_BATCH_SIZE"
echo "ZSTD lvl : $LEVEL"
echo "Workers  : $WORKERS"
echo ""

mkdir -p "$REPO_DIR/logs"

source "$REPO_DIR/.venv/bin/activate"

python "$REPO_DIR/scripts/recompress_dataset.py" \
    "$SRC" \
    "$DST" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --level "$LEVEL" \
    --workers "$WORKERS"
