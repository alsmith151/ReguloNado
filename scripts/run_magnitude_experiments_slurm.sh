#!/bin/bash
# Submit all magnitude-fix fine-tuning experiments as parallel SLURM jobs.
# Each runs on a single GPU and warm-starts from the same checkpoint.
#
# Required (or edit the defaults below):
#   INIT_WEIGHTS_FROM_CHECKPOINT   Path to a stage3 or later checkpoint dir
#   DATA_DIR                       Path to the prepared Arrow dataset
#
# Optional:
#   WANDB_PROJECT   W&B project name (default: regulonado)
#
# Usage:
#   bash scripts/run_magnitude_experiments_slurm.sh
#   # or override paths:
#   INIT_WEIGHTS_FROM_CHECKPOINT=/other/checkpoint DATA_DIR=/other/dataset \
#     bash scripts/run_magnitude_experiments_slurm.sh

set -euo pipefail

: "${INIT_WEIGHTS_FROM_CHECKPOINT:?INIT_WEIGHTS_FROM_CHECKPOINT must be set — path to a stage3 checkpoint dir}"
: "${DATA_DIR:?DATA_DIR must be set — path to the prepared Arrow dataset}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    exit 1
fi
if [[ ! -d "$INIT_WEIGHTS_FROM_CHECKPOINT" ]]; then
    echo "ERROR: INIT_WEIGHTS_FROM_CHECKPOINT does not exist: $INIT_WEIGHTS_FROM_CHECKPOINT" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_slurm.sh"

EXPERIMENTS=(
    # Baseline: raise Poisson weight 0.2 → 1.0, same multinomial structure
    magnitude_fix_high_poisson_weight

    # Per-bin losses with squash (LR 2e-5, numerically stable)
    magnitude_fix_log1p_huber
    magnitude_fix_poisson_nll

    # Per-bin losses without squash (LR 5e-6, raw count scale)
    magnitude_fix_log1p_huber_no_squash
    magnitude_fix_poisson_nll_no_squash

    # Composite: low multinomial + per-bin log1p MSE + top-K Huber
    magnitude_fix_transfer_calibration
)

echo "Submitting ${#EXPERIMENTS[@]} magnitude-fix experiments (1 GPU each)..."
echo "  Checkpoint : $INIT_WEIGHTS_FROM_CHECKPOINT"
echo "  Data       : $DATA_DIR"
echo ""

JOB_IDS=()
for EXP in "${EXPERIMENTS[@]}"; do
    JOB_ID=$(
        EXPERIMENT="$EXP" \
        INIT_WEIGHTS_FROM_CHECKPOINT="$INIT_WEIGHTS_FROM_CHECKPOINT" \
        DATA_DIR="$DATA_DIR" \
        WANDB_PROJECT="${WANDB_PROJECT:-regulonado}" \
        NPROC_PER_NODE=1 \
        sbatch --parsable --gres=gpu:1 "$TRAIN_SCRIPT"
    )
    JOB_IDS+=("$JOB_ID")
    echo "  Submitted $EXP → job $JOB_ID"
done

echo ""
echo "All jobs submitted: ${JOB_IDS[*]}"
echo "Monitor with: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
