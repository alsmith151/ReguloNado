#!/bin/bash
# GPU job: W&B launches one cached-feature tier-1 variant per agent process.
# Review these site-specific settings before submission.
#SBATCH --job-name=hl60-sweep
#SBATCH --partition=CHANGEME_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hl60-sweep_%A_%a.log

set -euo pipefail

: "${SWEEP_ID:?Set SWEEP_ID to the W&B entity/project/sweep ID}"
: "${CALIBRATION_CACHE:?Set CALIBRATION_CACHE to the feature cache path}"
export CALIBRATION_CACHE
wandb agent "$SWEEP_ID"
