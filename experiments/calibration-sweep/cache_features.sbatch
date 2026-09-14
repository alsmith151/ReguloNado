#!/bin/bash
# GPU job: run the trunk feature/cache step on a compute node.
# Review these site-specific settings before submission.
#SBATCH --job-name=hl60-cache
#SBATCH --partition=CHANGEME_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hl60-cache_%j.log

set -euo pipefail

: "${CALIBRATION_INPUT:?Set CALIBRATION_INPUT to the input tensor payload}"
: "${CALIBRATION_OUTPUT:?Set CALIBRATION_OUTPUT to the output cache path}"
python experiments/calibration-sweep/cache_features.py \
  --input "$CALIBRATION_INPUT" \
  --output "$CALIBRATION_OUTPUT"
