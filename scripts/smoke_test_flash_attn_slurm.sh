#!/usr/bin/env bash
# Quick GPU smoke test for flash_attn + Borzoi/FlashZoi forward pass.
#
# Usage:
#   sbatch scripts/smoke_test_flash_attn_slurm.sh [pretrained_name]
#   bash   scripts/smoke_test_flash_attn_slurm.sh [pretrained_name]
#SBATCH --job-name=smoke-flash-attn
#SBATCH --output=logs/smoke-flash-attn-%j.out
#SBATCH --error=logs/smoke-flash-attn-%j.err
#SBATCH --partition=gpu-ada
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

PRETRAINED_NAME="${1:-johahi/flashzoi-replicate-0}"
PIXI_ENV_NAME="${PIXI_ENV_NAME:-default}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

cd "$REPO_DIR"
pixi run -e "$PIXI_ENV_NAME" python scripts/smoke_test_flash_attn.py "$PRETRAINED_NAME"
