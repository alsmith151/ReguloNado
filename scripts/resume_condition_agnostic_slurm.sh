#!/bin/bash
#SBATCH --job-name=regulonado-resume
#SBATCH --output=logs/regulonado-resume-%j.out
#SBATCH --error=logs/regulonado-resume-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-ada
#SBATCH --account=gpu

set -euo pipefail

# All training hyperparameters live in scripts/experiment/resume_condition_agnostic_borzoi.yaml.
# This script only handles environment setup and injects that config via hydra.searchpath.

if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

SCRIPTS_DIR="${REPO_DIR}/scripts"
EXPERIMENT="${EXPERIMENT:-resume_condition_agnostic_borzoi}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/dataset/2026-05-21-regulonado-v2-rechunked}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/outputs/train/condition-agnostic-resume-${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-regulonado}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}-${SLURM_JOB_ID:-local}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

source "$REPO_DIR/.venv/bin/activate"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_DIR}/matplotlib}"
mkdir -p "$REPO_DIR/logs" "$RUN_DIR" "$MPLCONFIGDIR"

export WANDB_PROJECT
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_DIR="$RUN_DIR"

echo "Node      : $(hostname)"
echo "Repo      : $REPO_DIR"
echo "Data      : $DATA_DIR"
echo "Output    : $RUN_DIR"
echo "Experiment: $EXPERIMENT"
echo "W&B       : project=${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "Config    : ${SCRIPTS_DIR}/experiment/${EXPERIMENT}.yaml"
echo ""

OVERRIDES=(
    # Add scripts/ to Hydra's search path so it finds experiment/ configs there.
    "+hydra.searchpath=[file://${SCRIPTS_DIR}]"
    "+experiment=${EXPERIMENT}"
    "data.path=${DATA_DIR}"
    "output_dir=${RUN_DIR}"
)

MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))
if [[ "$NPROC_PER_NODE" == "1" ]]; then
    exec python -m regulonado.train "${OVERRIDES[@]}" "$@"
else
    exec torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
        -m regulonado.train "${OVERRIDES[@]}" "$@"
fi
