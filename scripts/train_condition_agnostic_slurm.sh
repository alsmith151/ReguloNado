#!/bin/bash
#SBATCH --job-name=regulonado-train
#SBATCH --output=logs/regulonado-train-%j.out
#SBATCH --error=logs/regulonado-train-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-ada
#SBATCH --account=gpu

set -euo pipefail

if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

EXPERIMENT="${EXPERIMENT:-condition_agnostic_borzoi}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/dataset/2026-05-21-regulonado-v2-rechunked}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/outputs/train/condition-agnostic-${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-regulonado}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}-${SLURM_JOB_ID:-local}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INIT_WEIGHTS_FROM_CHECKPOINT="${INIT_WEIGHTS_FROM_CHECKPOINT:-}"

source "$REPO_DIR/.venv/bin/activate"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_DIR}/matplotlib}"
mkdir -p "$REPO_DIR/logs" "$RUN_DIR" "$MPLCONFIGDIR"

export WANDB_PROJECT
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_DIR="$RUN_DIR"

OVERRIDES=(
    "+experiment=${EXPERIMENT}"
    "data.path=${DATA_DIR}"
    "output_dir=${RUN_DIR}"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" && -n "$INIT_WEIGHTS_FROM_CHECKPOINT" ]]; then
    echo "Set only one of RESUME_FROM_CHECKPOINT or INIT_WEIGHTS_FROM_CHECKPOINT" >&2
    exit 1
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    OVERRIDES+=("trainer.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "$INIT_WEIGHTS_FROM_CHECKPOINT" ]]; then
    OVERRIDES+=("trainer.init_weights_from_checkpoint=${INIT_WEIGHTS_FROM_CHECKPOINT}")
fi

echo "Node      : $(hostname)"
echo "Repo      : $REPO_DIR"
echo "Data      : $DATA_DIR"
echo "Output    : $RUN_DIR"
echo "Experiment: $EXPERIMENT"
echo "W&B       : project=${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "Resume    : ${RESUME_FROM_CHECKPOINT:-<fresh>}"
echo "Warm start: ${INIT_WEIGHTS_FROM_CHECKPOINT:-<none>}"
echo "Overrides : $*"
echo ""

MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))
if [[ "$NPROC_PER_NODE" == "1" ]]; then
    exec python -m regulonado.train "${OVERRIDES[@]}" "$@"
else
    exec torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
        -m regulonado.train "${OVERRIDES[@]}" "$@"
fi
