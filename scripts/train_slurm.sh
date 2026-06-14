#!/bin/bash
#SBATCH --job-name=regulonado-train
#SBATCH --output=logs/regulonado-train-%j.out
#SBATCH --error=logs/regulonado-train-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-ada
#SBATCH --account=gpu

set -euo pipefail

# ---------------------------------------------------------------------------
# Generic Regulonado training launcher.
#
# Required:
#   EXPERIMENT   Name of the Hydra experiment config (without .yaml extension).
#                Looked up first in python/configs/experiment/ (built-in), then
#                in scripts/experiment/ (run-specific overrides added via the
#                hydra.searchpath override below).
#   DATA_DIR     Path to the prepared Arrow dataset.
#
# Optional:
#   RUN_DIR           Output directory (default: outputs/train/<EXPERIMENT>-<JOBID>)
#   WANDB_PROJECT     W&B project name (default: regulonado)
#   WANDB_RUN_NAME    W&B run name (default: <EXPERIMENT>-<JOBID>)
#   NPROC_PER_NODE    GPUs per node (default: 2)
#
# Checkpoint fields (resume_from_checkpoint, init_weights_from_checkpoint) are
# defined in the experiment YAML, not here.  If the YAML uses ${oc.env:...}
# interpolation, you can inject them via the shell environment before sbatch.
#
# Examples:
#   DATA_DIR=/path/to/dataset EXPERIMENT=head_only_borzoi sbatch scripts/train_slurm.sh
#   DATA_DIR=... EXPERIMENT=stage2_unfreeze2_borzoi INIT_WEIGHTS_FROM_CHECKPOINT=/path/to/ckpt \
#     sbatch scripts/train_slurm.sh
# ---------------------------------------------------------------------------

: "${EXPERIMENT:?EXPERIMENT must be set — e.g. EXPERIMENT=head_only_borzoi}"
: "${DATA_DIR:?DATA_DIR must be set — path to the prepared Arrow dataset}"

if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

SCRIPTS_DIR="${REPO_DIR}/scripts"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/outputs/train/${EXPERIMENT}-${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-regulonado}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}-${SLURM_JOB_ID:-local}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

source "$REPO_DIR/.venv/bin/activate"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_DIR}/matplotlib}"
mkdir -p "$REPO_DIR/logs" "$RUN_DIR" "$MPLCONFIGDIR"

export WANDB_PROJECT
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_DIR="$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node      : $(hostname)"
echo "Repo      : $REPO_DIR"
echo "Experiment: $EXPERIMENT"
echo "Data      : $DATA_DIR"
echo "Output    : $RUN_DIR"
echo "W&B       : project=${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "Config    : ${SCRIPTS_DIR}/experiment/${EXPERIMENT}.yaml (or python/configs/experiment/)"
echo ""

OVERRIDES=(
    # Adds scripts/experiment/ to Hydra's config search path so YAML files
    # there are found alongside the built-in python/configs/experiment/ configs.
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
