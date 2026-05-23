#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-}"
BACKBONE="${BACKBONE:-borzoi}"
HEAD="${HEAD:-transfer_mlp}"
LOSS="${LOSS:-poisson_multinomial}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/outputs/train/slurm}"
TIMEOUT_MIN="${TIMEOUT_MIN:-240}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_GB="${MEM_GB:-96}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
PARTITION="${PARTITION:-gpu-ada}"
ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-1}"
SCHEDULER="${SCHEDULER:-linear}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
UNFREEZE_BACKBONE_STAGES_FROM_OUTPUT_END="${UNFREEZE_BACKBONE_STAGES_FROM_OUTPUT_END:-${UNFREEZE_BACKBONE_BLOCKS_FROM_END:-${UNFREEZE_LAST_N_BLOCKS:-0}}}"
USE_TRACK_METADATA="${USE_TRACK_METADATA:-false}"
SHARE_CONDITION_BASE_CHANNELS="${SHARE_CONDITION_BASE_CHANNELS:-false}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "DATA_DIR must point to a saved Hugging Face dataset directory" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

exec "${PYTHON_BIN}" -m regulonado.train -m \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition="${PARTITION}" \
  hydra.launcher.gpus_per_node="${GPUS_PER_NODE}" \
  hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
  hydra.launcher.mem_gb="${MEM_GB}" \
  hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
  hydra.launcher.array_parallelism="${ARRAY_PARALLELISM}" \
  hydra.job.chdir=false \
  backbone="${BACKBONE}" \
  head="${HEAD}" \
  loss="${LOSS}" \
  data.path="${DATA_DIR}" \
  output_dir="${RUN_DIR}" \
  trainer.scheduler="${SCHEDULER}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.freeze_backbone="${FREEZE_BACKBONE}" \
  trainer.unfreeze_backbone_stages_from_output_end="${UNFREEZE_BACKBONE_STAGES_FROM_OUTPUT_END}" \
  model.use_track_metadata="${USE_TRACK_METADATA}" \
  model.share_condition_base_channels="${SHARE_CONDITION_BASE_CHANNELS}" \
  "$@"