#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-}"
BACKBONE="${BACKBONE:-borzoi}"
HEAD="${HEAD:-transfer_mlp}"
LOSS="${LOSS:-poisson_multinomial}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/outputs/train/manual}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
BACKBONE_LR="${BACKBONE_LR:-1e-4}"
SCHEDULER="${SCHEDULER:-linear}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
UNFREEZE_BACKBONE_STAGES_FROM_OUTPUT_END="${UNFREEZE_BACKBONE_STAGES_FROM_OUTPUT_END:-${UNFREEZE_BACKBONE_BLOCKS_FROM_END:-${UNFREEZE_LAST_N_BLOCKS:-0}}}"
USE_TRACK_METADATA="${USE_TRACK_METADATA:-false}"
SHARE_CONDITION_BASE_CHANNELS="${SHARE_CONDITION_BASE_CHANNELS:-false}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "DATA_DIR must point to a saved Hugging Face dataset directory" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

exec "${PYTHON_BIN}" -m regulonado.train \
  backbone="${BACKBONE}" \
  head="${HEAD}" \
  loss="${LOSS}" \
  data.path="${DATA_DIR}" \
  output_dir="${RUN_DIR}" \
  trainer.batch_size="${BATCH_SIZE}" \
  trainer.eval_batch_size="${EVAL_BATCH_SIZE}" \
  trainer.num_workers="${NUM_WORKERS}" \
  trainer.learning_rate="${LEARNING_RATE}" \
  trainer.backbone_learning_rate="${BACKBONE_LR}" \
  trainer.scheduler="${SCHEDULER}" \
  trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.freeze_backbone="${FREEZE_BACKBONE}" \
  trainer.unfreeze_backbone_stages_from_output_end="${UNFREEZE_BACKBONE_STAGES_FROM_OUTPUT_END}" \
  model.use_track_metadata="${USE_TRACK_METADATA}" \
  model.share_condition_base_channels="${SHARE_CONDITION_BASE_CHANNELS}" \
  ${MAX_STEPS:+trainer.max_steps="${MAX_STEPS}"} \
  "$@"