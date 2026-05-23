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

# SLURM copies this script to /var/spool, so resolve the repo via SLURM_SUBMIT_DIR.
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

# --- required inputs ---------------------------------------------------------
DATA_DIR="${DATA_DIR:-/home/a/asmith/project_milne_group/software/Regulonado/dataset/2026-05-21-regulonado-v2-rechunked}"
# Checkpoints and final model land here — named by job ID so runs don't overwrite each other.
RUN_DIR="${RUN_DIR:-${REPO_DIR}/outputs/train/condition-agnostic-${SLURM_JOB_ID:-local}}"

# --- W&B logging -------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-regulonado}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${BACKBONE:-borzoi}-${HEAD:-transfer_mlp}-s${UNFREEZE_BACKBONE_STAGES:-2}-lr${LEARNING_RATE:-5e-4}-b$((${BATCH_SIZE:-12} * 2))-${SLURM_JOB_ID:-local}}"

# --- model -------------------------------------------------------------------
BACKBONE="${BACKBONE:-borzoi}"
HEAD="${HEAD:-transfer_mlp}"
LOSS="${LOSS:-poisson_multinomial}"

# --- trainer -----------------------------------------------------------------
# effective batch = batch_size × gradient_accumulation_steps × n_gpus = 12 × 1 × 2 = 24.
# With backbone mostly frozen, VRAM usage is ~75% at batch=12 on a 48 GB L40.
BATCH_SIZE="${BATCH_SIZE:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
# Each DataLoader worker decompresses one Arrow record batch at a time. With the
# default 231-sample batches that is ~8.5 GB per worker; after re-sharding with
# --max-batch-size 4 it drops to ~150 MB. Use 4 workers once the dataset has been
# re-sharded; keep at 2 if running against the original large-batch shards.
NUM_WORKERS="${NUM_WORKERS:-6}"

# Head LR is high because transfer_mlp is randomly initialised.
# Backbone LR is very low — unfrozen stages should drift minimally from pretrained weights.
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
BACKBONE_LR="${BACKBONE_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"

# Cosine decay suits small datasets better than linear — avoids a harsh cutoff.
SCHEDULER="${SCHEDULER:-cosine}"

# Warmup over ~5% of total steps; gives the randomly initialised head time to stabilise
# before backbone gradients matter.
WARMUP_STEPS="${WARMUP_STEPS:-200}"

# Streaming mode (required for LZ4-compressed Arrow) cannot count epochs.
# MAX_STEPS is mandatory. Estimate: ceil(n_train_samples / eff_batch) × n_epochs.
# With ~44k samples, eff_batch=24 (batch=12 × 2 GPUs), 5 epochs → ~9200 steps.
# Override via MAX_STEPS=<N> before sbatch.
MAX_STEPS="${MAX_STEPS:-10000}"

# No accumulation needed — batch=12 fits in VRAM and gives effective global batch=24.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"

# Log every 25 optimizer steps — frequent enough to track early loss behaviour.
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
# Evaluate on the validation split every 500 optimizer steps (~40 evals over 20k steps).
# Kept separate from LOG_EVERY_N_STEPS because running 9k-sample eval every 25 steps is
# prohibitively expensive (800 full val passes over the run).
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-500}"
# Save a checkpoint every 1000 optimizer steps (~20 saves over a 20k-step run).
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-1000}"

# --- backbone freeze ---------------------------------------------------------
# Freeze the backbone entirely; unfreeze the 2 stages nearest the output
# (transformer blocks + final_joined_convs). These are the most task-specific
# layers and benefit most from domain adaptation without destabilising the CNN trunk.
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
UNFREEZE_BACKBONE_STAGES="${UNFREEZE_BACKBONE_STAGES:-2}"

# ---------------------------------------------------------------------------

source "$REPO_DIR/.venv/bin/activate"

mkdir -p "$REPO_DIR/logs" "$RUN_DIR"

export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${WANDB_RUN_NAME}"
export WANDB_DIR="${RUN_DIR}"

echo "Node          : $(hostname)"
echo "Repo          : $REPO_DIR"
echo "Data          : $DATA_DIR"
echo "Output        : $RUN_DIR"
echo "Backbone      : $BACKBONE (freeze=${FREEZE_BACKBONE}, unfreeze_stages=${UNFREEZE_BACKBONE_STAGES})"
echo "Head          : $HEAD (condition-agnostic, transfer_mlp)"
echo "Loss          : $LOSS"
echo "LR            : head=${LEARNING_RATE}  backbone=${BACKBONE_LR}"
echo "Scheduler     : ${SCHEDULER}  warmup=${WARMUP_STEPS} steps"
echo "Epochs/steps  : max_steps=${MAX_STEPS}  (streaming: epoch counting disabled)"
echo "Batch         : train=${BATCH_SIZE}  eval=${EVAL_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  gpus=2  (effective global=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2)))"
echo "Mixed prec    : $MIXED_PRECISION"
echo "Checkpoint    : every ${CHECKPOINT_EVERY_N_STEPS} optimizer steps  →  ${RUN_DIR}"
echo "W&B           : project=${WANDB_PROJECT}  run=${WANDB_RUN_NAME}"
echo ""

OVERRIDES=(
    backbone="${BACKBONE}"
    head="${HEAD}"
    loss="${LOSS}"
    data.path="${DATA_DIR}"
    output_dir="${RUN_DIR}"
    data.streaming=true
    model.use_track_metadata=false
    model.share_condition_base_channels=false
    trainer.batch_size="${BATCH_SIZE}"
    trainer.eval_batch_size="${EVAL_BATCH_SIZE}"
    trainer.num_workers="${NUM_WORKERS}"
    trainer.learning_rate="${LEARNING_RATE}"
    trainer.backbone_learning_rate="${BACKBONE_LR}"
    trainer.weight_decay="${WEIGHT_DECAY}"
    trainer.scheduler="${SCHEDULER}"
    trainer.warmup_steps="${WARMUP_STEPS}"
    trainer.max_steps="${MAX_STEPS}"
    trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    trainer.mixed_precision="${MIXED_PRECISION}"
    trainer.gradient_clip_norm="${GRADIENT_CLIP_NORM}"
    trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}"
    trainer.eval_every_n_steps="${EVAL_EVERY_N_STEPS}"
    trainer.checkpoint_every_n_steps="${CHECKPOINT_EVERY_N_STEPS}"
    trainer.freeze_backbone="${FREEZE_BACKBONE}"
    trainer.unfreeze_backbone_stages_from_output_end="${UNFREEZE_BACKBONE_STAGES}"
    "trainer.report_to=[wandb]"
)


MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))
exec torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" -m regulonado.train "${OVERRIDES[@]}" "$@"
