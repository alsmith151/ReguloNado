#!/bin/bash
# Submit the full four-phase training pipeline as dependent Slurm jobs.
#
# Phase 1 starts immediately.  Phases 2–4 are held until the previous phase
# exits successfully, and each picks up the best checkpoint from the prior
# phase via INIT_WEIGHTS_FROM_CHECKPOINT.
#
# Phase summary:
#   1  head_only           — backbone frozen, train head to convergence
#   2  stage2_unfreeze2    — unfreeze 2 output-end backbone stages, cosine LR
#   3  stage3_deep_finetune — unfreeze 4 stages + RC augmentation, lower LR
#   4  stage4_peak_finetune — topk_additive loss, sharpen peak predictions
#
# Required:
#   DATA_DIR   Path to the prepared Arrow dataset.
#
# Optional (overrides defaults):
#   BASE_DIR        Root for all run output dirs (default: outputs/train)
#   WANDB_PROJECT   W&B project (default: regulonado)
#   NPROC_PER_NODE  GPUs per node (default: 2)
#   START_FROM_PHASE  Skip earlier phases (1–4, default: 1)
#   STOP_AFTER_PHASE  Stop after this phase (1–4, default: 4)
#   PEAK_LOSS       Loss variant for phase 4: topk_additive or topk_reweight
#                   (default: topk_additive)
#
# Usage:
#   DATA_DIR=/path/to/dataset bash scripts/train_pipeline_slurm.sh
#   DATA_DIR=/path/to/dataset BASE_DIR=/scratch/myproject bash scripts/train_pipeline_slurm.sh
#   DATA_DIR=/path/to/dataset STOP_AFTER_PHASE=3 bash scripts/train_pipeline_slurm.sh
#   DATA_DIR=/path/to/dataset START_FROM_PHASE=4 bash scripts/train_pipeline_slurm.sh

set -euo pipefail

: "${DATA_DIR:?DATA_DIR must be set — path to the prepared Arrow dataset}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

BASE_DIR="${BASE_DIR:-${REPO_DIR}/outputs/train}"
WANDB_PROJECT="${WANDB_PROJECT:-regulonado}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
START_FROM_PHASE="${START_FROM_PHASE:-1}"
STOP_AFTER_PHASE="${STOP_AFTER_PHASE:-4}"
PEAK_LOSS="${PEAK_LOSS:-topk_additive}"

# ---------------------------------------------------------------------------
# Helper: find the best (lowest eval_loss) or latest checkpoint in a run dir.
# Prints the checkpoint path, or exits with an error if none found.
# ---------------------------------------------------------------------------
best_checkpoint() {
    local run_dir="$1"
    # Prefer the checkpoint recorded as best_model_checkpoint in trainer_state.
    local state="${run_dir}/trainer_state.json"
    if [[ -f "$state" ]]; then
        local best
        best=$(python3 -c "
import json, sys
s = json.load(open('${state}'))
ckpt = s.get('best_model_checkpoint')
if ckpt:
    print(ckpt)
" 2>/dev/null || true)
        if [[ -n "$best" && -d "$best" ]]; then
            echo "$best"
            return
        fi
    fi
    # Fallback: latest checkpoint directory by step number.
    local latest
    latest=$(find "$run_dir" -maxdepth 1 -name 'checkpoint-*' -type d \
        | sort -t- -k2 -n | tail -1)
    if [[ -z "$latest" ]]; then
        echo "ERROR: no checkpoint found in $run_dir" >&2
        exit 1
    fi
    echo "$latest"
}

# ---------------------------------------------------------------------------
# Phase 1 — head-only, backbone fully frozen
# ---------------------------------------------------------------------------
P1_RUN_DIR="${BASE_DIR}/head_only_borzoi-pipeline"

if [[ "${START_FROM_PHASE}" -le 1 ]]; then
    JOB1=$(sbatch \
        --parsable \
        --export=ALL,\
EXPERIMENT=head_only_borzoi,\
DATA_DIR="${DATA_DIR}",\
RUN_DIR="${P1_RUN_DIR}",\
WANDB_PROJECT="${WANDB_PROJECT}",\
WANDB_RUN_NAME="head_only_borzoi-pipeline",\
NPROC_PER_NODE="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/train_slurm.sh")
    echo "Phase 1 submitted: job ${JOB1}  →  ${P1_RUN_DIR}"
    P2_DEP="--dependency=afterok:${JOB1}"
else
    JOB1=""
    P2_DEP=""
    echo "Phase 1 skipped (START_FROM_PHASE=${START_FROM_PHASE}), using existing: ${P1_RUN_DIR}"
fi

# ---------------------------------------------------------------------------
# Phase 2 — unfreeze 2 backbone stages, warm-start from phase 1 best ckpt.
# Runs a small inline script as the job body so it can resolve the checkpoint
# path at runtime (after phase 1 has written trainer_state.json).
# ---------------------------------------------------------------------------
P2_RUN_DIR="${BASE_DIR}/stage2_unfreeze2_borzoi-pipeline"

if [[ "${START_FROM_PHASE}" -le 2 ]]; then
    JOB2=$(sbatch \
        --parsable \
        $P2_DEP \
        --job-name=regulonado-stage2 \
        --output="${REPO_DIR}/logs/regulonado-stage2-%j.out" \
        --error="${REPO_DIR}/logs/regulonado-stage2-%j.err" \
        --time=24:00:00 \
        --mem=256G \
        --cpus-per-task=16 \
        --nodes=1 \
        --gres=gpu:2 \
        --partition=gpu-ada \
        --account=gpu \
        --wrap="
set -eu
CKPT=\$(python3 -c \"
import json, sys, os, pathlib
state = pathlib.Path('${P1_RUN_DIR}/trainer_state.json')
if state.exists():
    s = json.loads(state.read_text())
    best = s.get('best_model_checkpoint')
    if best and os.path.isdir(best):
        print(best); sys.exit(0)
import re
ckpts = sorted(pathlib.Path('${P1_RUN_DIR}').glob('checkpoint-*'),
               key=lambda p: int(re.search(r'checkpoint-(\d+)', p.name).group(1)))
print(str(ckpts[-1]))
\")
echo \"Phase 2 warm-starting from: \${CKPT}\"
EXPERIMENT=stage2_unfreeze2_borzoi \\
DATA_DIR='${DATA_DIR}' \\
RUN_DIR='${P2_RUN_DIR}' \\
WANDB_PROJECT='${WANDB_PROJECT}' \\
WANDB_RUN_NAME='stage2_unfreeze2_borzoi-pipeline' \\
NPROC_PER_NODE='${NPROC_PER_NODE}' \\
INIT_WEIGHTS_FROM_CHECKPOINT=\"\${CKPT}\" \\
bash '${SCRIPT_DIR}/train_slurm.sh'
")
    echo "Phase 2 submitted: job ${JOB2}  →  ${P2_RUN_DIR}  (depends on ${JOB1:-none})"
    P3_DEP="--dependency=afterok:${JOB2}"
else
    JOB2=""
    P3_DEP=""
    echo "Phase 2 skipped (START_FROM_PHASE=${START_FROM_PHASE}), using existing: ${P2_RUN_DIR}"
fi

# ---------------------------------------------------------------------------
# Phase 3 — deep fine-tune with RC augmentation, warm-start from phase 2.
# ---------------------------------------------------------------------------
P3_RUN_DIR="${BASE_DIR}/stage3_deep_finetune_borzoi-pipeline"
JOB3=""

if [[ "${STOP_AFTER_PHASE}" -ge 3 && "${START_FROM_PHASE}" -le 3 ]]; then
JOB3=$(sbatch \
    --parsable \
    $P3_DEP \
    --job-name=regulonado-stage3 \
    --output="${REPO_DIR}/logs/regulonado-stage3-%j.out" \
    --error="${REPO_DIR}/logs/regulonado-stage3-%j.err" \
    --time=24:00:00 \
    --mem=256G \
    --cpus-per-task=16 \
    --nodes=1 \
    --gres=gpu:2 \
    --partition=gpu-ada \
    --account=gpu \
    --wrap="
set -eu
CKPT=\$(python3 -c \"
import json, sys, os, pathlib
state = pathlib.Path('${P2_RUN_DIR}/trainer_state.json')
if state.exists():
    s = json.loads(state.read_text())
    best = s.get('best_model_checkpoint')
    if best and os.path.isdir(best):
        print(best); sys.exit(0)
import re
ckpts = sorted(pathlib.Path('${P2_RUN_DIR}').glob('checkpoint-*'),
               key=lambda p: int(re.search(r'checkpoint-(\d+)', p.name).group(1)))
print(str(ckpts[-1]))
\")
echo \"Phase 3 warm-starting from: \${CKPT}\"
EXPERIMENT=stage3_deep_finetune_borzoi \\
DATA_DIR='${DATA_DIR}' \\
RUN_DIR='${P3_RUN_DIR}' \\
WANDB_PROJECT='${WANDB_PROJECT}' \\
WANDB_RUN_NAME='stage3_deep_finetune_borzoi-pipeline' \\
NPROC_PER_NODE='${NPROC_PER_NODE}' \\
INIT_WEIGHTS_FROM_CHECKPOINT=\"\${CKPT}\" \\
bash '${SCRIPT_DIR}/train_slurm.sh'
")

    echo "Phase 3 submitted: job ${JOB3}  →  ${P3_RUN_DIR}  (depends on ${JOB2:-none})"
elif [[ "${START_FROM_PHASE}" -le 3 ]]; then
    echo "Phase 3 skipped (STOP_AFTER_PHASE=${STOP_AFTER_PHASE})"
else
    echo "Phase 3 skipped (START_FROM_PHASE=${START_FROM_PHASE}), using existing: ${P3_RUN_DIR}"
fi
P4_DEP="${JOB3:+--dependency=afterok:${JOB3}}"

# ---------------------------------------------------------------------------
# Phase 4 — peak-sharpening fine-tune with top-K loss.
# Switches from poisson_multinomial to the PEAK_LOSS variant so the model
# receives stronger gradient signal on the most active bins.
# ---------------------------------------------------------------------------
P4_RUN_DIR="${BASE_DIR}/stage4_peak_finetune_borzoi-pipeline"
JOB4=""

if [[ "${STOP_AFTER_PHASE}" -ge 4 && "${START_FROM_PHASE}" -le 4 ]]; then
    JOB4=$(sbatch \
        --parsable \
        $P4_DEP \
        --job-name=regulonado-stage4 \
        --output="${REPO_DIR}/logs/regulonado-stage4-%j.out" \
        --error="${REPO_DIR}/logs/regulonado-stage4-%j.err" \
        --time=12:00:00 \
        --mem=256G \
        --cpus-per-task=16 \
        --nodes=1 \
        --gres=gpu:2 \
        --partition=gpu-ada \
        --account=gpu \
        --wrap="
set -eu
CKPT=\$(python3 -c \"
import json, sys, os, pathlib
state = pathlib.Path('${P3_RUN_DIR}/trainer_state.json')
if state.exists():
    s = json.loads(state.read_text())
    best = s.get('best_model_checkpoint')
    if best and os.path.isdir(best):
        print(best); sys.exit(0)
import re
ckpts = sorted(pathlib.Path('${P3_RUN_DIR}').glob('checkpoint-*'),
               key=lambda p: int(re.search(r'checkpoint-(\d+)', p.name).group(1)))
print(str(ckpts[-1]))
\")
echo \"Phase 4 warm-starting from: \${CKPT}  (loss: ${PEAK_LOSS})\"
EXPERIMENT=stage4_peak_finetune_borzoi \\
DATA_DIR='${DATA_DIR}' \\
RUN_DIR='${P4_RUN_DIR}' \\
WANDB_PROJECT='${WANDB_PROJECT}' \\
WANDB_RUN_NAME='stage4_peak_finetune_borzoi-pipeline' \\
NPROC_PER_NODE='${NPROC_PER_NODE}' \\
INIT_WEIGHTS_FROM_CHECKPOINT=\"\${CKPT}\" \\
bash '${SCRIPT_DIR}/train_slurm.sh' \
    loss=${PEAK_LOSS}
")
    echo "Phase 4 submitted: job ${JOB4}  →  ${P4_RUN_DIR}  (depends on ${JOB3:-none}, loss: ${PEAK_LOSS})"
elif [[ "${START_FROM_PHASE}" -le 4 ]]; then
    echo "Phase 4 skipped (STOP_AFTER_PHASE=${STOP_AFTER_PHASE})"
else
    echo "Phase 4 skipped (START_FROM_PHASE=${START_FROM_PHASE}), using existing: ${P4_RUN_DIR}"
fi

SUBMITTED_JOBS=$(printf '%s\n' "${JOB1}" "${JOB2}" "${JOB3}" "${JOB4}" | grep -v '^$' | tr '\n' ',')
SUBMITTED_JOBS="${SUBMITTED_JOBS%,}"
echo ""
echo "Pipeline queued: ${JOB1:-[skipped]} → ${JOB2:-[skipped]} → ${JOB3:-[skipped]} → ${JOB4:-[skipped]}"
echo "Monitor with:  squeue -j ${SUBMITTED_JOBS}"
