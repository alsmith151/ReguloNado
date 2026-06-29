#!/usr/bin/env bash
# Install flash-attn into an existing .venv.
# Targets Python 3.12 + torch 2.6 + CUDA 12.4 (cu124), where prebuilt wheels
# exist on GitHub releases. Falls back to source compilation if the wheel fetch
# fails or the version doesn't match.
#
# Usage:
#   sbatch scripts/install_flash_attn_slurm.sh            # batch submission
#   bash   scripts/install_flash_attn_slurm.sh            # interactive node
#   VENV_DIR=/path/to/.venv sbatch ...                    # override venv path
#   CUDA_MODULE=cuda/12.3 sbatch ...                      # override CUDA version
#SBATCH --job-name=install-flash-attn
#SBATCH --output=logs/install-flash-attn-%j.out
#SBATCH --error=logs/install-flash-attn-%j.err
#SBATCH --partition=gpu-ada
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"

# Resolve repo root.
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # sbatch always sets SLURM_SUBMIT_DIR to the submission directory.
    # BASH_SOURCE[0] is unreliable inside sbatch (script is copied to spool).
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

echo "Repo:        $REPO_DIR"
echo "Venv:        $VENV_DIR"
echo "CUDA module: $CUDA_MODULE"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: No Python found at $PYTHON" >&2
    exit 1
fi

# Detect installed torch/cuda to pick the right prebuilt wheel.
read -r TORCH_VER TORCH_CUDA <<< "$("$PYTHON" -c "
import torch, re
v = torch.__version__           # e.g. 2.6.0+cu124
m = re.match(r'(\d+\.\d+)', v)
cuda = re.search(r'cu(\d+)', v)
print(m.group(1) if m else '', cuda.group(1) if cuda else '')
")"

echo "torch:       $TORCH_VER  (cu$TORCH_CUDA)"

module load "$CUDA_MODULE"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
echo "CUDA_HOME:   $CUDA_HOME"

# The cuda/12.9 module ships nvc++ (NVHPC) alongside nvcc; pin g++ explicitly.
export CXX=g++
export CC=gcc
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"

# Try a prebuilt GitHub-release wheel first (saves ~40 min of compilation).
WHEEL_BASE="https://github.com/Dao-AILab/flash-attention/releases/download"
WHEEL_FILE="flash_attn-${FLASH_ATTN_VERSION}+cu${TORCH_CUDA}torch${TORCH_VER}cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
WHEEL_URL="${WHEEL_BASE}/v${FLASH_ATTN_VERSION}/${WHEEL_FILE}"

echo ""
echo "Trying prebuilt wheel: $WHEEL_FILE"
if "$PIP" install --no-deps "$WHEEL_URL" 2>/dev/null; then
    echo "Prebuilt wheel installed successfully."
else
    echo "Prebuilt wheel not available; building flash-attn from source..."
    MAX_JOBS="${MAX_JOBS:-$(nproc)}"
    export MAX_JOBS
    "$PIP" install "flash_attn>=${FLASH_ATTN_VERSION%.*}" --no-build-isolation
fi

echo ""
echo "Installed:"
"$PYTHON" -c "
import torch, flash_attn
print(f'  torch       {torch.__version__}')
print(f'  flash_attn  {flash_attn.__version__}')
print(f'  cuda avail  {torch.cuda.is_available()}')
"
