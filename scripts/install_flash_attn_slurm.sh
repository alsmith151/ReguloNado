#!/usr/bin/env bash
# Install flash-attn into the current project's environment.
# Prefers Pixi env when pixi.toml is present, falls back to .venv.
#
# Usage:
#   sbatch scripts/install_flash_attn_slurm.sh
#   bash scripts/install_flash_attn_slurm.sh
#   PIXI_ENV_NAME=default sbatch scripts/install_flash_attn_slurm.sh
#   CUDA_MODULE=cuda/12.9 sbatch scripts/install_flash_attn_slurm.sh
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
LOCAL_TMP_BASE="${LOCAL_TMP_BASE:-/tmp/${USER}/flash-attn}"

# Resolve repo root.
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # sbatch sets SLURM_SUBMIT_DIR to the submission directory.
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"
PIXI_ENV_NAME="${PIXI_ENV_NAME:-default}"
PREFER_PIXI="${PREFER_PIXI:-1}"
USE_PIXI=0

echo "Repo:        $REPO_DIR"
echo "Venv:        $VENV_DIR"
echo "CUDA module: $CUDA_MODULE"

# Keep pip/pixi temp/cache on local disk to avoid cross-device rename errors.
# Use *_OVERRIDE vars for explicit customization; otherwise force local defaults.
mkdir -p "$LOCAL_TMP_BASE/pip-cache" "$LOCAL_TMP_BASE/pip-tmp" "$LOCAL_TMP_BASE/pixi-cache"
export TMPDIR="${TMPDIR_OVERRIDE:-${TMPDIR:-$LOCAL_TMP_BASE/pip-tmp}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_OVERRIDE:-$LOCAL_TMP_BASE/pip-cache}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR_OVERRIDE:-$LOCAL_TMP_BASE/pixi-cache}"
echo "TMPDIR:      $TMPDIR"
echo "PIP_CACHE:   $PIP_CACHE_DIR"
echo "PIXI_CACHE:  $PIXI_CACHE_DIR"

# Prefer Pixi env in current project dir when requested and available.
if [[ "$PREFER_PIXI" == "1" ]] && command -v pixi >/dev/null 2>&1 && [[ -f "$REPO_DIR/pixi.toml" ]]; then
    USE_PIXI=1
    echo "Using Pixi environment '$PIXI_ENV_NAME' in $REPO_DIR"
elif [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: No Python found at $PYTHON and no pixi.toml at $REPO_DIR" >&2
    exit 1
fi

# Detect installed torch/cuda to pick the right prebuilt wheel.
if [[ "$USE_PIXI" -eq 1 ]]; then
    read -r TORCH_VER TORCH_CUDA TORCH_ABI <<< "$(cd "$REPO_DIR" && pixi run -e "$PIXI_ENV_NAME" python -c "
import torch, re
v = torch.__version__
m = re.match(r'(\d+\.\d+)', v)
cuda = re.search(r'cu(\d+)', v)
abi = getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', None)
print(m.group(1) if m else '', cuda.group(1) if cuda else '', 'TRUE' if abi else 'FALSE')
")"
else
    read -r TORCH_VER TORCH_CUDA TORCH_ABI <<< "$("$PYTHON" -c "
import torch, re
v = torch.__version__
m = re.match(r'(\d+\.\d+)', v)
cuda = re.search(r'cu(\d+)', v)
abi = getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', None)
print(m.group(1) if m else '', cuda.group(1) if cuda else '', 'TRUE' if abi else 'FALSE')
")"
fi

echo "torch:       $TORCH_VER  (cu$TORCH_CUDA, cxx11abi=$TORCH_ABI)"

module load "$CUDA_MODULE"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
echo "CUDA_HOME:   $CUDA_HOME"

# cuda module may ship nvc++; pin GNU toolchain for nvcc host compiler.
export CXX=g++
export CC=gcc
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"

pip_install() {
    if [[ "$USE_PIXI" -eq 1 ]]; then
        (cd "$REPO_DIR" && pixi run -e "$PIXI_ENV_NAME" python -m pip "$@")
    else
        "$PIP" "$@"
    fi
}

echo "Building flash-attn from source against the active torch install..."
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export MAX_JOBS
pip_install install --no-cache-dir --force-reinstall --no-deps "flash_attn==${FLASH_ATTN_VERSION}" --no-build-isolation

echo ""
echo "Installed:"
if [[ "$USE_PIXI" -eq 1 ]]; then
    cd "$REPO_DIR"
    pixi run -e "$PIXI_ENV_NAME" python -c "
import torch, flash_attn
print(f'  torch       {torch.__version__}')
print(f'  flash_attn  {flash_attn.__version__}')
print(f'  cuda avail  {torch.cuda.is_available()}')
"
else
    "$PYTHON" -c "
import torch, flash_attn
print(f'  torch       {torch.__version__}')
print(f'  flash_attn  {flash_attn.__version__}')
print(f'  cuda avail  {torch.cuda.is_available()}')
"
fi
