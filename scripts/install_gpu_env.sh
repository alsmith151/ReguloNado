#!/usr/bin/env bash
# Install regulonado[gpu], building flash-attn from source.
# flash-attn has no prebuilt cp313 wheels, so source compilation is unavoidable.
# Loads the CUDA module for headers, then forces g++ as host compiler to
# avoid nvcc picking up nvc++ from the NVHPC suite in the same module.
#
# Usage:
#   sbatch scripts/install_gpu_env.sh              # batch submission
#   bash scripts/install_gpu_env.sh                # interactive node
#   CUDA_MODULE=cuda/12.3 sbatch ...               # override CUDA version
#SBATCH --job-name=regulonado-install-gpu
#SBATCH --output=logs/install-gpu-%j.out
#SBATCH --error=logs/install-gpu-%j.err
#SBATCH --partition=gpu-ada
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:---extra gpu}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-${USER}}"

# Resolve repo root whether submitted via sbatch or run directly.
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

echo "Repo:        $REPO_DIR"
echo "CUDA module: $CUDA_MODULE"
echo "uv extras:   $INSTALL_EXTRAS"

module load "$CUDA_MODULE"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
echo "CUDA_HOME:   $CUDA_HOME"
nvcc --version

# The cuda/12.9 module ships nvc++ (NVHPC) alongside nvcc. nvcc refuses
# to use nvc++ as host compiler with a fatal error. Pin g++ explicitly.
export CXX=g++
export CC=gcc
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"

cd "$REPO_DIR"
export UV_CACHE_DIR
# shellcheck disable=SC2206
EXTRA_ARGS=($INSTALL_EXTRAS)
uv sync "${EXTRA_ARGS[@]}"

echo ""
echo "Installed:"
.venv/bin/python -c "
import torch, flash_attn
print(f'  torch       {torch.__version__}')
print(f'  flash_attn  {flash_attn.__version__}')
print(f'  cuda avail  {torch.cuda.is_available()}')
"
