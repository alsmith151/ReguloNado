#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu-ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/jupyter_%j.log

set -euo pipefail

PORT=${PORT:-8888}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$(dirname "$SCRIPT_DIR")/.venv}"

# Find a free port if the default is taken
while ss -tulpn | grep -q ":${PORT} "; do
    PORT=$((PORT + 1))
done

NODE=$(hostname -f)
echo "============================================================"
echo "Jupyter Lab running on: ${NODE}"
echo "Port: ${PORT}"
echo ""
echo "To connect, run this on your local machine:"
echo "  ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@<login-node>"
echo ""
echo "Then open: http://localhost:${PORT}"
echo "============================================================"

exec ${VENV}/bin/jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --notebook-dir="${SLURM_SUBMIT_DIR:-$HOME}"
