#!/bin/bash
#SBATCH --job-name=regulonado-build
#SBATCH --output=logs/regulonado-build-%j.out
#SBATCH --error=logs/regulonado-build-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --account=default

set -euo pipefail

# SLURM copies this script to /var/spool, so BASH_SOURCE doesn't point to
# the repo. Resolution order: REPO_DIR env var → sbatch submit dir → script dir.
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/python/regulonado" ]]; then
    REPO_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
fi

# --- required inputs ---------------------------------------------------------
BED_FILE="${BED_FILE:-/project/milne_group/asmith/Projects/2025-07-19-myeloid-specific-enhancer-identification/data/external/sequences_human.bed.gz}"
FASTA_FILE="${FASTA_FILE:-/ceph/project/milne_group/shared/seqnado_reference/hg38/UCSC/sequence/hg38.fa}"
OUTPUT_DIR="${OUTPUT_DIR:-/ceph/project/milne_group/asmith/software/Regulonado/dataset/2026-05-21-regulonado-v1}"
BIGWIG_LIST="${BIGWIG_LIST:-$REPO_DIR/notebooks/2026-05-20-dataset-paths.txt}"

# --- build parameters --------------------------------------------------------
CONTEXT_LENGTH="${CONTEXT_LENGTH:-524288}"
BIN_SIZE="${BIN_SIZE:-32}"
N_PRED_BINS="${N_PRED_BINS:-6144}"
SHIFT_MAX_BP="${SHIFT_MAX_BP:-64}"

# Rayon thread count for parallel BigWig extraction — set to full CPU allocation.
N_EXTRACT_THREADS="${N_EXTRACT_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"

# Concurrent Arrow shard writers after each chromosome scan. With 2295 tracks,
# each full-size writer batch can hold ~9 GB labels+sequence before compression.
# 4 writers is intended for 256 GB jobs; try 6-8 with 512 GB if profiling helps.
ARROW_WRITE_THREADS="${ARROW_WRITE_THREADS:-4}"

# Arrow record batch size. The builder auto-caps this to avoid i32 offset overflow:
# safe limit = floor(2^31 / (n_tracks × n_bins)). For a 2295-track full build
# that cap is ~152 samples/batch. Setting higher is fine; it gets capped with a
# log warning. Keep it generous so smaller builds stay uncapped.
ARROW_BATCH_SIZE="${ARROW_BATCH_SIZE:-512}"
ARROW_COMPRESSION="${ARROW_COMPRESSION:-lz4}"

# --- rechunk / recompress parameters -----------------------------------------
# Run recompress_dataset.py after building to rechunk into small batches and
# recompress with ZSTD.  Output lands at RECHUNK_DST (distinct from OUTPUT_DIR).
RECHUNK="${RECHUNK:-true}"
RECHUNK_DST="${RECHUNK_DST:-${OUTPUT_DIR}-rechunked}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
RECHUNK_WORKERS="${RECHUNK_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
# Set to true to delete the raw build output after successful recompression.
RECHUNK_REMOVE_SRC="${RECHUNK_REMOVE_SRC:-false}"

# --- I/O flags ---------------------------------------------------------------
# STAGE=true copies FASTA + BigWigs to SLURM_TMPDIR before building.
# Strongly recommended on Ceph: profiling shows read_bin is I/O-bound on Ceph
# (thread/wall ratio ~7%), making staging the single biggest performance lever.
# Requires SLURM_TMPDIR to have enough free space (roughly n_tracks × avg_bw_size).
STAGE="${STAGE:-true}"
DROP_MISSING="${DROP_MISSING:-true}"
DEDUPE_TRACKS="${DEDUPE_TRACKS:-content}"
OVERWRITE="${OVERWRITE:-false}"
PROFILE="${PROFILE:-false}"

# Scratch root for Arrow output and optional staging.
# SLURM_TMPDIR is per-job local SSD (auto-cleaned); preferred over Ceph.
# Falls back to TMPDIR, then to a subdir of OUTPUT_DIR for interactive runs.
export SLURM_TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-$OUTPUT_DIR/.scratch}}"

# ---------------------------------------------------------------------------

source "$REPO_DIR/.venv/bin/activate"

mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR" "$SLURM_TMPDIR"

if [[ ! -f "$REPO_DIR/python/regulonado/_rs.cpython-313-x86_64-linux-gnu.so" ]]; then
    echo "Missing compiled Rust extension. Rebuild with: uv pip install -e $REPO_DIR" >&2
    exit 1
fi

# Build --bigwig args from paths file (strip CR, quotes, blank lines)
mapfile -t BW_PATHS < "$BIGWIG_LIST"
BW_ARGS=()
for p in "${BW_PATHS[@]}"; do
    p="${p%$'\r'}"
    p="${p#\"}"; p="${p%\"}"
    p="${p#\'}"; p="${p%\'}"
    [[ -z "$p" ]] && continue
    BW_ARGS+=("--bigwig" "$p")
done

N_TRACKS=$(( ${#BW_ARGS[@]} / 2 ))

echo "Node              : $(hostname)"
echo "Repo              : $REPO_DIR"
echo "BED               : $BED_FILE"
echo "FASTA             : $FASTA_FILE"
echo "BigWig list       : $BIGWIG_LIST"
echo "Tracks            : $N_TRACKS"
echo "Output            : $OUTPUT_DIR"
echo "Scratch (TMPDIR)  : $SLURM_TMPDIR"
echo "Extract threads   : $N_EXTRACT_THREADS"
echo "Arrow write thrds : $ARROW_WRITE_THREADS"
echo "Stage to scratch  : $STAGE"
echo "Dedupe tracks     : $DEDUPE_TRACKS"
echo "Shift max bp      : $SHIFT_MAX_BP"
echo "Arrow batch       : $ARROW_BATCH_SIZE (auto-capped if tracks×bins exceeds i32)"
echo "Arrow compression : $ARROW_COMPRESSION"
echo "Profile           : $PROFILE"
echo "Rechunk           : $RECHUNK"
if [[ "$RECHUNK" == "true" ]]; then
echo "Rechunk dst       : $RECHUNK_DST"
echo "ZSTD level        : $ZSTD_LEVEL"
echo "Max batch size    : $MAX_BATCH_SIZE"
echo "Rechunk workers   : $RECHUNK_WORKERS"
echo "Remove src        : $RECHUNK_REMOVE_SRC"
fi
echo ""

ARGS=(
    "$BED_FILE"
    "$FASTA_FILE"
    "$OUTPUT_DIR"
    "${BW_ARGS[@]}"
    --context-length    "$CONTEXT_LENGTH"
    --bin-size          "$BIN_SIZE"
    --n-pred-bins       "$N_PRED_BINS"
    --shift-max-bp      "$SHIFT_MAX_BP"
    --n-extract-threads "$N_EXTRACT_THREADS"
    --arrow-write-threads "$ARROW_WRITE_THREADS"
    --arrow-batch-size  "$ARROW_BATCH_SIZE"
    --arrow-compression "$ARROW_COMPRESSION"
    --dedupe-tracks     "$DEDUPE_TRACKS"
)

[[ "$STAGE"        == "true" ]] && ARGS+=(--stage)
[[ "$DROP_MISSING" == "true" ]] && ARGS+=(--drop-missing)
[[ "$OVERWRITE"    == "true" ]] && ARGS+=(--overwrite)
[[ "$PROFILE"      == "true" ]] && ARGS+=(--profile)

python -m regulonado build "${ARGS[@]}"

if [[ "$RECHUNK" == "true" ]]; then
    echo ""
    echo "=== Rechunking / recompressing to ZSTD ==="
    echo "  src : $OUTPUT_DIR"
    echo "  dst : $RECHUNK_DST"

    RECOMPRESS_ARGS=(
        "$OUTPUT_DIR"
        "$RECHUNK_DST"
        --level   "$ZSTD_LEVEL"
        --workers "$RECHUNK_WORKERS"
        --max-batch-size "$MAX_BATCH_SIZE"
    )
    [[ "$RECHUNK_REMOVE_SRC" == "true" ]] && RECOMPRESS_ARGS+=(--remove-src)

    python "$REPO_DIR/scripts/recompress_dataset.py" "${RECOMPRESS_ARGS[@]}"
fi
