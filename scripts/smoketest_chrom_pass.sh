#!/usr/bin/env bash
#
# Quick smoke test for the chrom_pass writer on a tiny real subset of the
# production inputs. Picks chr22 (smallest human autosome) + the first N
# bigwigs so the whole pipeline runs in a few minutes — enough to validate
# the build end-to-end without burning a full compute slot.
#
# Run interactively (login node or `salloc`):
#   ./scripts/smoketest_chrom_pass.sh
#
# Override knobs via env vars, e.g.:
#   N_BW=100 CHROM=chr19 ./scripts/smoketest_chrom_pass.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- knobs (override via env) -----------------------------------------------
BED_FILE="${BED_FILE:-/project/milne_group/asmith/Projects/2025-07-19-myeloid-specific-enhancer-identification/data/external/sequences_human.bed.gz}"
FASTA_FILE="${FASTA_FILE:-/ceph/project/milne_group/shared/seqnado_reference/hg38/UCSC/sequence/hg38.fa}"
BIGWIG_LIST="${BIGWIG_LIST:-$REPO_DIR/notebooks/2026-05-20-dataset-paths.txt}"

CHROM="${CHROM:-chr22}"          # restrict to one chromosome
N_BW="${N_BW:-50}"               # how many bigwigs to use (first N from list)
N_EXTRACT_THREADS="${N_EXTRACT_THREADS:-16}"

OUT_ROOT="${OUT_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/regulonado_smoketest}"
OUTPUT_DIR="$OUT_ROOT/dataset"

# --- preflight --------------------------------------------------------------
source "$REPO_DIR/.venv/bin/activate"

if [[ ! -f "$REPO_DIR/python/regulonado/_rs.cpython-313-x86_64-linux-gnu.so" ]]; then
    echo "Native extension not built. Run: maturin develop --release" >&2
    exit 1
fi

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

# --- count intervals on the target chromosome (for log only) ----------------
if [[ "$BED_FILE" == *.gz ]]; then
    N_INTERVALS=$(zcat "$BED_FILE" | awk -v c="$CHROM" '$1 == c' | wc -l)
else
    N_INTERVALS=$(awk -v c="$CHROM" '$1 == c' "$BED_FILE" | wc -l)
fi
if [[ "$N_INTERVALS" -eq 0 ]]; then
    echo "No intervals on $CHROM in $BED_FILE" >&2
    exit 1
fi

# --- subset the bigwig list -------------------------------------------------
mapfile -t ALL_BW < "$BIGWIG_LIST"
BW_ARGS=()
n=0
for p in "${ALL_BW[@]}"; do
    p="${p%$'\r'}"; p="${p#\"}"; p="${p%\"}"; p="${p#\'}"; p="${p%\'}"
    [[ -z "$p" ]] && continue
    [[ ! -f "$p" ]] && continue   # silently skip missing
    BW_ARGS+=("--bigwig" "$p")
    n=$((n + 1))
    [[ $n -ge $N_BW ]] && break
done
if [[ $n -eq 0 ]]; then
    echo "No bigwigs found from $BIGWIG_LIST" >&2
    exit 1
fi

echo "Chromosome    : $CHROM"
echo "Intervals     : $N_INTERVALS"
echo "BigWigs       : $n (subset of $(wc -l < "$BIGWIG_LIST"))"
echo "Output dir    : $OUTPUT_DIR"
echo "Extract thr.  : $N_EXTRACT_THREADS"
echo ""

# --- run --------------------------------------------------------------------
# Use the full production interval size (524288 ctx, 32 bp bins, 6144 pred bins,
# 64 bp shift) so the smoke test exercises the same code path as a real run.
t0=$(date +%s.%N)
python -m regulonado build \
    "$BED_FILE" \
    "$FASTA_FILE" \
    "$OUTPUT_DIR" \
    "${BW_ARGS[@]}" \
    --context-length    524288 \
    --bin-size          32 \
    --n-pred-bins       6144 \
    --shift-max-bp      64 \
    --n-extract-threads "$N_EXTRACT_THREADS" \
    --arrow-batch-size  512 \
    --arrow-compression zstd \
    --strategy          chrom_pass \
    --chrom             "$CHROM" \
    --profile \
    --drop-missing \
    --overwrite \
    --split "train:fold0,fold1,fold2,fold3,fold4,fold5,fold6,fold7"
t1=$(date +%s.%N)
elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')

# --- post-build sanity check ------------------------------------------------
python - <<PY
from datasets import load_from_disk
ds = load_from_disk("$OUTPUT_DIR")
print(f"\n[smoke] DatasetDict loaded: {ds}")
for split in ds:
    n = len(ds[split])
    cols = ds[split].column_names
    row0 = ds[split][0]
    labels_shape = tuple(__import__('numpy').asarray(row0['labels']).shape)
    print(f"[smoke] {split}: {n} rows, columns={cols}, labels shape={labels_shape}")
    print(f"[smoke]   first row: index={row0['index']} local_index={row0['local_index']} interval={row0['interval']}")
PY

echo ""
echo "[smoke] wall time: ${elapsed}s"
echo "[smoke] OK"
