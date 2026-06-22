# ReguloNado — developer notes for Claude

## What this repo is

Mixed Rust/Python package that builds Arrow `DatasetDict`s for training sequence-to-function
genomics models (Borzoi, Enformer etc.). The hot path reads BigWig + FASTA sources and writes
HuggingFace-compatible Arrow IPC shards without materialising a dense signal intermediate.

The default production path is the Rust `chrom_pass` writer, which scans one chromosome at a
time and writes one Arrow shard per chromosome. The older sample-batched fast path is still
available as a fallback for parity checks and benchmarking.

## Layout

```text
src/lib.rs                  PyO3 module registration for regulonado._rs
src/chrom_pass.rs           Production chromosome-pass Arrow writer
src/writers.rs              Legacy sample-batched Arrow writers and helpers
src/bigwig_io.rs            BigWig extraction and binning helpers
src/fasta.rs                FASTA sequence loading / one-hot encoding
src/schema.rs               Arrow / dataset schema helpers
python/regulonado/
  dataset.py                build_dataset / build_dataset_fast orchestration
  __main__.py               Typer CLI  →  regulonado build / scale
  scaling.py                BigWig scale-factor inference
  _rs.cpython-...so         compiled extension (do not edit)
tests/
  test_chrom_pass.py        chrom_pass parity + HF loadability coverage
  test_dataset_staging.py   staging and BigWig dedupe coverage
configs/                    Hydra YAML configs
scripts/
  build_dataset_slurm.sh   SLURM dataset build entry point
  smoketest_chrom_pass.sh   small chrom_pass smoke run
```

`python -m regulonado build ...` is the main entry point for dataset creation. It accepts either
`--bigwig` repeated per track or `--bigwig-dir`, and defaults to `--strategy chrom_pass`.

## Building the Rust extension

The `.venv` in the repo root is the canonical environment. Always use it explicitly:

```bash
export VIRTUAL_ENV=/ceph/project/milne_group/asmith/software/Regulonado/.venv
.venv/bin/maturin develop --release
```

Omitting `VIRTUAL_ENV` causes maturin to pick up a macOS interpreter and fail. The `-i` flag
does not exist in this version of maturin; use `VIRTUAL_ENV` instead.

After any change to `src/*.rs`, run `maturin develop --release` before testing Python code.

If the extension has not been rebuilt, the SLURM wrapper in `scripts/build_dataset_slurm.sh` will fail
fast when no `python/regulonado/_rs*.so` extension is present.

For editable reinstalls, `uv sync` also rebuilds the extension in the repo venv.

The debug entry points (`extract_bigwig_to_file`, `extract_all_tracks_to_dir`, etc.) are gated
behind the `debug-writers` Cargo feature and are **not** included in a normal build.  To expose
them for benchmarking individual pipeline stages:

```bash
VIRTUAL_ENV=/ceph/project/milne_group/asmith/software/Regulonado/.venv \
  .venv/bin/maturin develop --release --features debug-writers
```

## Common workflows

Scale BigWigs:

```bash
.venv/bin/python -m regulonado scale /path/to/bigwigs --output scale-factors.parquet
```

Build a dataset with the default chrom-pass strategy:

```bash
.venv/bin/python -m regulonado build intervals.bed genome.fa out/ \
  --bigwig-dir bw/ \
  --stage \
  --profile
```

Run the SLURM build wrapper with environment overrides:

```bash
OVERWRITE=true PROFILE=true sbatch scripts/build_dataset_slurm.sh
```

Discover and launch training. Experiment discovery lives in
`python/regulonado/experiments.py` and is shared by the `experiments` command,
the `train --interactive` picker, and `train`'s did-you-mean validation:

```bash
.venv/bin/regulonado experiments                       # list experiments + summaries
.venv/bin/regulonado experiments show <name>           # description + effective settings
.venv/bin/regulonado train DATASET -e <name> --slurm   # submit scripts/train_slurm.sh via sbatch
.venv/bin/regulonado train --interactive               # guided picker
```

`train --slurm` shells out to `scripts/train_slurm.sh`, passing `EXPERIMENT` /
`DATA_DIR` / `REPO_DIR` / `RUN_DIR` / `NPROC_PER_NODE` via the environment and
appending friendly/raw Hydra overrides after the script (forwarded by `"$@"`).
The local path keeps passing `+experiment=` / `data.path=` as Hydra args.

## Running tests

```bash
.venv/bin/pytest tests/
```

Useful narrower checks:

```bash
.venv/bin/pytest tests/test_chrom_pass.py
.venv/bin/pytest tests/test_dataset_staging.py
```

## Ruff / code style

Line length 100. Linting: `E`, `F`, `I`. Run `ruff check python/` before committing.

## Cargo / Rust notes

- Uses PyO3 0.23, Rayon, bigtools (BigWig), arrow2 ecosystem.
- `src/lib.rs` is now a thin module-registration layer; implementation lives in the sibling Rust
  modules under `src/`.
- The `#[pymodule]` registers both debug helpers and the production writers, including
  `chrom_pass::write_arrow_split_chrom_pass` and
  `chrom_pass::write_arrow_splits_chrom_pass`.
- When changing the Rust writer path, validate both numerical parity (`tests/test_chrom_pass.py`)
  and Python-side staging / metadata behavior (`tests/test_dataset_staging.py`).
