# Regulonado Architecture

## Overview

Regulonado is a mixed Rust/Python package for building Arrow datasets from BigWig and FASTA sources, optimized for training sequence-to-function genomics models (Borzoi, Enformer, etc.). The design principle is simple: hot paths that manipulate dense data and I/O go in Rust and release the GIL; orchestration, CLI, config, and model training remain in Python.

## Rust vs Python: What lives where and why

### Rust (`src/`)

The production hot path is **entirely in Rust**. Its centerpiece is the chromosome-pass writer:

- **`chromosome_scan_writer.rs`** — the production Arrow writer. For each chromosome, it decodes the binned signal of all tracks once into an in-RAM `(n_tracks, n_chrom_bins)` matrix, then slices per-sample rows out of it. This collapses ~N_samples random BigWig seeks per chromosome into one sequential pass per (chrom, track) pair. It releases the GIL and fans out over tracks with Rayon, and over shard writers with a second thread pool. A Python port using pybigtools would have to re-serialise on the GIL to hold and slice that matrix — this is the exact reason this is not pure Python.
- **`sample_batch_writer.rs`** — the sample-batched fallback writer (`write_arrow_split_from_bigwigs`), kept for parity testing.
- **`binning.rs`** — interval-to-bin accumulation. Holds reusable scratch buffers (`BinningScratch`) to avoid repeated allocations in tight loops.
- **`fasta.rs`** — `.fai` index parsing and one-hot sequence loading. Computes byte offsets directly from samtools-style FASTA indices; avoids Python and avoids constructing a GenomeIntervalDataset during Arrow writing.
- **`arrow_schema.rs`** — Arrow schema construction and HuggingFace metadata. Builds HF-compatible nested-list Arrow types with extension metadata for `Array2D` features.
- **`io_utils.rs`** — progress logging, IPC write options, Rayon pool configuration. Exposes Rayon thread count control without rebuilding the global pool (which can only be built once per process).
- **`bigwig_io.rs`** — BigWig extraction and interval iteration via the `bigtools` crate.

All error handling is via `Result<_, String>` funnelled into `PyRuntimeError`. Python callers cannot distinguish a missing FASTA index from an out-of-range BED row from an Arrow overflow — consider this a known rough edge.

### Python (`python/regulonado/`)

Everything except the hot path:

- **`dataset/build.py`** — dataset construction, staging, metadata assembly, and Rust writer orchestration.
- **`cli/app.py`** — Typer CLI entry point, with grouped `normalization` and API-backed `pipeline` commands.
- **`training/runner.py`** — PyTorch/HuggingFace trainer. Hydra orchestration, checkpoint resumption, logging.
- **`inference.py`** — inference engine. Converts model predictions to BigWig tracks with bin collapsing.
- **`normalization.py`** — BigWig scale-factor inference (RPKM, TMM, clip thresholds).
- **`metrics.py`** — evaluation metrics for training callbacks.
- **`recompress.py`** — utility to rechunk/recompress Arrow shards with different compression.

Training infrastructure:

- **`python/regulonado/model/`** — backbone loaders, task-specific heads, transfer adapters, and model wrapper.
- **`python/regulonado/training/`** — loss functions, training metrics, callbacks, data loading transforms, and training config.

Configuration:

- **`python/configs/train.yaml`** — base training config.
- **`python/configs/experiment/`** — per-experiment overrides (e.g. `condition_agnostic_borzoi.yaml`).
- **`python/configs/backbone/`** — model selection (e.g. borzoi, enformer, mistral).
- **`python/configs/head/`** — task head definitions.
- **`python/configs/loss/`** — loss function configs.

The only production `#[pyfunction]` is `write_arrow_splits_chrom_pass` called from `dataset.py`. The fallback writer `write_arrow_split_from_bigwigs` is also exposed for parity testing.

### Extension binding

- **Language**: PyO3 0.23
- **Build tool**: maturin (≥1.7,<2.0)
- **Installation**: compiled as `regulonado._rs` inside the `regulonado` package
- **Mixed layout**: `src/` for Rust, `python/` for Python packages; maturin orchestrates both

## Directory tree

```
src/
  lib.rs                           PyO3 module registration
  chromosome_scan_writer.rs       Production chromosome-scan writer
  sample_batch_writer.rs           Sample-batched fallback + feature-gated debug writers
  bigwig_io.rs                     BigWig extraction and binning
  binning.rs                       Interval→bin accumulation and scratch buffers
  fasta.rs                         FASTA index and one-hot encoding
  arrow_schema.rs                  Arrow schema / HuggingFace metadata
  io_utils.rs                      Progress logging, IPC options, Rayon pool config

python/regulonado/
  __init__.py
  __main__.py                      Typer CLI entry point
  dataset.py                       build_dataset_fast orchestration
  train.py                         PyTorch Lightning trainer + Hydra integration
  predict.py                       Inference → BigWig conversion
  scaling.py                       Scale-factor inference (RPKM, TMM, clipping)
  metrics.py                       Evaluation metrics for training callbacks
  recompress.py                    Arrow rechunking and recompression

  model/
    __init__.py
    config.py                      Model config / registry
    wrapper.py                      Unified model wrapper
    heads.py                        Task-specific prediction heads
    adapters.py                    Transfer learning adapters (LoRA, etc.)

  training/
    __init__.py
    config.py                      Training config structures
    losses.py                       Loss function implementations
    metrics.py                      Training metrics
    callbacks.py                   Training callbacks (logging, plotting)
    transforms.py                  Data loading augmentations
    data.py                        Data pipeline config
    provenance.py                  Dataset / model provenance tracking

python/configs/
  train.yaml                       Base training config (Hydra)
  experiment/                      Per-experiment overrides
    condition_agnostic_borzoi.yaml  Example experiment config
  backbone/                        Backbone model selection configs
  head/                            Task head configs
  loss/                            Loss function configs

tests/
  test_chrom_pass.py              Chrom-pass parity and HF loadability
  test_dataset_staging.py         Staging and BigWig deduplication
  test_losses.py                  Loss function numerics
  test_model_transfer.py          Transfer learning adapters
  test_predict_bigwig.py          End-to-end prediction → BigWig
  test_train_metrics.py           Training metrics

workflow/
  Snakefile                        Snakemake orchestration
  rules/                           Rule files
  profiles/slurm/                  Cluster-specific settings
  scripts/                         Helper scripts
  config.yaml                      Pipeline configuration

Cargo.toml                         Rust package metadata, features
pyproject.toml                     Python package metadata, extras, maturin config
CLAUDE.md                          Developer notes (this repo)
```

## Where do I add X?

| Task | Location | Notes |
|------|----------|-------|
| New CLI command | `python/regulonado/__main__.py` | Add a Typer `@app.command()` |
| New loss function | `python/regulonado/training/losses.py` + `python/configs/loss/*.yaml` | Implement class, add config |
| Change Arrow output schema | `src/chromosome_scan_writer.rs`, `src/arrow_schema.rs` | Rebuild with `maturin develop --release`, run tests/test_chrom_pass.py |
| New BigWig feature extraction | `src/bigwig_io.rs` | Rebuild extension |
| New training metric | `python/regulonado/training/metrics.py` | Called during training callbacks |
| Training callback (plots, logging) | `python/regulonado/training/callbacks.py` | Integrated with PyTorch Lightning |
| Model architecture / head | `python/regulonado/model/heads.py` | Register in `model/config.py` |
| Transfer adapter (LoRA, prefix-tuning) | `python/regulonado/model/adapters.py` | Use peft library |
| Snakemake workflow rule | `workflow/rules/` | Sourced by main `Snakefile` |
| Experiment config override | `python/configs/experiment/` | Use with `--experiment myexp` |

## Known rough edges

1. **Error handling** — All Rust errors are `Result<_, String>` funnelled into `PyRuntimeError`. Python callers cannot distinguish root causes (missing FASTA index vs out-of-range BED row vs Arrow overflow). Consider wrapping with more structured exceptions if debugging becomes painful.

2. **Duplicate Arrow assembly** — `sample_batch_writer.rs` duplicates some of `chromosome_scan_writer.rs`'s record-batch construction logic. Refactoring to share the batch-building code would reduce maintenance burden but requires careful handling of the different I/O patterns.

3. **Rayon pool rebuilding** — Rayon's global thread pool can only be configured once per process. When `write_arrow_splits_chrom_pass` is called multiple times from the same interpreter, the `n_threads` parameter on the second call is silently ignored. This is unavoidable but now logs a warning.

## Building and testing

### Rebuild the Rust extension

```bash
export VIRTUAL_ENV=$PWD/.venv
.venv/bin/maturin develop --release
```

The `VIRTUAL_ENV` environment variable is essential — omitting it causes maturin to pick a macOS interpreter and fail. There is no `-i` flag in this maturin version.

After any change to `src/*.rs`, re-run `maturin develop --release` before testing Python code.

### Test

```bash
.venv/bin/pytest tests/
```

Full suite is 97 tests. Key narrower checks:

```bash
.venv/bin/pytest tests/test_chrom_pass.py        # Chrom-pass parity
.venv/bin/pytest tests/test_dataset_staging.py   # Staging and BigWig deduplication
.venv/bin/pytest tests/test_losses.py            # Loss function numerics
```

## Reverse mapping: Rust ↔ Python call sites

| Rust module | Called from Python | Purpose |
|-------------|-------------------|---------|
| `chrom_pass::write_arrow_splits_chrom_pass` | `dataset.py:build_dataset_fast` | Production dataset write |
| `writers::write_arrow_split_from_bigwigs` | test suite | Parity checking |
| All others (binning, fasta, schema, io_utils) | called by the above | Supporting routines |
