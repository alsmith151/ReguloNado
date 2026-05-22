# Regulonado — developer notes for Claude

## What this repo is

Mixed Rust/Python package that builds Arrow `DatasetDict`s for training sequence-to-function
genomics models (Borzoi, Enformer etc.). The hot path reads BigWig + FASTA sources and writes
HuggingFace-compatible Arrow IPC shards without materialising a dense signal intermediate.

## Layout

```
src/lib.rs                  Rust hot path (PyO3 extension, installed as regulonado._rs)
python/regulonado/
  dataset.py                build_dataset / build_dataset_fast (main Python API)
  __main__.py               Typer CLI  →  regulonado build / scale
  scaling.py                BigWig scale-factor inference
  _rs.cpython-313-...so     compiled extension (do not edit)
tests/                      pytest suite
configs/                    Hydra YAML configs
scripts/                    SLURM job scripts (e.g. create_dataset.sh)
```

## Building the Rust extension

The `.venv` in the repo root is the canonical environment. Always use it explicitly:

```bash
export VIRTUAL_ENV=/ceph/project/milne_group/asmith/software/Regulonado/.venv
.venv/bin/maturin develop --release
```

Omitting `VIRTUAL_ENV` causes maturin to pick up a macOS interpreter and fail. The `-i` flag
does not exist in this version of maturin; use `VIRTUAL_ENV` instead.

After any change to `src/lib.rs`, run `maturin develop --release` before testing Python code.

## Running tests

```bash
.venv/bin/pytest tests/
```

## Key entry points

| Symbol | File | Purpose |
|---|---|---|
| `build_dataset_fast` | dataset.py:632 | main dataset builder; prefer over legacy path |
| `write_arrow_split_from_bigwigs` | lib.rs:1283 | Rust core: BigWig→Arrow per split |
| `extract_bigwig_labels_batch` | lib.rs:939 | Rayon-parallel BigWig read for one batch |
| `bin_region_into` | lib.rs:51 | binning kernel |

## Profiling (added 2026-05-21)

Pass `--profile` to the CLI (or `profile=True` to `build_dataset_fast`) to emit a per-phase
timing summary to stderr after each split:

```
[regulonado_rs] profile batches=275
  opens:    total_thread_s=X.X  per_batch_ms p50=Y p95=Z
  read_bin: total_thread_s=...  ...
  assemble: total_wall_s=...    ...
  fasta:    total_wall_s=...    ...
  arrow:    total_wall_s=...    ...
```

`opens` and `read_bin` are **summed thread-time** across Rayon cores; `assemble`, `fasta`, and
`arrow` are wall-clock. When profiling is off the instrumentation is entirely zero-cost (no
atomics, no branches in the hot loop).

Decision table for the next optimisation phase:

| Phase dominates | Likely fix |
|---|---|
| `opens` | Hoist BigWigRead handles above the batch loop |
| `read_bin` high thread/wall ratio | CPU-bound on binning — SIMD / zoom levels |
| `read_bin` low thread/wall ratio | Ceph I/O wait — sort intervals or use `--stage` |
| `arrow` | Drop to `zstd-1` / `lz4`; confirm output is on local scratch |
| `fasta` | Cache FASTA index; mmap |

## Scratch paths and Ceph I/O

The builder logs a warning if `scratch_out` or `output_dir` resolves under `/ceph` or
`/project`, because Arrow I/O to Ceph is ~10× slower than to `$SLURM_TMPDIR`.

Set `$SLURM_TMPDIR` (automatically set on SLURM nodes) to get fast local scratch. When unset,
`/tmp` is used. Pass `--stage` to also copy FASTA and BigWigs to local scratch before building.

## Arrow batch size cap

The builder automatically caps `arrow_batch_size` to avoid Arrow i32 offset overflow
(`batch × n_tracks × n_bins > 2^31`). A warning is logged when the cap fires. For typical
builds (2295 tracks, 6144 bins) the safe limit is 152 samples/batch.

## Shift augmentation

`shift_max_bp` must be a multiple of `bin_size`. When non-zero, the stored interval is
`context_length + 2 × shift_max_bp` bp wide and `stored_n_bins = n_pred_bins + 2 × shift_bins`.
The model then samples random sub-windows at training time.

## Ruff / code style

Line length 100. Linting: `E`, `F`, `I`. Run `ruff check python/` before committing.

## Cargo / Rust notes

- Uses PyO3 0.23, Rayon, bigtools (BigWig), arrow2 ecosystem.
- `src/lib.rs` is a single file; all PyO3 functions are registered in the `#[pymodule]` at the
  bottom.
- `Instant::now()` / `Duration` are already imported at the top of lib.rs.
