# Contributing

## Set up the repository

ReguloNado supports Python 3.12 and 3.13 and requires Rust 1.85 or newer when
building the extension from source.

```bash
git clone https://github.com/alsmith151/ReguloNado.git
cd ReguloNado
uv sync --extra dev --extra train --extra workflow
source .venv/bin/activate
```

Build the Rust extension after changing anything under `src/`:

```bash
export VIRTUAL_ENV="$PWD/.venv"
.venv/bin/maturin develop --release
```

Setting `VIRTUAL_ENV` keeps maturin on the repository interpreter. `uv sync`
also rebuilds the editable extension when needed.

## Find the right area

| Change | Location |
| --- | --- |
| CLI behavior | `python/regulonado/cli/` |
| Dataset construction | `python/regulonado/dataset/` and `src/` |
| Training, loss, or metrics | `python/regulonado/training/` |
| Model heads or adapters | `python/regulonado/model/` |
| Training defaults and presets | `python/configs/` |
| Pipeline rules and Slurm profile | `python/regulonado/workflow/` |
| Prediction | `python/regulonado/inference.py` |

The packaged workflow directory is the only workflow source. Do not add a
second top-level copy.

## Check a change

```bash
.venv/bin/python -m pytest tests/
.venv/bin/ruff check python/ tests/
cargo fmt --check
cargo clippy --all-targets
```

Run focused tests while iterating:

```bash
.venv/bin/python -m pytest tests/test_chrom_pass.py
.venv/bin/python -m pytest tests/test_dataset_staging.py
.venv/bin/python -m pytest tests/test_losses.py
.venv/bin/python -m pytest tests/test_train_metrics.py
```

When changing the Rust writer, rebuild the extension and run both the writer
parity and dataset staging tests. When changing workflow configuration, run a
dry run against the example:

```bash
regulonado pipeline examples/flashzoi_four_replicates.yaml --dry-run
```

## Style and commits

Python uses a 100-character line length with Ruff rules `E`, `F`, and `I`.
Prefer comments that explain a constraint or design choice; omit comments that
only restate the next line.

Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, or
`chore:`). Release Please manages version numbers and the changelog, so do not
edit them by hand.

Before opening a pull request, include tests for changed behavior and confirm
that copied commands and file paths in the documentation still work.
