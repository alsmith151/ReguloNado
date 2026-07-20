# Contributing to Regulonado

## Development setup

### Prerequisites

- Python 3.12 or 3.13
- Rust 1.85+ (via rustup or your package manager)
- maturin >=1.7,<2.0 (installed as a dev dependency)

### Initial setup

```bash
git clone https://github.com/alsmith151/ReguloNado.git
cd ReguloNado

# Use the repo's venv
python -m venv .venv
source .venv/bin/activate

# Install the package with all dev dependencies
uv sync
```

### Building the Rust extension

**IMPORTANT: The `VIRTUAL_ENV` environment variable must be set explicitly. Maturin will pick the wrong interpreter if you omit this.**

```bash
export VIRTUAL_ENV=$PWD/.venv
.venv/bin/maturin develop --release
```

**Why this is needed:** Without `VIRTUAL_ENV`, maturin may pick up a macOS or system interpreter instead of the `.venv` Python. Setting `VIRTUAL_ENV` is the only reliable way to point maturin at the right interpreter. There is no `-i` flag in this maturin version.

After **any change to `src/*.rs`**, rebuild:

```bash
export VIRTUAL_ENV=$PWD/.venv
.venv/bin/maturin develop --release
```

Do not skip this step. The SLURM build wrapper (`scripts/build_dataset_slurm.sh`) will fail fast if the extension (`python/regulonado/_rs*.so`) has not been rebuilt.

For editable reinstalls, `uv sync` also rebuilds the extension automatically.

## Running tests

### Quick test run

```bash
.venv/bin/pytest tests/
```

The full test suite has 97 tests across 6 files and takes a few minutes.

### Narrower checks (recommended for quick feedback)

```bash
# Chrom-pass writer parity and HuggingFace loadability
.venv/bin/pytest tests/test_chrom_pass.py

# Dataset staging and BigWig deduplication
.venv/bin/pytest tests/test_dataset_staging.py

# Loss function numerics
.venv/bin/pytest tests/test_losses.py
```

### CUDA library caveat

On a machine without CUDA libraries, the repo venv's PyTorch may fail to import with:
```
libcudnn.so.9: cannot open shared object file
```

This blocks 4 of the 6 test files. The workaround is a CPU-only PyTorch venv:

```bash
# Create a separate CPU-only venv
uv venv --python 3.12 cputorch

# Install CPU PyTorch
export VIRTUAL_ENV=$PWD/cputorch
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Regulonado with minimal ML deps (no CUDA dependencies)
uv pip install -e . --no-deps
uv pip install numpy scipy transformers torchmetrics datasets pytest \
  pyfaidx pyarrow pybigtools loguru pydantic typer pandas polars matplotlib \
  hydra-core peft accelerate borzoi-pytorch enformer-pytorch
```

Then run tests with:
```bash
VIRTUAL_ENV=$PWD/cputorch $PWD/cputorch/bin/pytest tests/test_chrom_pass.py
```

Note: `borzoi-pytorch` and `hydra-core` are needed just to *collect* the tests, because `regulonado.model` and `train.py` import them at module scope.

## Code style

### Python

Line length: **100 characters**

Linting rules: `E` (PEP 8), `F` (PyFlakes), `I` (isort imports)

```bash
.venv/bin/ruff check python/ tests/
```

Before committing, ensure no ruff violations:

```bash
.venv/bin/ruff check python/
```

### Rust

Use clippy for linting:

```bash
cargo clippy --all-targets
```

## Commit conventions

This repo uses **Conventional Commits** for automatic changelog and version management via release-please.

Format: `<type>(<scope>): <subject>`

**Types:**
- `feat` — new feature
- `fix` — bug fix
- `refactor` — code restructuring (no feature change)
- `docs` — documentation only
- `test` — test additions/fixes
- `chore` — build, CI, dependencies (not production code)
- `ci` — CI/CD configuration changes
- `style` — formatting, no logic change

**Examples:**

```
feat(chrom-pass): parallelize shard writes with Rayon

fix(dataset): handle missing FASTA index more gracefully

docs(architecture): explain Rust vs Python split

refactor(schema): consolidate Arrow record-batch builders

test(losses): add numerical stability checks for gradient-based loss

ci(release): bump release-please config version
```

**Important:** release-please automatically bumps both `pyproject.toml` and `Cargo.toml` (via extra-files config). Do not hand-edit version numbers in either file — commit conventions drive the version bump.

## Workflow and Snakemake

Pipeline rules live in `workflow/`:

```
workflow/
  Snakefile              — main orchestration
  rules/                 — rule files sourced by Snakefile
  config.yaml            — pipeline configuration (intervals, sample metadata)
  profiles/slurm/        — cluster job submission and resource configs
```

Configuration overrides go in `config/config.yaml` (or as command-line arguments).

Test workflow changes with:

```bash
snakemake --configfile config/config.yaml -n
```

(The `-n` flag shows what would run without executing.)

Cluster-specific settings (SLURM job params, resource allocation) live in `workflow/profiles/slurm/config.yaml`. Modify those if you're running on a different cluster or with different resource constraints.

## Pull request checklist

Before pushing a PR:

- [ ] Tests pass: `pytest tests/`
- [ ] Ruff clean: `ruff check python/ tests/`
- [ ] Clippy clean: `cargo clippy --all-targets` (if Rust changed)
- [ ] New public API has docstrings
- [ ] CHANGELOG.md not hand-edited (release-please manages it)
- [ ] Conventional Commits on all commits in the PR

## Building documentation

There are no generated docs in this repo. See `ARCHITECTURE.md` for high-level design, and `CLAUDE.md` for developer notes and common workflows.

## Troubleshooting

### Maturin picks the wrong Python interpreter

**Symptom:** `error: no default toolchain configured` or references to macOS when you're on Linux.

**Fix:** Ensure `VIRTUAL_ENV` is set before running maturin:
```bash
export VIRTUAL_ENV=$PWD/.venv
.venv/bin/maturin develop --release
```

### CUDA import failures in tests

**Symptom:** `libcuddn.so.9: cannot open shared object file`

**Fix:** Use the CPU-only PyTorch venv workaround (see CUDA library caveat above).

### Arrow shard files not created

**Symptom:** Dataset build completes but output directory is empty.

**Fix:** Check that the Rust extension was rebuilt after your last code change. Even small edits to `src/*.rs` require a rebuild. Run `maturin develop --release` again.

### Pre-commit hook failures

If a git hook (e.g., ruff formatting) blocks your commit:

1. Fix the issue (e.g., run `ruff check python/ --fix`)
2. Stage the changes: `git add python/`
3. Create a **new** commit (do not amend): `git commit -m "..."`

Amending after a hook failure can accidentally destroy unrelated changes.

## Questions?

- See `ARCHITECTURE.md` for high-level design and where to add features
- See `CLAUDE.md` for workflow commands and common patterns
- Check existing tests for usage examples
- Open an issue on GitHub with a clear example
