# ReguloNado

ReguloNado builds Hugging Face Arrow datasets from genomics inputs and trains
sequence-to-function models such as Borzoi and Enformer on those datasets. The
main data path is a Rust-backed chromosome-pass writer; the usual training
entrypoint is the friendly `regulonado train` CLI, backed by the Hugging Face
`Trainer`.

## Repository Layout

- `src/`: Rust/PyO3 data readers and Arrow writers.
- `python/regulonado/dataset.py`: dataset construction, transforms, scaling, and augmentation.
- `python/regulonado/train.py`: supported training entrypoint.
- `python/regulonado/model/`: backbone adapters and prediction heads.
- `python/configs/`: Hydra configs for backbones, heads, losses, and canonical experiments.
- `scripts/`: Slurm launchers; see `scripts/README.md`.
- `python/configs/experiment/`: canonical baseline experiment configs.
- `scripts/experiment/`: production run configs (the place to add new experiments).
- `tests/`: model, dataset, and smoke coverage.

## Environment

Use `uv` as the source of truth for the Python environment. The normal setup is:

```bash
uv sync
source .venv/bin/activate
```

For development and tests:

```bash
uv sync --extra dev
source .venv/bin/activate
```

For GPU installs with FlashAttention support:

```bash
sbatch scripts/install_gpu_env_slurm.sh
```

That wrapper loads the CUDA module, sets `CUDA_HOME`, pins the host compiler for
`nvcc`, and then runs the GPU-extra sync. For GPU development with tests:

```bash
INSTALL_EXTRAS="--extra dev --extra gpu" sbatch scripts/install_gpu_env_slurm.sh
```

After Rust changes, rebuild the extension:

```bash
.venv/bin/maturin develop --release
```

## Dataset Workflow

### Building a dataset

Set the required environment variables and submit the build job:

```bash
BED_FILE=/path/to/intervals.bed \
FASTA_FILE=/path/to/genome.fa \
OUTPUT_DIR=/path/to/output \
BIGWIG_LIST=/path/to/bigwig_paths.txt \
sbatch scripts/build_dataset_slurm.sh
```

`BIGWIG_LIST` is a newline-delimited file of BigWig paths, one per line. Optional
environment variables (with defaults in the script header) control context length,
bin size, staging, compression, rechunking, and worker counts.

The script builds with the chromosome-pass Rust writer and optionally rechunks the
output to small ZSTD batches for streaming:

```bash
RECHUNK=true MAX_BATCH_SIZE=4 ZSTD_LEVEL=3 \
BED_FILE=... FASTA_FILE=... OUTPUT_DIR=... BIGWIG_LIST=... \
sbatch scripts/build_dataset_slurm.sh
```

To rechunk an existing dataset separately:

```bash
SRC=/path/to/dataset DST=/path/to/rechunked \
sbatch scripts/rechunk_dataset_slurm.sh
```

Or using the CLI directly:

```bash
regulonado recompress-dataset /path/to/src /path/to/dst --max-batch-size 4
```

### Computing scale factors

Scale factors convert per-track BigWig signal (typically RPKM) to raw read counts
before loss computation. Compute them after building the dataset:

```bash
regulonado calculate-original-scaling dataset/regulonado_metadata.json
regulonado calculate-tmm-scaling dataset/regulonado_metadata.json
regulonado enrich-metadata dataset/regulonado_metadata.json dataset/scale_factors.parquet
```

## Training

### Quick start

```bash
EXPERIMENT=head_only_borzoi \
DATA_DIR=/path/to/dataset \
sbatch scripts/train_slurm.sh
```

`EXPERIMENT` selects a Hydra config by name. The launcher searches
`python/configs/experiment/` (built-in) and `scripts/experiment/` (production).

### Recommended progressive training workflow

Training is most effective when done progressively. Each phase warm-starts from
the previous one (model weights only, fresh optimizer):

```bash
# Phase 1 — head-only, backbone frozen, fast convergence (~5 k steps)
EXPERIMENT=head_only_borzoi DATA_DIR=... sbatch scripts/train_slurm.sh

# Phase 2 — unfreeze 2 output-end backbone stages
EXPERIMENT=stage2_unfreeze2_borzoi \
INIT_WEIGHTS_FROM_CHECKPOINT=outputs/train/head_only_borzoi-JOBID/checkpoint-NNNN \
DATA_DIR=... sbatch scripts/train_slurm.sh

# Phase 3 — unfreeze 4 stages + reverse-complement augmentation
EXPERIMENT=stage3_deep_finetune_borzoi \
INIT_WEIGHTS_FROM_CHECKPOINT=outputs/train/stage2_unfreeze2_borzoi-JOBID/checkpoint-NNNN \
DATA_DIR=... sbatch scripts/train_slurm.sh

# Phase 4 (optional) — peak sharpening with top-K loss
EXPERIMENT=stage4_peak_finetune_borzoi \
INIT_WEIGHTS_FROM_CHECKPOINT=outputs/train/stage3_deep_finetune_borzoi-JOBID/checkpoint-NNNN \
DATA_DIR=... sbatch scripts/train_slurm.sh
```

To submit all four phases as a dependency chain automatically:

```bash
DATA_DIR=/path/to/dataset bash scripts/train_pipeline_slurm.sh
```

Options for the pipeline script:

```bash
# Stop after phase 3 (skip peak-sharpening phase 4)
DATA_DIR=... STOP_AFTER_PHASE=3 bash scripts/train_pipeline_slurm.sh

# Resume from phase 3 (phases 1 and 2 already done)
DATA_DIR=... START_FROM_PHASE=3 bash scripts/train_pipeline_slurm.sh

# Use topk_reweight instead of topk_additive for phase 4
DATA_DIR=... PEAK_LOSS=topk_reweight bash scripts/train_pipeline_slurm.sh
```

### Experiment configs

All hyperparameters live in YAML files. To start a new experiment, copy the
nearest config and adjust what matters:

| Config | Description |
|--------|-------------|
| `head_only_borzoi.yaml` | Phase 1: frozen backbone, head-only, lr=1e-3 |
| `stage2_unfreeze2_borzoi.yaml` | Phase 2: 2 output-end stages unfrozen, lr=2e-4/2e-6 |
| `stage3_deep_finetune_borzoi.yaml` | Phase 3: 4 stages + RC augmentation, lr=5e-5/5e-7 |
| `stage4_peak_finetune_borzoi.yaml` | Phase 4: topk_additive loss for peak sharpening |

The `magnitude_fix_*.yaml` configs in `python/configs/experiment/` are ablation
templates comparing different loss functions and squash settings at a fixed training
budget — useful as starting points for loss function sweeps.

### Available loss functions

Loss configs live in `python/configs/loss/`:

| Loss | Description |
|------|-------------|
| `poisson_multinomial` | Default: Poisson total-count + multinomial profile loss (Borzoi-style) |
| `scaled_poisson_multinomial` | As above but with a per-track softmax scale step |
| `poisson_nll` | Per-bin Poisson NLL; stronger magnitude gradient at peaks |
| `log1p_huber` | Per-bin Huber loss in log1p space; robust to outliers |
| `topk_additive` | poisson_multinomial + additive second pass on top-K bins |
| `topk_reweight` | poisson_multinomial with per-bin rank weighting on the multinomial term |
| `transfer_calibration` | Composite: low multinomial weight + per-bin log1p MSE + top-K Huber |

Select a loss in an experiment YAML with `defaults: - override /loss: <name>`.

### Checkpoint reuse

**Full resume** — restores model weights, optimizer, scheduler, and RNG state.
Use when continuing an interrupted run:

```bash
EXPERIMENT=head_only_borzoi DATA_DIR=... \
sbatch scripts/train_slurm.sh \
  trainer.resume_from_checkpoint=outputs/train/head_only_borzoi-JOBID/checkpoint-NNNN
```

**Warm start** — loads model weights only, creates a fresh optimizer/scheduler.
Use when changing learning rate, scheduler, unfreezing policy, or training objective.
Stage 2–4 configs read the checkpoint from `INIT_WEIGHTS_FROM_CHECKPOINT` via
`${oc.env:...}` interpolation, or set `init_weights_from_checkpoint` directly in
the YAML.

### Local runs

```bash
regulonado train /path/to/dataset \
  --output-dir outputs/train/quick-check \
  --max-steps 1000 \
  --batch-size 8

regulonado train /path/to/dataset --nproc-per-node 2  # multi-GPU
regulonado train /path/to/dataset --max-steps 10 --no-wandb  # smoke test
```

Raw Hydra overrides can be appended to both the local CLI and the Slurm script:

```bash
regulonado train /path/to/dataset trainer.max_steps=2000
python -m regulonado.train +experiment=head_only_borzoi data.path=...  # advanced
```

### Loss comparison sweeps

To run all magnitude-fix ablations in parallel from a shared checkpoint:

```bash
INIT_WEIGHTS_FROM_CHECKPOINT=/path/to/checkpoint \
DATA_DIR=/path/to/dataset \
bash scripts/run_magnitude_experiments_slurm.sh
```

This submits six experiments (two per-bin loss variants × with/without squash, plus
high-Poisson-weight baseline and transfer_calibration), each on a single GPU.

## Reproducibility Outputs

Each run writes:

- `resolved_config.json`: fully resolved Hydra config.
- `provenance.json`: command, git commit/status, dataset hash and split summary,
  package versions, CUDA/Torch details, Slurm context, and checkpoint reuse mode.
- `git_diff.patch`: local diff when provenance diff capture is enabled.
- `trainer_state.json`, checkpoints, and `training_summary.json`.

These files live in `output_dir`, which defaults to the Hydra output directory or
the `RUN_DIR` used by the Slurm wrapper.

## Fit Metrics

W&B receives compact aggregate metrics at each eval step:

- `eval_loss`: mean training loss over the validation set.
- `eval_pearson_bin_median`: median per-track Pearson correlation over all bins.
- `eval_pearson_topN_median`: median Pearson correlation restricted to the top-N
  bins by target signal (where N is controlled by `trainer.topk_bins`, default 256).

The `metric_for_best_model` in each experiment config controls which metric is used
for checkpoint selection and early stopping. Set `greater_is_better: true` when
using `pearson_bin_median` (default in the provided configs).

## Predicted-vs-Real Examples

At each eval, ReguloNado saves predicted-vs-real signal plots for a small fixed set
of validation examples:

```text
examples/step_000500/example_00_track_0000.png
examples/step_000500/example_01_track_0001.png
```

The number of examples is set by `trainer.num_plot_examples` (default 4).
Set to 0 to disable plotting.

## Tests

`pytest` is included in the `dev` extra. Install the development environment first:

```bash
uv sync --extra dev
```

Run the full suite:

```bash
.venv/bin/python -m pytest tests/
```

Focused checks:

```bash
.venv/bin/python -m pytest tests/test_chrom_pass.py      # Rust writer parity
.venv/bin/python -m pytest tests/test_dataset_staging.py  # staging and deduplication
.venv/bin/python -m pytest tests/test_train_metrics.py    # training metrics
```
