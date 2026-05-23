# Regulonado

Regulonado builds Hugging Face Arrow datasets from genomics inputs and trains
sequence-to-function models such as Borzoi and Enformer on those datasets. The
main data path is a Rust-backed chromosome-pass writer; the usual training
entrypoint is the friendly `regulonado train` CLI, backed by the Hugging Face
`Trainer`.

## Repository Layout

- `src/`: Rust/PyO3 data readers and Arrow writers.
- `python/regulonado/dataset.py`: dataset construction, transforms, scaling, and augmentation.
- `python/regulonado/train.py`: supported training entrypoint.
- `python/regulonado/model/`: backbone adapters and prediction heads.
- `python/configs/`: Hydra configs for backbones, heads, losses, and experiments.
- `scripts/`: thin local/Slurm wrappers for common workflows.
- `tests/`: model, dataset, and smoke coverage.

`python/regulonado/training/accelerate_trainer.py` has been removed; use the
Hugging Face Trainer path for new work.

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
sbatch scripts/install_gpu_env.sh
```

That wrapper loads the CUDA module, sets `CUDA_HOME`, pins the host compiler for
`nvcc`, and then runs the GPU-extra sync. For GPU development with tests:

```bash
INSTALL_EXTRAS="--extra dev --extra gpu" sbatch scripts/install_gpu_env.sh
```

After Rust changes, rebuild the extension:

```bash
.venv/bin/maturin develop --release
```

## Dataset Workflow

Typical flow:

```bash
regulonado calculate-original-scaling dataset/regulonado_metadata.json
regulonado calculate-tmm-scaling dataset/regulonado_metadata.json
regulonado enrich-metadata dataset/regulonado_metadata.json dataset/scale_factors.parquet
```

Builds are usually launched through:

```bash
sbatch scripts/create_dataset.sh
```

Compressed datasets can be rechunked for training with:

```bash
sbatch scripts/rechunk_dataset.sh
```

## Training

The usual local entrypoint is:

```bash
regulonado train /path/to/dataset
```

The default condition-agnostic experiment is designed for many heterogeneous
tracks, such as cell-type and assay collections without perturbation conditions.
It uses a Borzoi backbone, `transfer_mlp` head, bf16, W&B logging, streamed Arrow
data, frozen backbone with the final two backbone stages unfrozen, and separate
head/backbone learning rates.

For Slurm:

```bash
DATA_DIR=/path/to/dataset sbatch scripts/train_condition_agnostic.sh
```

The Slurm wrapper intentionally accepts only run-specific environment variables:
`EXPERIMENT`, `DATA_DIR`, `RUN_DIR`, `WANDB_PROJECT`, `WANDB_RUN_NAME`,
`NPROC_PER_NODE`, `RESUME_FROM_CHECKPOINT`, and `INIT_WEIGHTS_FROM_CHECKPOINT`.
Training policy should live in a Hydra experiment file, not in shell variables.

Useful local overrides:

```bash
regulonado train /path/to/dataset \
  --output-dir outputs/train/quick-check \
  --max-steps 1000 \
  --batch-size 8 \
  --learning-rate 2e-4
```

Use more than one local GPU/process with:

```bash
regulonado train /path/to/dataset --nproc-per-node 2
```

Disable W&B for smoke tests:

```bash
regulonado train /path/to/dataset --max-steps 10 --no-wandb
```

Any raw Hydra override can still be appended for one-off experiments:

```bash
regulonado train /path/to/dataset trainer.fit_examples.num_examples=8
```

For Slurm, pass one-off Hydra overrides as script arguments so they are visible in
the job command and provenance:

```bash
DATA_DIR=/path/to/dataset \
sbatch scripts/train_condition_agnostic.sh trainer.max_steps=2000 trainer.batch_size=8
```

For a real experiment, prefer copying
`python/configs/experiment/condition_agnostic_borzoi.yaml`, naming the new file
after the run, and launching it explicitly:

```bash
EXPERIMENT=condition_agnostic_borzoi_lr2e4 \
DATA_DIR=/path/to/dataset \
sbatch scripts/train_condition_agnostic.sh
```

## Checkpoint Reuse

Full resume restores model weights, optimizer, scheduler, RNG state, and Trainer
state. Use this when continuing an interrupted run:

```bash
regulonado train /path/to/dataset \
  --output-dir outputs/train/condition-agnostic-5719240 \
  --resume-from-checkpoint outputs/train/condition-agnostic-5719240/checkpoint-3000
```

Warm start loads model weights only and creates a fresh optimizer/scheduler. Use
this when changing learning rate, scheduler, batch size, unfreezing policy, or
training objective:

```bash
regulonado train /path/to/dataset \
  --output-dir outputs/train/condition-agnostic-from-3k-freshopt \
  --init-weights-from-checkpoint outputs/train/condition-agnostic-5719240/checkpoint-3000
```

The Slurm wrapper exposes the same modes through `RESUME_FROM_CHECKPOINT` and
`INIT_WEIGHTS_FROM_CHECKPOINT`. Set only one resume mode at a time.

The lower-level Hydra entrypoint remains available for advanced use:

```bash
python -m regulonado.train +experiment=condition_agnostic_borzoi data.path=/path/to/dataset
```

## Reproducibility Outputs

Each run writes:

- `resolved_config.json`: fully resolved Hydra config.
- `provenance.json`: command, git commit/status, dataset hash and split summary,
  package versions, CUDA/Torch details, Slurm context, and checkpoint reuse mode.
- `git_diff.patch`: local diff when provenance diff capture is enabled.
- `trainer_state.json`, checkpoints, and `training_summary.json`.

These files live in `output_dir`, which defaults to the Hydra output directory or
the `RUN_DIR` used by the Slurm wrapper.

## Fit Assessment

W&B receives compact aggregate metrics only, for example:

- `fit/pearson_mean`
- `fit/pearson_median`
- `fit/pearson_q10`
- `fit/spearman_median`
- `fit/high_signal_pearson_median`
- `fit/total_signal_ratio_median`
- `eval_loss`

Full per-track diagnostics are saved locally under:

```text
fit/step_000500/per_track_metrics.parquet
fit/step_000500/per_track_metrics.csv
fit/step_000500/group_metrics.parquet
fit/step_000500/summary.json
```

Individual track metrics are not logged as 1500 W&B scalar series.

## Predicted-vs-Real Examples

At each eval, Regulonado samples a small reproducible set of validation examples
and tracks, then saves predicted-vs-real signal plots:

```text
fit/examples/step_000500/example_00_track_0381.png
fit/examples/step_000500/manifest.json
```

The default is 4 examples and 3 tracks per example. A small image panel can be
sent to W&B; all plots are always available locally when example diagnostics are
enabled.

## Tests

`pytest` is included in the `dev` extra. Install the development environment first:

```bash
uv sync --extra dev
```

Then run the focused Python tests with:

```bash
.venv/bin/python -m pytest tests/test_model_transfer.py
```

Run the full suite with:

```bash
.venv/bin/python -m pytest tests/
```
