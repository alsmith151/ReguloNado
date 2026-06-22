# ReguloNado

ReguloNado turns genomic tracks into trained **sequence-to-function** models. You
point it at a reference genome and a set of BigWig coverage tracks, and it
produces a Hugging Face Arrow dataset and a fine-tuned model (Borzoi or Enformer
backbone) that predicts those tracks' signal directly from DNA sequence.

It is built for genomics-scale data: the data path is a Rust/PyO3 writer that
streams BigWig + FASTA straight into compressed Arrow shards without ever
materialising a dense signal intermediate, and the training path is a thin,
friendly CLI on top of the Hugging Face `Trainer`.

```
   ┌──────────┐   ┌──────────┐   ┌───────────┐
   │ BED      │   │ FASTA    │   │ BigWig × T │
   │ intervals│   │ genome   │   │ tracks     │
   └────┬─────┘   └────┬─────┘   └─────┬─────┘
        └──────────────┼───────────────┘
                       ▼
            regulonado build           ← Rust chrom-pass writer
                       ▼
        Arrow DatasetDict (train/val/test)
          input_ids: one-hot (4, L)
          labels:    binned signal (T, B)
                       ▼
            regulonado scale / *-scaling    ← per-track RPKM → raw-count factors
                       ▼
            regulonado train               ← Borzoi/Enformer + prediction head
                       ▼
        checkpoints · metrics · provenance · plots
```

A built dataset stores each example as one-hot DNA (`input_ids`, int8 `(4, L)`)
and per-track binned coverage (`labels`, float32 `(T, B)`), split into
`train` / `validation` / `test` by the fold label in column 4 of the BED file.

## Contents

- [Installation](#installation)
- [End-to-end quickstart](#end-to-end-quickstart)
- [Building a dataset](#building-a-dataset)
- [Scale factors](#scale-factors)
- [Training](#training)
- [Experiment & loss configs](#experiment--loss-configs)
- [Checkpoint reuse](#checkpoint-reuse)
- [Run outputs](#run-outputs)
- [External dependencies](#external-dependencies)
- [Development & tests](#development--tests)
- [Repository layout](#repository-layout)

## Installation

ReguloNado ships as layered extras so you only install what a given task needs.
The Rust extension is compiled at install time, so a working Rust toolchain is
required.

| Goal | Install |
|------|---------|
| CLI only (no ML deps) | `pip install regulonado` |
| Build datasets | `pip install "regulonado[data]"` |
| Full training stack | `pip install "regulonado[train]"` |
| GPU training + FlashAttention | `pip install "regulonado[gpu]"` |
| Plotting / visualisation | `pip install "regulonado[viz]"` |
| Development (ruff, pytest, maturin) | `pip install "regulonado[dev]"` |

The `gpu` extra requires `nvcc` and a matching CUDA toolkit; `ninja` is pulled in
automatically to compile the `flash_attn` extension.

### With uv (recommended for local development)

`uv` is the source of truth for the local environment:

```bash
uv sync                  # core deps
uv sync --extra data     # dataset building
uv sync --extra train    # torch + full training stack
uv sync --extra dev      # ruff, pytest, maturin
source .venv/bin/activate
```

For GPU installs with FlashAttention, use the Slurm wrapper — it loads the CUDA
module, sets `CUDA_HOME`, pins the host compiler for `nvcc`, then runs the
GPU-extra sync:

```bash
sbatch scripts/install_gpu_env_slurm.sh
# with tests too:
INSTALL_EXTRAS="--extra dev --extra gpu" sbatch scripts/install_gpu_env_slurm.sh
```

After any change to the Rust sources in `src/`, rebuild the extension:

```bash
.venv/bin/maturin develop --release
```

## End-to-end quickstart

The full pipeline is four steps: **build → scale → enrich → train**.

```bash
# 1. Build the Arrow dataset from BED + FASTA + a directory of BigWigs.
regulonado build intervals.bed genome.fa dataset/ --bigwig-dir bw/ --stage

# 2. Infer RPKM → raw-count scale factors for each track.
regulonado calculate-original-scaling dataset/regulonado_metadata.json

# 3. Write those factors into the dataset metadata train.py reads.
regulonado enrich-metadata dataset/regulonado_metadata.json dataset/scale_factors.parquet

# 4. Train (smoke test shown; drop the limits for a real run).
regulonado train dataset/ --experiment head_only_borzoi --max-steps 10 --no-wandb
```

Steps 2–3 are only needed when your BigWigs are in RPKM / normalised units and
you want the model to train on raw read counts. If they are already raw counts,
set `apply_scale: false` in the experiment config and skip them.

On a cluster, each step has a Slurm wrapper under `scripts/` (see below); the
local CLI commands above accept the same options.

## Building a dataset

### Locally

```bash
regulonado build intervals.bed genome.fa out/ \
    --bigwig-dir bw/ \
    --split train:fold0,fold1,fold2 --split validation:fold4 --split test:fold3 \
    --shift-max-bp 128 --num-proc 16 --stage
```

- Provide tracks as either `--bigwig-dir DIR` (sorted by name) or repeated
  `--bigwig file.bw` (order preserved). Track order is the column order of
  `labels`.
- `--split NAME:FOLD1,FOLD2` maps BED column-4 fold labels to splits. Omit it to
  use the default `train` / `validation` / `test` split.
- `--stage` copies FASTA + BigWigs to local scratch first (recommended on Ceph).
- `--strategy chrom_pass` (default) writes one shard per chromosome with ~10×
  fewer BigWig seeks; `--strategy fast` is the sample-batched fallback.

Run `regulonado build --help` for the full set of context-length, bin-size,
threading, compression, and dedupe options.

### On Slurm

```bash
BED_FILE=/path/to/intervals.bed \
FASTA_FILE=/path/to/genome.fa \
OUTPUT_DIR=/path/to/output \
BIGWIG_LIST=/path/to/bigwig_paths.txt \
sbatch scripts/build_dataset_slurm.sh
```

`BIGWIG_LIST` is a newline-delimited file of BigWig paths. Optional env vars
(defaults in the script header) control context length, bin size, staging,
compression, rechunking, and worker counts. To rechunk to small ZSTD batches for
streaming during the build:

```bash
RECHUNK=true MAX_BATCH_SIZE=4 ZSTD_LEVEL=3 \
BED_FILE=... FASTA_FILE=... OUTPUT_DIR=... BIGWIG_LIST=... \
sbatch scripts/build_dataset_slurm.sh
```

To rechunk an existing dataset separately:

```bash
regulonado recompress-dataset /path/to/src /path/to/dst --max-batch-size 4
# or: SRC=... DST=... sbatch scripts/rechunk_dataset_slurm.sh
```

## Scale factors

Scale factors convert per-track BigWig signal (typically RPKM) to raw read counts
before loss computation. Compute and apply them after building:

```bash
regulonado calculate-original-scaling dataset/regulonado_metadata.json
regulonado calculate-tmm-scaling dataset/regulonado_metadata.json        # optional TMM correction
regulonado enrich-metadata dataset/regulonado_metadata.json dataset/scale_factors.parquet
```

- `calculate-original-scaling` reads BigWig header metadata (via BamNado) to infer
  library sizes and the RPKM→raw-count factor per track.
- `calculate-tmm-scaling` layers an edgeR-style TMM normalisation on top, estimated
  from the Arrow shards.
- `enrich-metadata` writes the resulting `scale_factor` / `clip_soft` / `clip_hard`
  into `final_track_records`, which `train.py` reads at training time.

## Training

### Find an experiment

Experiments are named Hydra configs. List them and inspect any one without
opening the YAML:

```bash
regulonado experiments                          # list all with one-line summaries
regulonado experiments show sharp_transfer_learning   # full description + effective settings
```

Not sure which to run? Let the CLI walk you through it:

```bash
regulonado train --interactive    # pick experiment, dataset, GPUs, local vs Slurm
```

### Quick start (Slurm)

Submit straight from the CLI — no need to remember the env-var incantation:

```bash
regulonado train /path/to/dataset -e head_only_borzoi --slurm
```

This submits `scripts/train_slurm.sh` via `sbatch`, setting `EXPERIMENT`,
`DATA_DIR`, and `REPO_DIR` for you. Add `--dry-run` to preview the command,
`--slurm-gpus`/`--slurm-time`/`--slurm-partition` to override the `#SBATCH`
defaults, and any `--max-steps`/`--lr`/raw Hydra overrides as usual. The
equivalent manual form still works:

```bash
EXPERIMENT=head_only_borzoi \
DATA_DIR=/path/to/dataset \
sbatch scripts/train_slurm.sh
```

`EXPERIMENT` names a Hydra config; the launcher searches `python/configs/experiment/`
(built-in) and `scripts/experiment/` (production).

### Local runs

```bash
regulonado train /path/to/dataset \
  --output-dir outputs/train/quick-check \
  --max-steps 1000 --batch-size 8

regulonado train /path/to/dataset --nproc-per-node 2     # multi-GPU via torchrun
regulonado train /path/to/dataset --max-steps 10 --no-wandb   # smoke test
```

An unknown `-e` name prints the available experiments and a did-you-mean
suggestion instead of a Hydra stack trace.

Raw Hydra overrides can be appended to either the CLI or the Slurm script:

```bash
regulonado train /path/to/dataset trainer.max_steps=2000
```

### Recommended progressive workflow

Training works best in phases, each warm-starting from the previous one (model
weights only, fresh optimizer):

```bash
# Phase 1 — head only, backbone frozen, fast convergence (~5k steps)
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

Submit all four phases as a dependency chain automatically:

```bash
DATA_DIR=/path/to/dataset bash scripts/train_pipeline_slurm.sh

# variants:
DATA_DIR=... STOP_AFTER_PHASE=3 bash scripts/train_pipeline_slurm.sh   # skip phase 4
DATA_DIR=... START_FROM_PHASE=3 bash scripts/train_pipeline_slurm.sh   # resume from phase 3
DATA_DIR=... PEAK_LOSS=topk_reweight bash scripts/train_pipeline_slurm.sh
```

## Experiment & loss configs

All hyperparameters live in Hydra YAML. To start a new experiment, copy the
nearest config in `python/configs/experiment/` and adjust what matters.

| Experiment config | Phase / purpose |
|--------|-------------|
| `head_only_borzoi.yaml` | Phase 1: frozen backbone, head only, lr=1e-3 |
| `stage2_unfreeze2_borzoi.yaml` | Phase 2: 2 output-end stages unfrozen, lr=2e-4/2e-6 |
| `stage3_deep_finetune_borzoi.yaml` | Phase 3: 4 stages + RC augmentation, lr=5e-5/5e-7 |
| `stage4_peak_finetune_borzoi.yaml` | Phase 4: topk_additive loss for peak sharpening |
| `magnitude_fix_*.yaml` | Ablation templates for loss-function / squash sweeps |

Loss configs live in `python/configs/loss/`. Select one in an experiment YAML with
`defaults: - override /loss: <name>`.

| Loss | Description |
|------|-------------|
| `poisson_multinomial` | Default: Poisson total-count + multinomial profile (Borzoi-style) |
| `scaled_poisson_multinomial` | As above, with a per-track softmax scale step |
| `poisson_nll` | Per-bin Poisson NLL; stronger magnitude gradient at peaks |
| `log1p_huber` | Per-bin Huber loss in log1p space; robust to outliers |
| `topk_additive` | poisson_multinomial + additive second pass on top-K bins |
| `topk_reweight` | poisson_multinomial with per-bin rank weighting on the multinomial term |
| `transfer_calibration` | Composite: low multinomial weight + per-bin log1p MSE + top-K Huber |

To run all magnitude-fix ablations in parallel from a shared checkpoint:

```bash
INIT_WEIGHTS_FROM_CHECKPOINT=/path/to/checkpoint \
DATA_DIR=/path/to/dataset \
bash scripts/run_magnitude_experiments_slurm.sh
```

## Checkpoint reuse

**Full resume** restores model weights, optimizer, scheduler, and RNG state — use
to continue an interrupted run:

```bash
EXPERIMENT=head_only_borzoi DATA_DIR=... \
sbatch scripts/train_slurm.sh \
  trainer.resume_from_checkpoint=outputs/train/head_only_borzoi-JOBID/checkpoint-NNNN
```

**Warm start** loads model weights only with a fresh optimizer/scheduler — use when
changing learning rate, scheduler, unfreezing policy, or training objective. Stage
2–4 configs read the checkpoint from `INIT_WEIGHTS_FROM_CHECKPOINT` via
`${oc.env:...}` interpolation, or set `init_weights_from_checkpoint` directly in the
YAML.

## Run outputs

Each run writes, into `output_dir` (defaults to the Hydra output dir or the Slurm
wrapper's `RUN_DIR`):

- `resolved_config.json` — fully resolved Hydra config.
- `provenance.json` — command, git commit/status, dataset hash and split summary,
  package versions, CUDA/Torch details, Slurm context, checkpoint reuse mode.
- `git_diff.patch` — local diff when provenance diff capture is enabled.
- `trainer_state.json`, checkpoints, and `training_summary.json`.

**Metrics** logged to W&B at each eval step:

- `eval_loss` — mean validation loss.
- `eval_pearson_bin_median` — median per-track Pearson over all bins.
- `eval_pearson_topN_median` — median Pearson restricted to the top-N bins by target
  signal (`trainer.topk_bins`, default 256).

`metric_for_best_model` in each experiment config controls checkpoint selection and
early stopping (set `greater_is_better: true` for `pearson_bin_median`).

**Predicted-vs-real plots** are saved for a small fixed set of validation examples at
each eval:

```
examples/step_000500/example_00_track_0000.png
```

Count is set by `trainer.num_plot_examples` (default 4); set to 0 to disable.

## External dependencies

### BamNado

`regulonado calculate-original-scaling` requires the
[BamNado](https://github.com/alsmith151/BamNado) binary (`bamnado`) on `PATH`. It
reads BigWig header metadata to infer library size and compute RPKM→raw-count
factors.

```bash
# Linux x86-64 — check the releases page for the latest tag
wget https://github.com/alsmith151/BamNado/releases/latest/download/bamnado-x86_64-unknown-linux-musl.tar.gz
tar -xzf bamnado-x86_64-unknown-linux-musl.tar.gz
mv bamnado ~/.local/bin/      # or any directory on PATH
```

If not on `PATH`, point to it with the `BAMNADO` environment variable:

```bash
BAMNADO=/path/to/bamnado regulonado calculate-original-scaling metadata.json
```

## Development & tests

`pytest` is included in the `dev` extra:

```bash
uv sync --extra dev
.venv/bin/python -m pytest tests/
```

Focused checks:

```bash
.venv/bin/python -m pytest tests/test_chrom_pass.py       # Rust writer parity
.venv/bin/python -m pytest tests/test_dataset_staging.py  # staging and deduplication
.venv/bin/python -m pytest tests/test_train_metrics.py    # training metrics
```

Lint before committing (line length 100; rules `E`, `F`, `I`):

```bash
ruff check python/
```

See [CLAUDE.md](CLAUDE.md) for deeper developer notes on the Rust build and writer
internals.

## Repository layout

- `src/` — Rust/PyO3 BigWig + FASTA readers and Arrow writers (`chrom_pass.rs` is the
  production writer).
- `python/regulonado/dataset.py` — dataset construction, transforms, scaling, augmentation.
- `python/regulonado/train.py` — training entrypoint (Hydra + HF `Trainer`).
- `python/regulonado/model/` — backbone adapters and prediction heads.
- `python/configs/` — Hydra configs for backbones, heads, losses, and experiments.
- `scripts/` — Slurm launchers (see `scripts/README.md`); `scripts/experiment/` holds
  production run configs.
- `tests/` — model, dataset, and smoke coverage.
