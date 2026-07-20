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
        config.json + model.safetensors    ← self-contained HF checkpoint
                       ▼
            regulonado predict             ← BigWig tracks from any region or whole genome
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
- [Prediction](#prediction)
- [Run outputs](#run-outputs)
- [External dependencies](#external-dependencies)
- [License](#license)
- [Development & tests](#development--tests)
- [Repository layout](#repository-layout)

**Additional documentation:**
- [ARCHITECTURE.md](ARCHITECTURE.md) — design rationale: what lives in Rust vs Python and why.
- [CONTRIBUTING.md](CONTRIBUTING.md) — developer setup, build instructions, testing, code style.

## Installation

ReguloNado ships as layered extras so you only install what a given task needs.
Prebuilt wheels are published for Linux (x86_64, aarch64) and macOS (x86_64, aarch64) on Python 3.12 and 3.13,
so most users need no Rust toolchain. Building from source (or on platforms without wheels) requires a working
Rust toolchain.

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

The full pipeline orchestrates: **build → recompress → scale factors → enrich metadata → train (multi-phase) → predict (optional)**.

### Locally

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

### On a cluster (Snakemake)

The workflow orchestrates all steps end-to-end. Copy `config/config.yaml`, edit the paths
and experiment config, then run:

```bash
snakemake --configfile config/config.yaml -n                        # dry run, prints the DAG
snakemake --configfile config/config.yaml --cores 8                 # run locally
snakemake --configfile config/config.yaml --profile workflow/profiles/slurm   # run on SLURM
```

Cluster specifics (partition, account, GPU type) live in `workflow/profiles/slurm/config.yaml`,
not in the workflow itself, so the pipeline is portable between sites. You will need to
install `snakemake` and (for cluster runs) `snakemake-executor-plugin-slurm`:

```bash
pip install snakemake snakemake-executor-plugin-slurm
```

## Building a dataset

Build datasets locally or via the Snakemake workflow. The CLI is the same either way.

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
  fewer BigWig seeks.

Run `regulonado build --help` for the full set of context-length, bin-size,
threading, compression, shard sizing, and deduplication options.

### Recompression

To rechunk an existing dataset into small ZSTD batches for faster random-access reads during training:

```bash
regulonado recompress-dataset /path/to/src /path/to/dst --max-batch-size 4
```

The Snakemake workflow orchestrates this automatically when enabled in `config/config.yaml`.

## Scale factors

Scale factors convert per-track BigWig signal (typically RPKM) to raw read counts
before loss computation. Compute and apply them after building:

```bash
regulonado calculate-original-scaling dataset/regulonado_metadata.json
regulonado calculate-tmm-scaling dataset/regulonado_metadata.json        # optional TMM correction
regulonado enrich-metadata dataset/regulonado_metadata.json dataset/scale_factors.parquet
```

- `calculate-original-scaling` reads BigWig header metadata to infer library sizes
  and the RPKM→raw-count factor per track. This command requires the BamNado binary
  on PATH and cannot be installed via pip (see [BamNado](#bamnado) below).
- `calculate-tmm-scaling` (the workflow default) layers an edgeR-style TMM normalisation
  on top, estimated from the Arrow shards. It has no external dependencies.
- `enrich-metadata` writes the resulting `scale_factor` / `clip_soft` / `clip_hard`
  into `final_track_records`, which `train.py` reads at training time.

## Training

### Local runs

```bash
regulonado train /path/to/dataset \
  --output-dir outputs/train/quick-check \
  --max-steps 1000 --batch-size 8

regulonado train /path/to/dataset --nproc-per-node 2     # multi-GPU via torchrun
regulonado train /path/to/dataset --max-steps 10 --no-wandb   # smoke test
```

Raw Hydra overrides can be appended:

```bash
regulonado train /path/to/dataset trainer.max_steps=2000
```

### Multi-phase training

Training works best in phases, each warm-starting from the previous one (model
weights only, fresh optimizer). The Snakemake workflow orchestrates this automatically,
reading phase definitions from `config/config.yaml`. Each phase can use a different
experiment config.

If running phases manually, use `--init-weights-from-checkpoint` for warm-start:

```bash
# Phase 1 — head only, backbone frozen
regulonado train /path/to/dataset \
  --experiment head_only_borzoi \
  --output-dir outputs/train/phase1

# Phase 2 — unfreeze 2 stages, warm-start from phase 1
regulonado train /path/to/dataset \
  --experiment stage2_unfreeze2_borzoi \
  --init-weights-from-checkpoint outputs/train/phase1/checkpoint-NNNN \
  --output-dir outputs/train/phase2

# Phase 3 — deeper unfreeze + RC augmentation
regulonado train /path/to/dataset \
  --experiment stage3_deep_finetune_borzoi \
  --init-weights-from-checkpoint outputs/train/phase2/checkpoint-NNNN \
  --output-dir outputs/train/phase3

# Phase 4 (optional) — peak sharpening with top-K loss
regulonado train /path/to/dataset \
  --experiment stage4_peak_finetune_borzoi \
  --init-weights-from-checkpoint outputs/train/phase3/checkpoint-NNNN \
  --output-dir outputs/train/phase4
```

The workflow handles checkpoint resolution from each phase's `trainer_state.json` automatically.

## Experiment & loss configs

All hyperparameters live in Hydra YAML. To start a new experiment, copy the
nearest config in `python/configs/experiment/` and adjust what matters.

| Experiment config | Phase / purpose |
|--------|-------------|
| `head_only_borzoi.yaml` | Phase 1: frozen backbone, head only, lr=1e-3 |
| `stage2_unfreeze2_borzoi.yaml` | Phase 2: 2 output-end stages unfrozen, lr=2e-4/2e-6 |
| `stage3_deep_finetune_borzoi.yaml` | Phase 3: 4 stages + RC augmentation, lr=5e-5/5e-7 |
| `stage4_peak_finetune_borzoi.yaml` | Phase 4: topk_additive loss for peak sharpening |

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

## Checkpoint reuse

Checkpoints saved by `RegulonadoTrainer` are self-contained Hugging Face
`PreTrainedModel` directories (`config.json` + `model.safetensors`). Any saved
checkpoint can be loaded directly for inference:

```python
from regulonado.model import RegulonadoModel
model = RegulonadoModel.from_pretrained("outputs/train/my_run/checkpoint-5000")
```

**Full resume** restores model weights, optimizer, scheduler, and RNG state — use
to continue an interrupted run:

```bash
regulonado train /path/to/dataset \
  --experiment head_only_borzoi \
  --resume-from-checkpoint outputs/train/head_only_borzoi-JOBID/checkpoint-NNNN
```

**Warm start** loads model weights only with a fresh optimizer/scheduler — use when
changing learning rate, scheduler, unfreezing policy, or training objective. Use
`--init-weights-from-checkpoint` to warm-start from a previous phase's best checkpoint.
Experiment configs can also set `init_weights_from_checkpoint` directly in the YAML.

## Prediction

Once a model is trained, `regulonado predict` writes one BigWig per track directly
from the checkpoint, without needing the original Arrow dataset.

### Targeted mode (BED regions)

Predict the central output window centred on each row in a BED file:

```bash
regulonado predict outputs/train/my_run genome.fa predictions/ \
    --bed regions.bed
```

Each window in the BED is centred on `(start + end) / 2` and the model's full
context is extracted around that centre. Only the central prediction region
(`n_pred_bins × bin_size` bp) is written to the BigWig; windows must be spaced far
enough apart that their predicted regions do not overlap.

### Whole-genome mode

Tile every chromosome into adjacent, non-overlapping prediction windows:

```bash
regulonado predict outputs/train/my_run genome.fa predictions/ \
    --whole-genome \
    --chromsizes hg38.chrom.sizes
```

`--chromsizes` selects which chromosomes to tile and sets the BigWig header lengths.
When omitted the FASTA `.fai` index is used (every contig). For whole-genome runs
with many tracks, use `--tracks` to write only the tracks you need.

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--tracks` | all | Comma-separated track names or integer indices |
| `--batch-size` | 4 | Windows per forward pass |
| `--device` | auto | Torch device (`cuda`, `cpu`, `cuda:1`, …) |
| `--rtol` | 0.01 | Relative tolerance for collapsing adjacent equal bins |
| `--inverse-squash` | off | Undo the `(x+1)^0.75` training squash to approximate raw counts |

Legacy run roots that pre-date the HF checkpoint format (no `config.json`) are also
supported; pass `--dataset /path/to/dataset` to supply the metadata:

```bash
regulonado predict outputs/train/my_run genome.fa predictions/ \
    --dataset /path/to/original/dataset \
    --bed regions.bed
```

### Python API

For notebook or script use, `RegionPredictor` wraps the model and FASTA and
exposes a simple call interface:

```python
from regulonado.predict import RegionPredictor, RegionPredictionConfig

predictor = RegionPredictor(
    RegionPredictionConfig(
        checkpoint_dir="outputs/train/my_run",
        fasta_path="genome.fa",
        tracks=["my_track_1", "my_track_2"],   # optional subset
        inverse_squash=True,
    )
)

# Single region — returns a RegionPrediction with .values (n_tracks, n_bins)
pred = predictor("chr1", 1_000_000, 1_010_000)
print(pred.values.shape)      # (2, n_pred_bins)
print(pred.track_names)       # ['my_track_1', 'my_track_2']

# Multiple regions
preds = predictor.predict_many([
    ("chr1", 1_000_000, 1_010_000),
    ("chr2", 5_000_000, 5_010_000),
])

# Long-form records (chrom/start/end/track/value) for downstream analysis
import pandas as pd
df = pd.DataFrame(pred.as_records())
```

`RegionPrediction.bin_starts` / `.bin_ends` give the genomic coordinates of each
bin in the output array.

## Run outputs

Each run writes, into `output_dir` (defaults to the Hydra output dir or the Slurm
wrapper's `RUN_DIR`):

- `config.json` — `RegulonadoConfig` (architecture, geometry, track names/metadata);
  enables `RegulonadoModel.from_pretrained` and `regulonado predict` without any
  external metadata files.
- `model.safetensors` — merged model weights (no `model.` key prefix).
- `training_args.json` — HF `TrainingArguments` serialised alongside the weights.
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

## License

ReguloNado is released under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

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
- `python/regulonado/predict.py` — `RegionPredictor` and `predict_to_bigwig`; BigWig
  generation from a trained checkpoint.
- `python/regulonado/model/` — backbone adapters, prediction heads, `RegulonadoModel`
  (`PreTrainedModel`), and `RegulonadoConfig`.
- `python/configs/` — Hydra configs for backbones, heads, losses, and experiments.
- `workflow/` — Snakemake rules for orchestrating the full pipeline (build → recompress
  → scale factors → enrich → train → predict).
- `workflow/profiles/slurm/` — SLURM executor profile for the workflow; adjust partition,
  account, and GPU resource requests here.
- `scripts/` — Environment setup scripts (`install_gpu_env_slurm.sh`, `install_flash_attn_slurm.sh`,
  `jupyter_slurm.sh`).
- `tests/` — model, dataset, and smoke coverage.
