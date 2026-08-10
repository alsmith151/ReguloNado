# ReguloNado

ReguloNado builds sequence-to-function datasets from a reference genome and
BigWig tracks, then fine-tunes Borzoi- or Enformer-based models on them. The
command line is intended to cover the common path without making you learn the
internal Python API or Hydra syntax.

## Install

Python 3.12 or 3.13 is required.

```bash
pip install "regulonado[data]"       # build datasets
pip install "regulonado[train]"      # build and train
pip install "regulonado[train,workflow]" # run the training pipeline
```

For local development, use `uv`:

```bash
uv sync --extra dev --extra train --extra workflow
source .venv/bin/activate
```

The optional `gpu` extra builds FlashAttention and therefore needs a compatible
CUDA toolkit and `nvcc`.

## A small end-to-end run

Build an Arrow dataset from a BED file, an indexed FASTA, and a directory of
BigWigs:

```bash
regulonado build intervals.bed genome.fa dataset/ \
  --bigwig-dir bigwigs/ \
  --split train:fold0,fold1,fold2 \
  --split validation:fold4 \
  --split test:fold3 \
  --stage
```

Inspect the exact training configuration, then run it:

```bash
regulonado train dataset/ --preset head_only --print-config
regulonado train dataset/ --preset head_only --output-dir results/quick-run
```

Change a setting without copying a configuration file by repeating `--set`:

```bash
regulonado train dataset/ --preset head_only \
  --set trainer.batch_size=8 \
  --set trainer.max_steps=1000 \
  --set seed=17
```

`--set` accepts known configuration keys only. Use `--print-config` after adding
overrides when you want to check what will be submitted.

## Four independent FlashZoi runs

The pipeline is the easiest way to train the four published FlashZoi
replicates with different seeds. From a source checkout, copy the example:

```bash
cp examples/flashzoi_four_replicates.yaml my_flashzoi_runs.yaml
regulonado pipeline my_flashzoi_runs.yaml --dry-run
regulonado pipeline my_flashzoi_runs.yaml --cores 4
```

Each entry under `train.runs` names its model and seed. Each run follows the
same ordered phase list:

```text
flashzoi_0: head_only -> unfreeze_output -> deep_finetune -> peak_finetune
flashzoi_1: head_only -> unfreeze_output -> deep_finetune -> peak_finetune
flashzoi_2: head_only -> unfreeze_output -> deep_finetune -> peak_finetune
flashzoi_3: head_only -> unfreeze_output -> deep_finetune -> peak_finetune
```

Runs are independent and can execute at the same time. Within a run, each
phase starts from that run's previous phase. Outputs are kept separate under
`results/train/<run>/<phase>/`, so checkpoints cannot cross between replicates.

Edit `train.common` to change all 16 jobs, a phase entry to change one phase in
every run, or a run entry to change one replicate. See
[Training and configuration](docs/training.md) for precedence and examples.

## Use the output

Prediction is a separate, explicit command. It is not run automatically after
training:

```bash
regulonado predict results/train/flashzoi_0/peak_finetune/checkpoint-N \
  genome.fa predictions/ --bed regions.bed
```

## Guides

- [Build a dataset](docs/building-datasets.md)
- [Calculate and apply normalization](docs/normalization.md)
- [Train and change configuration](docs/training.md)
- [Run on Slurm](docs/slurm.md)
- [Resume or reuse checkpoints](docs/checkpoints.md)
- [Generate predictions](docs/prediction.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)

Run `regulonado --help` or `regulonado COMMAND --help` for the complete option
reference. ReguloNado is released under the [BSD 3-Clause License](LICENSE).
