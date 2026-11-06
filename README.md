# ReguloNado

ReguloNado builds sequence-to-function datasets from a reference genome and
BigWig or BAM files, then trains models on coverage profiles or region-level
counts. Supported backbones include Borzoi, FlashZoi, Enformer, and AlphaGenome.
Region-count models can use cached embeddings or fine-tune a live backbone. The
command line is intended to cover the common path without making you learn the
internal Python API or Hydra syntax.

## Install

Python 3.12 is required.

```bash
pip install "regulonado[data]"       # build BigWig datasets or count BAM events
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

## Stability

At version 0.x, the stable public interfaces are:

- **Command-line interface** (`regulonado` commands)
- **On-disk file formats** (`tracks.parquet`, dataset Arrow files, saved checkpoints)

The Python API beyond the five entry points—`build_dataset`, `load_model_for_inference`,
`read_track_table`, `RegionPredictor`, and `run_training`—is experimental and may change
without notice. Submodule internals (`regulonado.model`, `regulonado.training`, etc.) are
not part of the stable API.

## Working alongside SeqNado

[SeqNado](https://github.com/Milne-Group/SeqNado) produces the BigWigs and BAMs
ReguloNado trains on. The two tools share one set of Snakemake execution presets
in `~/.config/snakemake/`, one genome registry in
`~/.config/seqnado/genome_config.json`, and one sample-sheet vocabulary. With
the optional extra installed, ReguloNado reads track paths, BAMs, per-sample
annotation and spike-in normalisation factors straight out of a SeqNado output
directory.

```bash
pip install "regulonado[seqnado]"
regulonado init                                  # install the shared presets
regulonado config --from-seqnado expA=/data/expA/seqnado_output --genome hg38
regulonado pipeline config.yaml --preset sg
```

`regulonado config` writes `config.yaml` plus a `track_sheet.csv` derived from
the project; add `source` and `timepoint_minutes` to that sheet, since SeqNado
has no equivalent columns. Draw on several projects by repeating the flag:

```bash
regulonado config --from-seqnado expA=/data/expA/seqnado_output \
                  --from-seqnado expB=/data/expB/seqnado_output
```

See [Work alongside SeqNado](docs/seqnado-interop.md) for the sheet columns,
what aggregation guarantees, and what is shared rather than duplicated.

## A small end-to-end run

For coverage-profile models, discover tracks from a directory of BigWigs,
assemble them into a track table, then build an
Arrow dataset from a BED file, an indexed FASTA, and that table:

```bash
regulonado tracks discover results/tracks/_stages/discovered.parquet --bigwig-dir bigwigs/
regulonado tracks assemble results/tracks/_stages/discovered.parquet \
  --output results/tracks/tracks.parquet

regulonado dataset intervals.bed genome.fa dataset/ \
  --track-table results/tracks/tracks.parquet \
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

## Work directly with BAM files

For region-count models, start from aligned BAMs and a BED or Parquet region
table. A track sheet supplies each sample's `bam`, `group`, and `assay`
(ATAC-seq, ChIP-seq, or CUT&RUN). BAM paths are relative to the sheet; use
`--bam-dir` to resolve samples whose paths are omitted.

```bash
regulonado tracks discover discovered.parquet --format bam --track-sheet tracks.csv
regulonado tracks assemble discovered.parquet --output tracks.parquet
regulonado counts regions regions.bed --output region_set.parquet \
  --val-chroms chr8 --test-chroms chr9
regulonado counts bam region_set.parquet --tracks tracks.parquet \
  --anchor-regions anchors.bed --background-regions background.bed \
  --out-dir counts_cache
regulonado counts gather region_set.parquet --tracks tracks.parquet \
  --out-dir counts_cache --dataset-dir region_dataset
```

ATAC-seq targets count Tn5-shifted insertions; ChIP-seq and CUT&RUN targets count
fragments. Counting is resumable, with one cached result per track. Anchor and
background regions provide count-based normalization factors.

In the pipeline, set `targets.region_counts` and configure runs with
`target: region_counts`. Choose `trunk: cached` to train on shared pretrained
embeddings, or `trunk: live` to fine-tune the backbone. BAM-only region-count
runs do not require BigWigs. See [Region-count models](docs/region-counts.md)
for track-sheet fields, embedding commands, and complete training examples.

## Run the Snakemake pipeline

Use `regulonado pipeline` when you want one reproducible DAG that builds the
dataset, computes normalization, and runs every configured training phase. It
is not the SeqNado FASTQ pipeline: ReguloNado starts with a reference FASTA,
interval BED or Parquet region table, and BigWig or BAM tracks (supplied directly,
through a track sheet, or from a SeqNado project).

Create or edit a workflow config, then inspect the DAG before submitting it:

```bash
regulonado config -o config.yaml
regulonado pipeline config.yaml --dry-run --cores 4
```

For a cluster, choose a Snakemake execution preset; `--preset` selects where
jobs run, while the `train.recipes` and `train.runs` entries select what is
trained:

```bash
regulonado pipeline config.yaml --preset sg --cores 4
```

For coverage-profile models, the workflow contains these stages:

```text
track_discovery
    ├─ scale_factors ─┐
    └─ track_qc ───────┴─ track_assemble
                            └─ build_dataset
                                └─ train_phase for each run and phase
```

For this profile path, track discovery, scaling, and QC all run **before** the dataset
build, directly from BigWigs —
`track_assemble` merges them into `tracks.parquet`, the one file `build_dataset` and every later
stage reads. `build_dataset` reads the FASTA, BED, and that table, and writes a Hugging Face Parquet
dataset. The scaling stage writes per-track factors; its method is selected by `scaling.method`
(`original`, `tmm`, `bamnado`, `seqnado`, or `anchor`). QC is opt-in (`qc.checks`) and can drop
tracks at assembly rather than training on them. Each `train_phase` runs one training preset. Phases
are sequential within a run (later phases warm-start from the previous checkpoint), while separate
runs can execute concurrently.

For region-count runs (`target: region_counts`), the workflow discovers BAM tracks,
counts events over a shared region table, gathers the count dataset, and prepares
any required embedding caches before training. Mixed profile and region-count
runs share one track table containing both BigWig and BAM paths; see
[Region-count models](docs/region-counts.md).

The pipeline does not align reads or call peaks; those are upstream SeqNado work. Prediction
BigWigs are optional via `prediction:`. It also
does not replace the standalone commands: use `regulonado dataset`,
`regulonado normalization ...`, or `regulonado train` when you need to run one
stage manually. Snakemake records outputs under `results_dir` and skips stages
whose declared outputs already exist, so rerunning the same command resumes
completed work.

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

Prediction can be an explicit standalone command:

```bash
regulonado predict results/train/flashzoi_0/peak_finetune/checkpoint-N \
  genome.fa predictions/ --bed regions.bed
```

Or make it the final pipeline stage. Omit `tracks` to write every dataset track;
set it to the named tracks you want:

```yaml
prediction:
  run: flashzoi_0
  tracks: [H3K27ac, CTCF]  # omit for all tracks
  whole_genome: true
  chromsizes: hg38.chrom.sizes
```

## Guides

- [Build a dataset](docs/building-datasets.md)
- [The track table](docs/track-table.md)
- [Track QC](docs/qc.md)
- [Work alongside SeqNado](docs/seqnado-interop.md)
- [Calculate and apply normalization](docs/normalization.md)
- [Train and change configuration](docs/training.md)
- [Count BAM events and train region-count models](docs/region-counts.md)
- [Run on Slurm](docs/slurm.md)
- [Resume or reuse checkpoints](docs/checkpoints.md)
- [Generate predictions](docs/prediction.md)
- [Design synthetic enhancers](docs/design.md)
- [Numeric-conditioned diffusion and gradient design](docs/generation.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)

Run `regulonado --help` or `regulonado COMMAND --help` for the complete option
reference. ReguloNado is released under the [BSD 3-Clause License](LICENSE).
