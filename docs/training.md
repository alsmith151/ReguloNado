# Train and configure models

## Run one training job

The dataset is the positional argument. A preset supplies the phase defaults:

```bash
regulonado train dataset/ --preset head_only --output-dir results/head_only
```

The supported phase presets are:

- `head_only`: train the task head with the backbone frozen.
- `unfreeze_output`: unfreeze the output end of the backbone.
- `deep_finetune`: fine-tune deeper backbone stages.
- `peak_finetune`: finish with the peak-focused objective.

Before starting a long job, inspect the complete resolved configuration:

```bash
regulonado train dataset/ --preset deep_finetune \
  --set trainer.batch_size=4 \
  --set trainer.learning_rate=0.00005 \
  --print-config
```

Repeat `--set KEY=VALUE` for small changes. Unknown keys and invalid value types
are errors. Put repeated, shared settings in YAML rather than building a long
command line.

To use a `tracks.parquet` other than the dataset's own copy — e.g. one with different
scale factors after re-running `tracks assemble` — select it with `--metadata`:

```bash
regulonado train dataset/ --preset head_only \
  --metadata results/tracks/tracks.parquet
```

## Configure several runs

A pipeline configuration separates settings shared by all jobs from phase and
run differences:

```yaml
train:
  common:
    trainer:
      batch_size: 8
      num_workers: 8

  phases:
    - name: head_only
      preset: head_only
    - name: unfreeze_output
      preset: unfreeze_output
    - name: deep_finetune
      preset: deep_finetune
    - name: peak_finetune
      preset: peak_finetune

  runs:
    - name: flashzoi_0
      seed: 0
      pretrained_model: johahi/flashzoi-replicate-0
    - name: flashzoi_1
      seed: 1
      pretrained_model: johahi/flashzoi-replicate-1
    - name: flashzoi_2
      seed: 2
      pretrained_model: johahi/flashzoi-replicate-2
    - name: flashzoi_3
      seed: 3
      pretrained_model: johahi/flashzoi-replicate-3
```

Use `examples/flashzoi_four_replicates.yaml` for the complete file, including
dataset construction and normalization.

Settings are resolved in this order, with later layers winning:

```text
base defaults -> phase preset -> train.common -> phase settings -> run settings
```

For example, change `train.common.trainer.batch_size` to update every job, add a
key under a phase's `settings` mapping to affect only that phase, or add a key
under `flashzoi_2.settings` to affect only that replicate. The run's `seed` and
`pretrained_model` fields always identify that run and take precedence.

## Inspect and run the matrix

```bash
regulonado pipeline my_runs.yaml --dry-run
regulonado pipeline my_runs.yaml --cores 4
```

The dry run prints each run and phase with its model, seed, output directory,
and dependency. Independent runs may execute concurrently. Phases are ordered
within a run and never consume another run's checkpoint.

Before allocating a GPU, resolve a single job against the built Parquet dataset:

```bash
regulonado train results/dataset --preset head_only --schedule-only \
  --set trainer.max_epochs=2 --set trainer.evals_per_epoch=4
```

This reads only dataset metadata and prints the effective batch size, dropped
rows, optimizer updates, warmup fraction, and exact logging/evaluation/checkpoint
steps. Pass the same `--nproc-per-node` intended for the real job.

The Snakemake pipeline performs this preflight automatically for every run and
phase, writes the result to `results/train/<run>/<phase>/schedule.json`, and
allows the GPU-backed `train_phase` job to start only after it succeeds. W&B
sweep trials run the same check inside each allocated agent job before building
their model.

Outputs use `results/train/<run>/<phase>/`. If the workflow stops, fix the
cause and run the same pipeline command again; completed outputs remain valid
and Snakemake schedules only missing or stale work.

## Run a parameter sweep

W&B owns the search space and run bookkeeping. The Snakemake stage only supplies
the dataset dependency and a GPU Slurm job, while each W&B command invokes the
normal `regulonado train` entrypoint. The pipeline YAML does not need a `train:`
matrix when only this stage is being used. Add the stage to the workflow YAML:

```yaml
parameter_sweep:
  enabled: true
  sweep_config: examples/hl60_parameter_sweep_wandb.yaml
  agents: 24
  trials_per_agent: 1
  cpus_per_agent: 4
  mem_mb_per_agent: 64000
  runtime_minutes_per_agent: 240
  wandb_project: regulonado-parameter-sweep
```

Run only this stage on the configured Slurm GPU profile:

```bash
regulonado pipeline hl60_anchor_folds.yaml parameter-sweep --preset sg
```

The workflow registers the sweep once, then submits `agents` independent one-GPU
jobs. Each processes `trials_per_agent` trials and forwards Slurm signals to its
training child. W&B stores the individual runs and metrics; the workflow writes
`results/parameter-sweep/sweep.done` after all agents finish.

For grid searches, `agents: N` with `trials_per_agent: 1` gives maximum parallelism
and isolates every trial in its own allocation. For Bayesian searches, use fewer
longer-lived agents so later suggestions can incorporate earlier results; for a
24-trial sweep, `agents: 8` and `trials_per_agent: 3` is a good starting point.
One-by-one jobs remain preferable when queue policy, wall-time prediction, or
preemption isolation matters more than Bayesian adaptivity.

## Label space

`data.label_space` picks what the loss is fitted against. Each option below is a single
config value, so moving between them needs no other change.

- `transformed` (default): stored signal goes through `apply_scale` (× `scale_factor`
  after subtracting `background`), `apply_clip` and `apply_squash`, and the model predicts
  in that transformed space.
- `counts`: labels stay raw. They are converted to `data.count_unit` and never
  background-subtracted, clipped or squashed (the `apply_*` flags are ignored). The model
  predicts a rate `r` in `data.exposure` units, and the likelihood (the base loss) sees
  `r × exposure` against the counts. This is a GLM offset: sequencing depth is a known
  per-track constant rather than something the head has to learn, and deeper tracks carry
  proportionally more weight in the likelihood. The contrast loss, eval metrics, example
  plots, output-bias init and prediction BigWigs all work in exposure-normalised units
  (`counts / exposure`).

| `data.count_unit` | One label unit | Needs |
|---|---|---|
| `fragments` (default) | `mean_coverage × bin_size / fragment_length` ≈ fragments per bin | `fragment_length` per track, or `scale_library_size` + `fp_genome_sum` |
| `coverage_sum` | summed per-base coverage in the bin (Borzoi's convention) | — |
| `mean_coverage` | the stored value | — |

| `data.exposure` | Output 1.0 means | Needs |
|---|---|---|
| `anchor` (default) | the track's housekeeping-promoter anchor level | `anchor` scaling |
| `anchor_minus_background` | the anchor level above background | `anchor` scaling |
| `library_size` | one unit per million mapped reads | `scale_library_size` |
| `none` | one count unit (the model predicts counts) | — |

Poisson variance matches fragment counts, not coverage. Summed coverage counts each
fragment once for every base it covers, which overstates the information in each bin by
roughly the fragment length. When deriving fragment length from
`fp_genome_sum / scale_library_size`, check what `library_size` counts: paired-end *reads*
give half the fragment length. Supplying `fragment_length` directly (for example through
`inputs.track_annotations`) avoids the ambiguity.

`regulonado tracks fragment-lengths` measures it from the BAMs (named `<bigwig-stem>.bam`)
and writes an annotations file for `tracks assemble --annotations`:

```bash
regulonado tracks fragment-lengths results/tracks/_stages/discovered.parquet \
  --bam-dir path/to/bams -o results/tracks/_stages/fragment_lengths.parquet
```

Paired-end BAMs use the mean proper-pair template length. Single-end BAMs use strand
cross-correlation: the shift at which minus-strand 5' ends best line up with plus-strand
5' ends, excluding the peak at the read length, so fragments within 10 bp of the read
length cannot be resolved. Check the `coverage_per_unit` column, which is the
BigWig's genome-wide coverage divided by the mapped fragments (or reads). It sits near
`fragment_length` when reads were extended to fragments and near `read_length` when they
were not. Re-run with `--length-source read` in the second case, because unextended
coverage converts to reads, not fragments.

Loss weights tuned under `transformed` labels do not carry over. The Poisson and
multinomial terms grow with the count scale, while the contrast terms do not.

`data.mask_missing` (default `true`) removes missing bins (NaN, from datasets built with
`--missing-bins nan`) from every loss term and metric. With `false`, they are trained as
zero signal.

## Cross-track contrast

`loss.contrast_weight`, `loss.contrast_magnitude_weight` and `trainer.contrast_*` (`contrast_region_bins`,
`contrast_pseudocount`, `contrast_active_fraction`) are one coupled knob set, not
two independent ones. The `trainer.contrast_*` values define the region geometry
for cross-track specificity — group-balanced log deviations over the most active
fraction of regions per example — and that same definition is used both by the
`contrast_*` eval metrics (`contrast_pearson_median`, `contrast_sd_ratio_median`)
and, when `loss.contrast_weight > 0`, by the training
loss term. The loss term optimises `1 - mean(r)` of exactly the quantity
`contrast_pearson_median` reports (the loss reduces per-track r with `mean`, the
metric with `median`), so raising `loss.contrast_weight` should move
`contrast_pearson_median` directly rather than a proxy for it.

Regions are ranked by the strongest group's replicate-averaged observed signal, not the
family mean. With 20 groups, a region open in one cell type has a family mean of 1/20 of
its signal and would lose to every shared peak; ranked by its strongest group, it
competes on its own height. Active regions therefore cover cell-type-specific sites in
both directions — open in a track's cell type (positive specificity) and open elsewhere
but not there (negative) — rather than mostly constitutive peaks.

`loss.contrast_weight` optimises `1 - mean(r)`, which is scale-free: predictions that rank
cell-type differences perfectly but shrink them all (`contrast_sd_ratio` below one) cost it
nothing. `loss.contrast_magnitude_weight` adds a Huber loss between predicted and observed
specificity on the same regions, in natural-log units, so predicted effect sizes have to
match in size and not only in order. Set it when the differences are read quantitatively —
synthetic-enhancer design scores designs by how much more active the target group is — and
watch `contrast_sd_ratio_median` move towards one.

## Evaluation metrics

Each metric is computed per track over the whole evaluation set, then reduced to the
median across tracks. W&B shows them as `eval/<metric>`. Every metric answers one
question; the checkpoint-selection objectives combine them.

| Metric | Question | Ideal | Reading a miss |
| --- | --- | --- | --- |
| `loss` | Training objective on held-out windows | lower | Only comparable between runs with the same loss settings |
| `pearson_bin_median` | Is the profile shape right, bin by bin? | 1 | Peaks misplaced or smeared |
| `pearson_top256_median` | Is shape right where the signal is? (top 256 target bins per window) | 1 | Low while `pearson_bin` is high: background fitted, peaks not |
| `pearson_total_median` | Does each window get the right amount of signal relative to other windows? | 1 | Strong and weak loci not told apart |
| `total_ratio_median` | Is overall scale calibrated? (predicted total / observed total) | 1 | Below 1: systematic under-prediction |
| `abs_log_ratio_total_median` | `\|log total_ratio\|`, the calibration error an objective can minimise | 0 | Same as `total_ratio_median`, sign dropped |
| `amplitude_ratio_median` | Are peak heights right? (predicted / observed 99th-percentile bin per window) | 1 | Below 1 with `total_ratio` near 1: peaks flattened into background |
| `dispersion_slope_median` | Is the dynamic range right? (slope of log prediction on log target) | 1 | Below 1: compressed towards the mean; a pure scale error leaves it at 1 |
| `contrast_pearson_median` | Are cell-type differences ranked right? ([Cross-track contrast](#cross-track-contrast)) | 1 | Model predicts a shared profile for every cell type |
| `contrast_sd_ratio_median` | Are cell-type differences the right size? | 1 | Below 1: differences shrunk; a post-hoc stretch by 1/ratio would undo it |
| `contrast_objective` | Selection objective: `abs_log_ratio - contrast_pearson - w * pearson_bin` | lower | Use as `metric_for_best_model` when contrast families exist |
| `calibration_shape_objective` | Selection objective without contrast: `abs_log_ratio - w * pearson_bin` | lower | Fallback when tracks carry no `assay_class`/`group` families |

`w` is `trainer.calibration_shape_pearson_weight`. `dispersion_slope_median` adds
`log_pseudocount` (0.1) to both sides so zero-count bins do not dominate it.

Per-track values behind every median are written to
`per_track_metrics/validation_step_<step>.csv` after each evaluation, with the track's
`group` and `assay_class`, so a median can be broken down without adding W&B panels.

When the dataset has a `test` split, the final model (the best checkpoint when
selection is on) is scored on it once after training. The result goes to
`training_summary.json` under `test_metrics`, to `per_track_metrics/test.csv`, and to
the W&B run summary as `test/<metric>` with one `test/per_track_metrics` table. It
never reaches early stopping or checkpoint selection.

## Read the output

A completed run directory contains the resolved training configuration and
provenance alongside Hugging Face checkpoints. Each checkpoint keeps
`config.json` with `model.safetensors`; trainer state records evaluation,
checkpoint, and resume progress. Treat the whole checkpoint directory as one
artifact rather than moving the weights file alone.

Evaluation metrics and any configured reporting backend are controlled by the
resolved `trainer` settings. Check the resolved configuration saved with a run
before comparing experiments so that differences in seeds, model identifiers,
batch sizes, and learning rates are explicit.

W&B training runs default to the `regulonado-training` project. Pipeline runs
are grouped as `<results_dir-name>/<fold>`, named `<fold>/<phase>`, and use the
phase name as their job type. Override `trainer.wandb_project`,
`trainer.wandb_group`, `trainer.wandb_job_type`, `trainer.wandb_run_name`, or
`trainer.wandb_tags` under `train.common` when a different organization is
needed. Parameter sweeps default to `regulonado-parameter-sweep`, while design
runs default to `regulonado-design`.

Training several independent folds this way is also what
[synthetic enhancer design](design.md) needs: it optimises against some folds
and scores the result on a held-out one.
