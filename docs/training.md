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
