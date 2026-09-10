# Run the pipeline on Slurm

Install the workflow and executor support:

```bash
pip install "regulonado[train,workflow]" snakemake-executor-plugin-slurm
```

## Install the execution presets

Snakemake profiles live in `~/.config/snakemake/profile_<words>/config.yaml`.
Install the ones ReguloNado ships with:

```bash
regulonado init
```

This writes three presets:

| Shortcode | Profile directory | Use |
| --- | --- | --- |
| `le` | `profile_local_environment` | Local execution; dry runs and smoke tests |
| `sg` | `profile_slurm_gpu` | Slurm, GPU training in the ambient environment |
| `ssg` | `profile_slurm_singularity_gpu` | As `sg`, but rules run under Apptainer |

The shortcode is the initials of the words after `profile_`. Presets that
already exist are left alone, so `regulonado init` is safe to re-run; add
`--dry-run` to see what it would write, or `--force` to overwrite.

This is the same directory and naming scheme SeqNado uses. If SeqNado is
installed, its presets (`le`, `lc`, `ls`, `ld`, `ss`, `a`) resolve here too and
the two tools share one set of execution configs. Where a shortcode is claimed
by both, ReguloNado's wins, because its resource requests are the ones sized for
this workflow. Selecting a SeqNado preset that has no `set-resources` for
`build_dataset` or `train_phase` prints a warning: its defaults (`mem: 3G`,
`runtime: 1h`) are sized for alignment jobs and will starve a training job.

## Edit the Slurm preset once

Open `~/.config/snakemake/profile_slurm_gpu/config.yaml` and replace every
`CHANGEME_*` placeholder — they are site-specific and will not exist on your
cluster:

- `default-resources.slurm_partition`: `CHANGEME_CPU_PARTITION`
- `default-resources.slurm_account`: `CHANGEME_ACCOUNT`
- `set-resources.train_phase.slurm_partition`: `CHANGEME_GPU_PARTITION`
- `set-resources.train_phase.slurm_account`: `CHANGEME_GPU_ACCOUNT`

Then review runtime, memory, CPU count, GPU type, and maximum concurrent jobs.
Because this lives in your home directory, you edit it once and reuse it across
every project.

Use `profile_slurm_singularity_gpu` (`ssg`) instead if rules should run inside
an Apptainer image; it carries the same scheduling settings plus
`software-deployment-method: apptainer`.

## Resources

`default-resources` covers the light rules (`track_discovery`, `track_qc`, `track_assemble`,
`scale_factors`) at 240 minutes, 16 GB, and 1 CPU. The heavy rules are called out individually:

| Rule | Runtime | Memory | CPUs | Notes |
| --- | --- | --- | --- | --- |
| `build_dataset` | 1440 min | 256 GB | 32 | Memory- and I/O-hungry rather than GPU-bound |
| `recompress_dataset` | 720 min | 128 GB | 8 | |
| `train_phase` | 1440 min | 256 GB | 16 | GPU rule; own partition and account |

Snakemake's Slurm executor has no generic GPU resource, so the GRES request is
passed through verbatim in `slurm_extra`:

```yaml
  train_phase:
    slurm_extra: "'--gres=gpu:2'"
```

The inner quotes are part of the value: Snakemake forwards the string to
`sbatch`, and without them the argument is split. Adjust the GPU count and add a
type where your cluster requires one, for example `'--gres=gpu:a100:2'`. Match
your site's GRES naming — a wrong type name is rejected at submission.

`regulonado pipeline` also honours a profile's `use-conda`, `use-apptainer`,
`software-deployment-method`, `apptainer-args`, `conda-prefix`, and `retries`
keys.

## Run

Validate both the experiment matrix and the workflow DAG before submitting:

```bash
regulonado pipeline my_runs.yaml --dry-run --preset sg
```

Submit with:

```bash
regulonado pipeline my_runs.yaml --preset sg
```

There is no `-p` short flag for `--preset`: `regulonado train -p` already means
a *training* preset (`head_only`, `deep_finetune`), which is unrelated to
execution.

`--profile <path>` still works and takes precedence over `--preset`. Use it for
a one-off profile that should not be installed globally — a reservation for a
single set of runs, or a profile checked into a project repository:

```bash
regulonado pipeline my_runs.yaml --profile ./reservation-profile
```

The profile controls scheduling resources; the experiment YAML controls models,
seeds, training phases, and hyperparameters. Keep those concerns separate so
the same experiment can run on another cluster with a different profile.

Four FlashZoi runs can occupy four workers at once, subject to the profile's
job and resource limits. A run's later phases remain blocked until its previous
phase has produced a valid trainer state.

## Troubleshooting

After a failed job, inspect its Slurm and training logs, correct the problem,
and submit the same pipeline command. Snakemake resumes from existing outputs.
Use `--keep-going` when independent runs should continue after one run fails.

`regulonado init` reports where each preset was installed; run it with
`--dry-run` if you are unsure which file a shortcode resolves to. A shortcode
that resolves to a preset you have not installed falls back to the packaged
read-only copy, still carrying the `CHANGEME_*` placeholders.
