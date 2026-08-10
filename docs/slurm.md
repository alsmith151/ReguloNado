# Run the pipeline on Slurm

Install the workflow and executor support:

```bash
pip install "regulonado[train,workflow]" snakemake-executor-plugin-slurm
```

Copy the example profile so cluster-specific changes stay outside the package:

```bash
mkdir -p profiles
cp -R examples/slurm profiles/regulonado-slurm
```

Edit `profiles/regulonado-slurm/config.yaml`. At minimum, replace the placeholder
CPU and GPU partitions and accounts. Then review runtime, memory, CPU count, GPU
type, and maximum concurrent jobs.

Validate both the experiment matrix and the workflow DAG before submitting:

```bash
regulonado pipeline my_runs.yaml --dry-run \
  --profile profiles/regulonado-slurm
```

Submit with:

```bash
regulonado pipeline my_runs.yaml --profile profiles/regulonado-slurm
```

The profile controls scheduling resources; the experiment YAML controls models,
seeds, training phases, and hyperparameters. Keep those concerns separate so
the same experiment can run on another cluster with a different profile.

Four FlashZoi runs can occupy four workers at once, subject to the profile's
job and resource limits. A run's later phases remain blocked until its previous
phase has produced a valid trainer state.

After a failed job, inspect its Slurm and training logs, correct the problem,
and submit the same pipeline command. Snakemake resumes from existing outputs.
Use `--keep-going` when independent runs should continue after one run fails.
