# Cluster helper scripts

These scripts help set up an interactive or GPU environment; they do not run
the ReguloNado pipeline.

| Script | Purpose |
| --- | --- |
| `install_gpu_env_slurm.sh` | Install GPU dependencies on a CUDA node |
| `install_flash_attn_slurm.sh` | Install FlashAttention from a wheel or source |
| `jupyter_slurm.sh` | Start Jupyter on a compute node |

They contain site-specific Slurm settings and assume a repository-local
`.venv`. Review their account, partition, modules, and paths before submitting
them.

Use the packaged workflow for dataset and training jobs:

```bash
regulonado pipeline examples/flashzoi_four_replicates.yaml --dry-run
regulonado pipeline examples/flashzoi_four_replicates.yaml \
  --profile examples/slurm
```

See [the Slurm guide](../docs/slurm.md) for profile setup.
