# Scripts

Environment-setup helpers for GPU clusters. **These are not the pipeline.**

The data and training pipeline is a Snakemake workflow — see [`workflow/`](../workflow/)
and [`config/config.yaml`](../config/config.yaml). The SLURM launchers that used to live
here (`build_dataset_slurm.sh`, `train_slurm.sh`, `train_pipeline_slurm.sh` and friends)
have been replaced by it:

```bash
snakemake --configfile config/config.yaml -n                       # dry run
snakemake --configfile config/config.yaml --profile workflow/profiles/slurm
```

Cluster specifics — partition, account, GPU type — live in
[`workflow/profiles/slurm/config.yaml`](../workflow/profiles/slurm/config.yaml), not in
these scripts.

Training hyperparameters still live in Hydra experiment configs under
`python/configs/experiment/`; the workflow's `train.phases` list selects which ones run
and in what order.

## What remains here

| Script | Purpose |
| --- | --- |
| `install_gpu_env_slurm.sh` | Install the `[gpu]` extras on a CUDA node. `INSTALL_EXTRAS="--extra dev --extra gpu"` adds test dependencies. |
| `install_flash_attn_slurm.sh` | Fetch a prebuilt FlashAttention wheel, falling back to a source build. |
| `jupyter_slurm.sh` | Launch a Jupyter server on a compute node. |

All three hardcode site-specific SLURM settings and assume a repo-local `.venv`. Edit
them for your cluster before use — unlike the workflow, they are not portable as written.
