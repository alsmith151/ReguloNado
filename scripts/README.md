# Scripts

Scripts are thin wrappers around the `regulonado` CLI for cluster workflows.
Training policy should live in Hydra configs under `python/configs/experiment/`;
scripts should only handle paths, Slurm resources, CUDA modules, and other
machine-specific plumbing.

## Training

- `train_condition_agnostic_slurm.sh`: Slurm launcher for an explicit Hydra experiment.
  Set `EXPERIMENT`, `DATA_DIR`, `RUN_DIR`, and checkpoint reuse variables as
  needed. Pass one-off Hydra overrides as script arguments.

## Dataset Build And Preparation

- `build_dataset_slurm.sh`: Slurm launcher for the production dataset builder.
- `rechunk_dataset_slurm.sh`: Slurm wrapper around
  `regulonado recompress-dataset`.
- `smoketest_chrom_pass.sh`: Small real-input smoke test for the Rust
  chromosome-pass writer.

## Scaling

- `calculate_original_scaling_slurm.sh`: Slurm wrapper for
  `regulonado calculate-original-scaling`, with optional metadata enrichment.
- `calculate_tmm_scaling_slurm.sh`: Slurm wrapper for
  `regulonado calculate-tmm-scaling`, with optional metadata enrichment.

## Environment

- `install_gpu_env_slurm.sh`: CUDA-node installer for GPU extras such as
  FlashAttention. Use `INSTALL_EXTRAS="--extra dev --extra gpu"` when tests
  should be installed too.

Redundant training wrappers should not be added here. Prefer `regulonado train`
locally and `train_condition_agnostic_slurm.sh` on Slurm.
