# Scripts

Scripts are thin wrappers around the `regulonado` CLI for cluster workflows.
All training policy lives in Hydra experiment configs under `scripts/experiment/`
(run-specific) or `python/configs/experiment/` (canonical baselines); scripts
handle only paths, Slurm resources, and machine-specific plumbing.

## Training

### Generic launcher

`train_slurm.sh` is the single training entry point.  Set `EXPERIMENT` and
`DATA_DIR`, then submit:

```bash
EXPERIMENT=head_only_borzoi \
DATA_DIR=/path/to/dataset \
sbatch scripts/train_slurm.sh
```

`EXPERIMENT` selects a Hydra config by name.  The launcher searches both
`python/configs/experiment/` (built-in baselines) and `scripts/experiment/`
(production runs).

One-off Hydra overrides can be appended as script arguments:

```bash
EXPERIMENT=head_only_borzoi DATA_DIR=... \
sbatch scripts/train_slurm.sh trainer.max_steps=2000
```

### Experiment configs

All training hyperparameters live in `scripts/experiment/*.yaml`.  To start a
new experiment, copy the nearest config, rename it, and adjust what matters.

| Config | Purpose |
|--------|---------|
| `head_only_borzoi.yaml` | Phase 1: frozen backbone, head-only at lr=1e-3 for 5 k steps |
| `stage2_unfreeze2_borzoi.yaml` | Phase 2: 2 output-end stages unfrozen, warm-start from phase 1 |
| `stage3_deep_finetune_borzoi.yaml` | Phase 3: 4 stages + RC augmentation, low LR for 15 k steps |
| `stage4_peak_finetune_borzoi.yaml` | Phase 4: topk_additive loss for peak sharpening, warm-start from phase 3 |

### Recommended three-phase workflow

Submit all three phases as a dependency chain with the pipeline script:

```bash
DATA_DIR=/path/to/dataset bash scripts/train_pipeline_slurm.sh
```

This queues three Slurm jobs immediately.  Each phase starts only after the
previous one succeeds and automatically picks up that phase's best checkpoint.

To run phases manually:

```bash
# Phase 1 — head only, fast convergence (~5 k steps)
EXPERIMENT=head_only_borzoi DATA_DIR=... sbatch scripts/train_slurm.sh

# Phase 2 — unfreeze 2 backbone stages, warm-start from best phase-1 checkpoint
EXPERIMENT=stage2_unfreeze2_borzoi \
INIT_WEIGHTS_FROM_CHECKPOINT=outputs/train/head_only_borzoi-JOBID/checkpoint-NNNN \
DATA_DIR=... sbatch scripts/train_slurm.sh

# Phase 3 — deep fine-tune with RC augmentation, warm-start from phase 2
EXPERIMENT=stage3_deep_finetune_borzoi \
INIT_WEIGHTS_FROM_CHECKPOINT=outputs/train/stage2_unfreeze2_borzoi-JOBID/checkpoint-NNNN \
DATA_DIR=... sbatch scripts/train_slurm.sh
```

Configs for stages 2 and 3 read `INIT_WEIGHTS_FROM_CHECKPOINT` from the
environment via `${oc.env:INIT_WEIGHTS_FROM_CHECKPOINT,}`.

## Dataset Build and Preparation

- `build_dataset_slurm.sh`: Slurm launcher for the production dataset builder.
- `rechunk_dataset_slurm.sh`: Slurm wrapper around `regulonado recompress-dataset`.

## Scaling

- `calculate_original_scaling_slurm.sh`: Slurm wrapper for
  `regulonado calculate-original-scaling`, with optional metadata enrichment.
- `calculate_tmm_scaling_slurm.sh`: Slurm wrapper for
  `regulonado calculate-tmm-scaling`, with optional metadata enrichment.

## Environment

- `install_gpu_env_slurm.sh`: CUDA-node installer for GPU extras (FlashAttention
  etc.).  Use `INSTALL_EXTRAS="--extra dev --extra gpu"` to include test
  dependencies.
