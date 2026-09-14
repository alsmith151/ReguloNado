# HL-60 calibration sweep

This directory contains the tiered sweep harness described in
`docs/hl-60-track-magnitude-squishy-sparrow.md`.

Tier 1 caches `BorzoiBackboneAdapter.forward_features` outputs and runs head/loss variants on
those tensors. It only evaluates head-only variants; backbone-unfreezing must be confirmed in tier
2. The cache is intentionally written to a user-selected scratch directory because a few hundred
6144-bin windows can occupy tens of gigabytes.

The feature-cache conversion/trunk pass and every tier-1 W&B agent must run on a GPU node. Submit
those jobs with the provided Slurm wrappers; run only the W&B API reporting commands on the login
node.

Typical workflow (edit the `#SBATCH` settings for your cluster first):

```bash
# `CALIBRATION_INPUT` is the GPU-produced tensor payload containing trunk features and labels.
sbatch --export=ALL,CALIBRATION_INPUT=/scratch/hl60-input.pt,CALIBRATION_OUTPUT=/scratch/hl60-cache.pt \
  experiments/calibration-sweep/cache_features.sbatch

SWEEP_ID=$(wandb sweep experiments/calibration-sweep/sweep_tier1.yaml | tail -1)
sbatch --export=ALL,SWEEP_ID="$SWEEP_ID",CALIBRATION_CACHE=/scratch/hl60-cache.pt \
  experiments/calibration-sweep/wandb_agent.sbatch

# These use the W&B API and do not need a GPU.
python experiments/calibration-sweep/analyse.py <entity/project/sweep-id>
python experiments/calibration-sweep/promote.py <entity/project/sweep-id> --top-n 3

# Tier 2 uses the normal SLURM workflow after promotion:
regulonado pipeline examples/hl60_anchor_folds.yaml --preset sg
```

The W&B run config stores Hydra override strings, so promoted configurations remain directly
reproducible with the normal training entrypoint. `results/` is local/archival output and is
gitignored.

`quantile_map.py` provides the post-hoc per-track monotone quantile-map arm for held-out
predictions; it uses NumPy's quantile interpolation and is evaluated with the same magnitude
metrics as training-side variants.
