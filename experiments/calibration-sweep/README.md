# HL-60 calibration sweep

This directory contains the tiered sweep harness described in
`docs/hl-60-track-magnitude-squishy-sparrow.md`.

Tier 1 caches `BorzoiBackboneAdapter.forward_features` outputs and runs head/loss variants on
those tensors. It only evaluates head-only variants; backbone-unfreezing must be confirmed in tier
2. The cache is intentionally written to a user-selected scratch directory because a few hundred
6144-bin windows can occupy tens of gigabytes.

Typical workflow:

```bash
python experiments/calibration-sweep/cache_features.py \
  data.path=/path/to/dataset data.cache_dir=/scratch/hl60-cache \
  cache.num_windows=300
wandb sweep experiments/calibration-sweep/sweep_tier1.yaml
wandb agent <sweep-id>
python experiments/calibration-sweep/analyse.py <entity/project/sweep-id>
python experiments/calibration-sweep/promote.py <entity/project/sweep-id> --top-n 3
```

The W&B run config stores Hydra override strings, so promoted configurations remain directly
reproducible with the normal training entrypoint. `results/` is local/archival output and is
gitignored.

`quantile_map.py` provides the post-hoc per-track monotone quantile-map arm for held-out
predictions; it uses NumPy's quantile interpolation and is evaluated with the same magnitude
metrics as training-side variants.
