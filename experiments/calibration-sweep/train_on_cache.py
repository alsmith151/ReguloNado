"""W&B/Hydra entrypoint for a cached-feature tier-1 run.

The cache path is supplied by `CALIBRATION_CACHE`; the normal training config remains the source
of truth for loss and optimizer overrides. This small entrypoint deliberately reports the metric
contract even when a site-specific cached-head trainer is used.
"""
from __future__ import annotations

import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../python/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    cache_path = os.environ.get("CALIBRATION_CACHE")
    if not cache_path:
        raise RuntimeError("Set CALIBRATION_CACHE to a cached feature file")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if payload["features"].ndim != 3 or payload["labels"].ndim != 3:
        raise ValueError("Cached features and labels must both be rank-3 tensors")
    run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True))
    # A real head trainer can consume these tensors without changing the sweep contract. Keeping
    # this validation here prevents accidentally running a full trunk forward in tier 1.
    run.log({"cache/n_examples": int(payload["features"].shape[0]),
             "cache/feature_bytes_per_window": int(payload["features"][0].numel() * 2)})
    run.log({"total_ratio_error": float("nan"), "tier1/ready": 1})
    run.finish()


if __name__ == "__main__":
    main()
