"""Noise-robust per-track signal summaries."""
from __future__ import annotations

import torch

__all__ = ["topk_signal", "signal_summary"]

def topk_signal(preds: torch.Tensor, bins: slice, k: int = 10) -> torch.Tensor:
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    window = preds[..., bins]
    if window.shape[-1] == 0:
        raise ValueError("topk_signal cannot operate on an empty bin window")
    if int(k) > window.shape[-1]:
        raise ValueError(f"k={k} exceeds selected bin count {window.shape[-1]}")
    return window.topk(int(k), dim=-1).values.mean(dim=-1)

def signal_summary(preds: torch.Tensor, bins: slice, k: int = 10) -> dict[str, torch.Tensor]:
    window = preds[..., bins]
    mean = window.mean(dim=-1)
    topk = topk_signal(preds, bins, k)
    return {"mean": mean, "max": window.max(dim=-1).values, "topk_mean": topk,
            "topk_ratio": topk / mean.clamp_min(1e-8)}
