"""Noise-robust per-track signal summaries."""
from __future__ import annotations
import torch

__all__ = ["topk_signal", "signal_summary"]

def topk_signal(preds: torch.Tensor, bins: slice, k: int = 10) -> torch.Tensor:
    window = preds[..., bins]
    if window.shape[-1] == 0:
        raise ValueError("topk_signal cannot operate on an empty bin window")
    return window.topk(min(int(k), window.shape[-1]), dim=-1).values.mean(dim=-1)

def signal_summary(preds: torch.Tensor, bins: slice, k: int = 10) -> dict[str, torch.Tensor]:
    window = preds[..., bins]
    mean = window.mean(dim=-1)
    topk = topk_signal(preds, bins, k)
    return {"mean": mean, "max": window.max(dim=-1).values, "topk_mean": topk,
            "topk_ratio": topk / mean.clamp_min(1e-8)}
