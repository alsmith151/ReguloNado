from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from transformers import EvalPrediction


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(x, y)[0, 1])
    return float("nan") if not np.isfinite(r) else r


def _make_preprocess_logits_for_metrics(topk_bins: int) -> Callable:
    """Return a preprocess_logits_for_metrics function that accumulates Pearson sufficient stats.

    Returns [B, T, 18] per batch:
      cols 0-5:  (sum_p, sum_t, sum_pt, sum_p², sum_t², n)  over all bins  — per-bin Pearson
      cols 6-11: same statistics restricted to the top-K bins by target signal
      cols 12-17: (sp, st, sp*st, sp², st², 1.0)  where sp/st are per-example track totals
                  — sufficient stats for pearson_total_median (Pearson of track sums across examples)
    """
    def preprocess(logits: torch.Tensor | tuple, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        p = logits
        # Labels may be loaded as [B, L, T] by the HF datasets library; align to [B, T, L].
        t = labels if labels.shape[-2:] == p.shape[-2:] else labels.transpose(-2, -1)
        B, T, L = p.shape
        k = min(topk_bins, L)

        sp   = p.sum(-1)
        st   = t.sum(-1)
        spt  = (p * t).sum(-1)
        sp2  = (p * p).sum(-1)
        st2  = (t * t).sum(-1)
        n    = torch.full((B, T), float(L), dtype=p.dtype, device=p.device)

        topk_idx = t.topk(k, dim=-1).indices  # [B, T, k]
        p_k = p.gather(-1, topk_idx)
        t_k = t.gather(-1, topk_idx)
        sp_k  = p_k.sum(-1)
        st_k  = t_k.sum(-1)
        spt_k = (p_k * t_k).sum(-1)
        sp2_k = (p_k * p_k).sum(-1)
        st2_k = (t_k * t_k).sum(-1)
        n_k   = torch.full((B, T), float(k), dtype=p.dtype, device=p.device)

        # Total-signal sufficient stats: sp, st are per-example track totals (summed over bins).
        # When accumulated across N eval examples: sum gives cross-example Pearson sufficient stats.
        ones  = torch.ones((B, T), dtype=p.dtype, device=p.device)

        return torch.stack(
            [sp, st, spt, sp2, st2, n,
             sp_k, st_k, spt_k, sp2_k, st2_k, n_k,
             sp, st, sp * st, sp * sp, st * st, ones],
            dim=-1,
        )
    return preprocess


def _make_compute_metrics(
    n_tracks: int,
) -> Callable[[EvalPrediction], dict[str, float]]:
    def _pearson_from_stats(
        sp: np.ndarray, st: np.ndarray, spt: np.ndarray,
        sp2: np.ndarray, st2: np.ndarray, n: np.ndarray,
    ) -> np.ndarray:
        num   = n * spt - sp * st
        denom = np.sqrt(
            np.maximum(n * sp2 - sp ** 2, 0.0) * np.maximum(n * st2 - st ** 2, 0.0)
        )
        return np.where(denom > 0, num / denom, np.nan)

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        # predictions: [N, T, 18] — sufficient stats accumulated over the full eval set
        stats = np.asarray(eval_pred.predictions, dtype=np.float64)
        s = stats.sum(axis=0)  # [T, 18] global sums

        r_all   = _pearson_from_stats(s[:,0], s[:,1], s[:,2], s[:,3], s[:,4],  s[:,5])
        r_topk  = _pearson_from_stats(s[:,6], s[:,7], s[:,8], s[:,9], s[:,10], s[:,11])
        r_total = _pearson_from_stats(s[:,12], s[:,13], s[:,14], s[:,15], s[:,16], s[:,17])

        fin_all   = r_all[np.isfinite(r_all)]
        fin_topk  = r_topk[np.isfinite(r_topk)]
        fin_total = r_total[np.isfinite(r_total)]
        topk_n    = int(round(float(stats[0, 0, 11]))) if stats.shape[0] > 0 else 0

        return {
            "pearson_bin_median": float(np.median(fin_all)) if fin_all.size else float("nan"),
            f"pearson_top{topk_n}_median": (
                float(np.median(fin_topk)) if fin_topk.size else float("nan")
            ),
            "pearson_total_median": float(np.median(fin_total)) if fin_total.size else float("nan"),
        }
    return compute_metrics
