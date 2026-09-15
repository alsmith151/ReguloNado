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


def make_preprocess_logits_for_metrics(topk_bins: int) -> Callable:
    """Return a preprocess_logits_for_metrics function that accumulates Pearson sufficient stats.

    Returns [B, T, 25] per batch:
      cols 0-5:  (sum_p, sum_t, sum_pt, sum_p², sum_t², n)  over all bins  — per-bin Pearson
      cols 6-11: same statistics restricted to the top-K bins by target signal
      cols 12-17: (sp, st, sp*st, sp², st², 1.0)  where sp/st are per-example
                  track totals — sufficient stats for pearson_total_median
      cols 18-19: per-example q99 prediction and target values
      cols 20-24: (sum_log_p, sum_log_t, sum_log_p*log_t, sum_log_p², n)
                   over bins — sufficient stats for dispersion_slope
    """

    def preprocess(logits: torch.Tensor | tuple, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        p = logits
        # Labels may be loaded as [B, L, T] by the HF datasets library; align to [B, T, L].
        t = labels if labels.shape[-2:] == p.shape[-2:] else labels.transpose(-2, -1)
        B, T, L = p.shape
        k = min(topk_bins, L)

        sp = p.sum(-1)
        st = t.sum(-1)
        spt = (p * t).sum(-1)
        sp2 = (p * p).sum(-1)
        st2 = (t * t).sum(-1)
        n = torch.full((B, T), float(L), dtype=p.dtype, device=p.device)

        topk_idx = t.topk(k, dim=-1).indices  # [B, T, k]
        p_k = p.gather(-1, topk_idx)
        t_k = t.gather(-1, topk_idx)
        sp_k = p_k.sum(-1)
        st_k = t_k.sum(-1)
        spt_k = (p_k * t_k).sum(-1)
        sp2_k = (p_k * p_k).sum(-1)
        st2_k = (t_k * t_k).sum(-1)
        n_k = torch.full((B, T), float(k), dtype=p.dtype, device=p.device)

        # Total-signal sufficient stats: sp, st are per-example track totals (summed over bins).
        # When accumulated across N eval examples: sum gives cross-example Pearson sufficient stats.
        ones = torch.ones((B, T), dtype=p.dtype, device=p.device)
        q99_p = torch.quantile(p.float(), 0.99, dim=-1).to(dtype=p.dtype)
        q99_t = torch.quantile(t.float(), 0.99, dim=-1).to(dtype=p.dtype)
        log_p = torch.log(p.float().clamp_min(1e-8))
        log_t = torch.log(t.float().clamp_min(1e-8))
        log_p_sum = log_p.sum(-1).to(dtype=p.dtype)
        log_t_sum = log_t.sum(-1).to(dtype=p.dtype)
        log_pt_sum = (log_p * log_t).sum(-1).to(dtype=p.dtype)
        log_p2_sum = (log_p * log_p).sum(-1).to(dtype=p.dtype)

        return torch.stack(
            [
                sp,
                st,
                spt,
                sp2,
                st2,
                n,
                sp_k,
                st_k,
                spt_k,
                sp2_k,
                st2_k,
                n_k,
                sp,
                st,
                sp * st,
                sp * sp,
                st * st,
                ones,
                q99_p,
                q99_t,
                log_p_sum,
                log_t_sum,
                log_pt_sum,
                log_p2_sum,
                n,
            ],
            dim=-1,
        )

    return preprocess


def make_compute_metrics(
    n_tracks: int,
    calibration_shape_pearson_weight: float = 0.1,
) -> Callable[[EvalPrediction], dict[str, float]]:
    def _pearson_from_stats(
        sp: np.ndarray,
        st: np.ndarray,
        spt: np.ndarray,
        sp2: np.ndarray,
        st2: np.ndarray,
        n: np.ndarray,
    ) -> np.ndarray:
        num = n * spt - sp * st
        denom = np.sqrt(np.maximum(n * sp2 - sp**2, 0.0) * np.maximum(n * st2 - st**2, 0.0))
        result = np.full_like(num, np.nan, dtype=np.float64)
        return np.divide(num, denom, out=result, where=denom > 0)

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        # predictions: [N, T, 25] — sufficient stats accumulated over the full eval set
        stats = np.asarray(eval_pred.predictions, dtype=np.float64)
        s = stats.sum(axis=0)  # [T, 25] global sums

        r_all = _pearson_from_stats(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5])
        r_topk = _pearson_from_stats(s[:, 6], s[:, 7], s[:, 8], s[:, 9], s[:, 10], s[:, 11])
        r_total = _pearson_from_stats(s[:, 12], s[:, 13], s[:, 14], s[:, 15], s[:, 16], s[:, 17])

        fin_all = r_all[np.isfinite(r_all)]
        fin_topk = r_topk[np.isfinite(r_topk)]
        fin_total = r_total[np.isfinite(r_total)]
        topk_n = int(round(float(stats[0, 0, 11]))) if stats.shape[0] > 0 else 0

        total_ratios = np.divide(
            s[:, 0], s[:, 1], out=np.full(n_tracks, np.nan), where=np.abs(s[:, 1]) > 1e-12
        )
        finite_ratios = total_ratios[np.isfinite(total_ratios) & (total_ratios > 0)]
        log_ratios = np.log(finite_ratios)
        amplitude_ratios = np.divide(
            stats[:, :, 18],
            stats[:, :, 19],
            out=np.full(stats[:, :, 18].shape, np.nan),
            where=np.abs(stats[:, :, 19]) > 1e-12,
        )
        amplitude_ratios = amplitude_ratios[np.isfinite(amplitude_ratios) & (amplitude_ratios > 0)]

        # Regress log(pred) on log(target) over all evaluated bins. A slope below one
        # indicates under-dispersion; a pure scale error has slope approximately one.
        slope_num = s[:, 24] * s[:, 22] - s[:, 20] * s[:, 21]
        slope_den = s[:, 24] * s[:, 23] - s[:, 20] * s[:, 20]
        slopes = np.divide(
            slope_num, slope_den, out=np.full(n_tracks, np.nan), where=slope_den > 1e-12
        )
        finite_slopes = slopes[np.isfinite(slopes)]
        # Keep the historical Pearson metric's NaN-excluding median for reporting,
        # but score undefined (flat) tracks as zero so they cannot disappear from
        # the sweep objective.
        objective_pearson = float(np.median(np.nan_to_num(r_all, nan=0.0)))
        abs_log_ratio = float(abs(np.median(log_ratios))) if log_ratios.size else float("nan")

        return {
            "pearson_bin_median": float(np.median(fin_all)) if fin_all.size else float("nan"),
            f"pearson_top{topk_n}_median": (
                float(np.median(fin_topk)) if fin_topk.size else float("nan")
            ),
            "pearson_total_median": float(np.median(fin_total)) if fin_total.size else float("nan"),
            "total_ratio_median": float(np.median(finite_ratios))
            if finite_ratios.size
            else float("nan"),
            "log_ratio_total_median": float(np.median(log_ratios))
            if log_ratios.size
            else float("nan"),
            "abs_log_ratio_total_median": abs_log_ratio,
            "calibration_shape_objective": (
                abs_log_ratio - calibration_shape_pearson_weight * objective_pearson
                if np.isfinite(abs_log_ratio)
                else float("nan")
            ),
            "amplitude_ratio_median": float(np.median(amplitude_ratios))
            if amplitude_ratios.size
            else float("nan"),
            "dispersion_slope": float(np.median(finite_slopes))
            if finite_slopes.size
            else float("nan"),
        }

    return compute_metrics
