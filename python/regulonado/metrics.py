from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
)


def paired_group_masks(
    condition_ids: torch.Tensor,
    shared_track_index: torch.Tensor | None = None,
    *,
    baseline_condition_id: int = 0,
    perturbed_condition_id: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    condition_ids = condition_ids.to(dtype=torch.long)
    if shared_track_index is None:
        baseline_mask = condition_ids == baseline_condition_id
        perturbed_mask = condition_ids == perturbed_condition_id
        if baseline_mask.any() and perturbed_mask.any():
            return [(baseline_mask, perturbed_mask)]
        return []

    shared_track_index = shared_track_index.to(device=condition_ids.device, dtype=torch.long)
    pair_masks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group_id in torch.unique(shared_track_index, sorted=True):
        in_group = shared_track_index == group_id
        baseline_mask = in_group & (condition_ids == baseline_condition_id)
        perturbed_mask = in_group & (condition_ids == perturbed_condition_id)
        if baseline_mask.any() and perturbed_mask.any():
            pair_masks.append((baseline_mask, perturbed_mask))
    return pair_masks


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float64)


def _pearson(pred_vals: np.ndarray, target_vals: np.ndarray) -> float:
    """Pearson r via torchmetrics. Callers must pre-filter non-finite and degenerate input."""
    return float(pearson_corrcoef(_as_tensor(pred_vals), _as_tensor(target_vals)))


def _spearman(pred_vals: np.ndarray, target_vals: np.ndarray) -> float:
    """Spearman rho via scipy.

    torchmetrics' ``spearman_corrcoef`` adds ``eps=1e-6`` to the denominator, which biases the
    result by roughly ``eps / std(ranks)``; that reaches ~4e-6 for two-element inputs. scipy is
    kept here so results stay bit-comparable with the previous implementation.
    """
    return float(spearmanr(pred_vals, target_vals).statistic)


def _finite_track_pairs(
    preds: np.ndarray,
    targets: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Split into per-track (index, preds, targets) triples with non-finite entries dropped."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")

    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for track_idx in range(preds.shape[-1]):
        pred_vals = preds[:, track_idx]
        target_vals = targets[:, track_idx]
        finite_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
        pairs.append((track_idx, pred_vals[finite_mask], target_vals[finite_mask]))
    return pairs


def _per_track_metric(
    preds: np.ndarray,
    targets: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_count: int,
    require_variance: bool,
) -> dict[int, float]:
    """Apply ``statistic`` per track, returning NaN for empty or degenerate-variance tracks."""
    metrics: dict[int, float] = {}
    for track_idx, pred_vals, target_vals in _finite_track_pairs(preds, targets):
        if pred_vals.size < min_count:
            metrics[track_idx] = float("nan")
        elif require_variance and (np.std(pred_vals) < 1e-8 or np.std(target_vals) < 1e-8):
            metrics[track_idx] = float("nan")
        else:
            metrics[track_idx] = float(statistic(pred_vals, target_vals))
    return metrics


def _concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def _safe_pearson_from_sums(
    *,
    count: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    if count < 2:
        return float("nan")

    numerator = count * sum_xy - sum_x * sum_y
    denom_x = count * sum_x2 - sum_x * sum_x
    denom_y = count * sum_y2 - sum_y * sum_y
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return float("nan")
    return float(numerator / np.sqrt(denom_x * denom_y))


def delta_log2fc_vectors(
    pred_A: torch.Tensor,
    pred_B: torch.Tensor,
    meas_A: torch.Tensor,
    meas_B: torch.Tensor,
    pseudocount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log2 fold-changes between two groups for predicted and measured.

    Returns (pred_log2fc, meas_log2fc) as 1D arrays. Each input is summed over
    tracks (dim -1) then averaged over samples (dim 1) before computing log2fc.
    """
    def _log2fc(a: torch.Tensor, b: torch.Tensor, ps: float) -> np.ndarray:
        a_mean = a.float().mean(dim=1).sum(dim=-1)
        b_mean = b.float().mean(dim=1).sum(dim=-1)
        return (torch.log2(b_mean + ps) - torch.log2(a_mean + ps)).cpu().numpy()

    return _log2fc(pred_A, pred_B, pseudocount), _log2fc(meas_A, meas_B, pseudocount)


def paired_delta_log2fc_vectors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    condition_ids: torch.Tensor,
    *,
    shared_track_index: torch.Tensor | None = None,
    pseudocount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log2 fold-changes for paired conditions (e.g., baseline vs. perturbed).

    Finds pairs of baseline (condition_id=0) and perturbed (condition_id=1)
    samples, optionally grouped by shared_track_index. For each pair, computes
    log2fc as log2(perturbed) - log2(baseline) averaged across tracks.
    Returns (pred_log2fc, target_log2fc) as concatenated 1D arrays.
    """
    pair_masks = paired_group_masks(condition_ids, shared_track_index)
    if not pair_masks:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for baseline_mask, perturbed_mask in pair_masks:
        pred_baseline = predictions[:, baseline_mask].mean(dim=1)
        pred_perturbed = predictions[:, perturbed_mask].mean(dim=1)
        target_baseline = targets[:, baseline_mask].mean(dim=1)
        target_perturbed = targets[:, perturbed_mask].mean(dim=1)

        pred_lfc = torch.log2(pred_perturbed + pseudocount) - torch.log2(
            pred_baseline + pseudocount
        )
        target_lfc = torch.log2(target_perturbed + pseudocount) - torch.log2(
            target_baseline + pseudocount
        )
        pred_chunks.append(pred_lfc.detach().cpu().numpy().reshape(-1))
        target_chunks.append(target_lfc.detach().cpu().numpy().reshape(-1))

    return np.concatenate(pred_chunks), np.concatenate(target_chunks)


def delta_log2fc_pearson(pred_lfc: np.ndarray, meas_lfc: np.ndarray) -> float:
    """Pearson r between predicted and measured log2 fold-changes.

    Non-finite values are dropped. Returns NaN if fewer than 2 finite points or
    if either array has zero standard deviation.
    """
    pred_lfc = np.asarray(pred_lfc, dtype=np.float64)
    meas_lfc = np.asarray(meas_lfc, dtype=np.float64)
    finite_mask = np.isfinite(pred_lfc) & np.isfinite(meas_lfc)
    pred_lfc = pred_lfc[finite_mask]
    meas_lfc = meas_lfc[finite_mask]

    if len(pred_lfc) < 2 or np.std(pred_lfc) < 1e-8 or np.std(meas_lfc) < 1e-8:
        return float("nan")
    return _pearson(pred_lfc, meas_lfc)


def delta_log2fc_metrics(pred_lfc: np.ndarray, meas_lfc: np.ndarray) -> dict[str, float]:
    """Pearson, Spearman, and top-variance Pearson correlations for delta log2fc.

    Top-variance Pearson is computed over the top 20% of measured fold-changes
    by absolute value. All three return NaN if fewer than 2 finite points or if
    either array has zero standard deviation.
    """
    pred_lfc = np.asarray(pred_lfc, dtype=np.float64)
    meas_lfc = np.asarray(meas_lfc, dtype=np.float64)
    finite_mask = np.isfinite(pred_lfc) & np.isfinite(meas_lfc)
    pred_lfc = pred_lfc[finite_mask]
    meas_lfc = meas_lfc[finite_mask]

    if len(pred_lfc) < 2 or np.std(pred_lfc) < 1e-8 or np.std(meas_lfc) < 1e-8:
        return {
            "pearson": float("nan"),
            "spearman": float("nan"),
            "top_variance_pearson": float("nan"),
        }

    pearson_r = _pearson(pred_lfc, meas_lfc)
    spearman_r = _spearman(pred_lfc, meas_lfc)
    var_thresh = np.percentile(np.abs(meas_lfc), 80)
    top_mask = np.abs(meas_lfc) >= var_thresh
    if top_mask.sum() >= 2 and np.std(pred_lfc[top_mask]) > 1e-8:
        top_r = _pearson(pred_lfc[top_mask], meas_lfc[top_mask])
    else:
        top_r = float("nan")
    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "top_variance_pearson": float(top_r),
    }


def init_validation_metric_state() -> dict[str, Any]:
    """Create an empty state dict for accumulating validation metrics across batches.

    Fields hold lists of arrays (delta_lfc, meas_lfc) and sums for raw reconstruction
    (count, sums, sums of squares, cross-products).
    """
    return {
        "pred_lfc_chunks": [],
        "meas_lfc_chunks": [],
        "raw_count": 0,
        "raw_sum_pred": 0.0,
        "raw_sum_target": 0.0,
        "raw_sum_pred_sq": 0.0,
        "raw_sum_target_sq": 0.0,
        "raw_sum_cross": 0.0,
        "raw_abs_error_sum": 0.0,
        "raw_sq_error_sum": 0.0,
    }


def update_validation_metric_state(
    state: dict[str, Any],
    pred_lfc: np.ndarray | None = None,
    meas_lfc: np.ndarray | None = None,
) -> None:
    """Accumulate log2 fold-changes into state. Non-finite values are filtered out.

    If both arrays are provided, their finite elements (element-wise intersection)
    are appended to state['pred_lfc_chunks'] and state['meas_lfc_chunks'].
    """
    if pred_lfc is not None and meas_lfc is not None:
        pred_lfc = np.asarray(pred_lfc, dtype=np.float64)
        meas_lfc = np.asarray(meas_lfc, dtype=np.float64)
        finite_mask = np.isfinite(pred_lfc) & np.isfinite(meas_lfc)
        if finite_mask.any():
            state["pred_lfc_chunks"].append(pred_lfc[finite_mask])
            state["meas_lfc_chunks"].append(meas_lfc[finite_mask])


def update_validation_reconstruction_state(
    state: dict[str, Any],
    raw_pred: np.ndarray,
    raw_target: np.ndarray,
) -> None:
    """Accumulate raw prediction vs. target statistics for reconstruction metrics.

    Computes and accumulates sums, sums of squares, and cross-products used by
    Pearson, MAE, and RMSE. Non-finite elements are filtered out first.
    """
    raw_pred = np.asarray(raw_pred, dtype=np.float64).reshape(-1)
    raw_target = np.asarray(raw_target, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(raw_pred) & np.isfinite(raw_target)
    if finite_mask.any():
        raw_pred = raw_pred[finite_mask]
        raw_target = raw_target[finite_mask]
        error = raw_pred - raw_target
        state["raw_count"] += int(raw_pred.size)
        state["raw_sum_pred"] += float(raw_pred.sum())
        state["raw_sum_target"] += float(raw_target.sum())
        state["raw_sum_pred_sq"] += float(np.square(raw_pred).sum())
        state["raw_sum_target_sq"] += float(np.square(raw_target).sum())
        state["raw_sum_cross"] += float((raw_pred * raw_target).sum())
        state["raw_abs_error_sum"] += float(np.abs(error).sum())
        state["raw_sq_error_sum"] += float(np.square(error).sum())


def finalize_validation_metric_state(state: dict[str, Any]) -> dict[str, float]:
    """Finalize accumulated state to a dict of metric values (floats, may be NaN).

    Computes delta_lfc metrics (pearson, spearman, top_variance_pearson) and
    reconstruction metrics (raw_pearson, raw_mae, raw_rmse) prefixed by their
    category. Returns balanced_score if both are available, else whichever exists.
    """
    pred_lfc = _concat_chunks(state["pred_lfc_chunks"])
    meas_lfc = _concat_chunks(state["meas_lfc_chunks"])
    delta_metrics = {}
    if pred_lfc.size > 0 and meas_lfc.size > 0:
        delta_metrics = {
            f"delta_lfc/{key}": value
            for key, value in delta_log2fc_metrics(pred_lfc, meas_lfc).items()
        }

    raw_count = int(state["raw_count"])
    raw_pearson = _safe_pearson_from_sums(
        count=raw_count,
        sum_x=float(state["raw_sum_pred"]),
        sum_y=float(state["raw_sum_target"]),
        sum_x2=float(state["raw_sum_pred_sq"]),
        sum_y2=float(state["raw_sum_target_sq"]),
        sum_xy=float(state["raw_sum_cross"]),
    )
    raw_mae = float(state["raw_abs_error_sum"]) / raw_count if raw_count > 0 else float("nan")
    raw_rmse = (
        float(np.sqrt(float(state["raw_sq_error_sum"]) / raw_count))
        if raw_count > 0
        else float("nan")
    )

    metrics = {
        **delta_metrics,
        "reconstruction/raw_pearson": raw_pearson,
        "reconstruction/raw_mae": raw_mae,
        "reconstruction/raw_rmse": raw_rmse,
    }
    # balanced_score averages the delta and reconstruction correlations, falling back to
    # whichever is available. The reconstruction half is only populated if the caller has
    # been feeding update_validation_reconstruction_state; without it raw_pearson is NaN,
    # and previously *neither* branch fired, so the key was silently absent altogether
    # rather than degrading to the delta-only score.
    delta_pearson = delta_metrics.get("delta_lfc/pearson", float("nan"))
    if np.isfinite(delta_pearson) and np.isfinite(raw_pearson):
        metrics["selection/balanced_score"] = float(0.5 * (delta_pearson + raw_pearson))
    elif np.isfinite(raw_pearson):
        metrics["selection/balanced_score"] = raw_pearson
    elif np.isfinite(delta_pearson):
        metrics["selection/balanced_score"] = float(delta_pearson)
    return metrics


def per_track_pearson(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    """Pearson r per track (column index) between preds and targets.

    Input arrays are (B, T) where B is batch/window count and T is tracks.
    Each track must have at least 2 finite points and nonzero variance; tracks
    failing these checks return NaN.
    """
    return _per_track_metric(preds, targets, _pearson, min_count=2, require_variance=True)


def per_track_spearman(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    """Spearman rho per track (column index) between preds and targets.

    Input arrays are (B, T). Each track must have at least 2 finite points and
    nonzero variance; tracks failing these checks return NaN.
    """
    return _per_track_metric(preds, targets, _spearman, min_count=2, require_variance=True)


def amplitude_calibration_per_track(
    preds: np.ndarray,
    targets: np.ndarray,
    quantile: float = 0.99,
) -> dict[int, float]:
    """Ratio of predicted to target amplitude at a percentile, per track.

    Amplitude is the absolute value at the given quantile (default 99th). Each
    track must have at least 2 finite points; tracks with zero target amplitude
    return NaN. Input arrays are (B, T).
    """
    def _ratio(pred_vals: np.ndarray, target_vals: np.ndarray) -> float:
        pred_p = np.percentile(np.abs(pred_vals), quantile * 100)
        target_p = np.percentile(np.abs(target_vals), quantile * 100)
        return float(pred_p / target_p) if target_p > 1e-8 else float("nan")

    return _per_track_metric(preds, targets, _ratio, min_count=2, require_variance=False)


def per_track_mse(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    """Mean squared error per track (column index) between preds and targets.

    Input arrays are (B, T). Each track must have at least 1 finite point; tracks
    with no finite points return NaN. No variance requirement.
    """
    def _mse(pred_vals: np.ndarray, target_vals: np.ndarray) -> float:
        return float(mean_squared_error(_as_tensor(pred_vals), _as_tensor(target_vals)))

    return _per_track_metric(preds, targets, _mse, min_count=1, require_variance=False)


def per_track_mae(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    """Mean absolute error per track (column index) between preds and targets.

    Input arrays are (B, T). Each track must have at least 1 finite point; tracks
    with no finite points return NaN. No variance requirement.
    """
    def _mae(pred_vals: np.ndarray, target_vals: np.ndarray) -> float:
        return float(mean_absolute_error(_as_tensor(pred_vals), _as_tensor(target_vals)))

    return _per_track_metric(preds, targets, _mae, min_count=1, require_variance=False)
