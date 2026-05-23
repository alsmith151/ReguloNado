from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def _paired_group_masks(
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


def _corr_stat(result: Any) -> float:
    if hasattr(result, "statistic"):
        return float(result.statistic)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


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
    pair_masks = _paired_group_masks(condition_ids, shared_track_index)
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
    pred_lfc = np.asarray(pred_lfc, dtype=np.float64)
    meas_lfc = np.asarray(meas_lfc, dtype=np.float64)
    finite_mask = np.isfinite(pred_lfc) & np.isfinite(meas_lfc)
    pred_lfc = pred_lfc[finite_mask]
    meas_lfc = meas_lfc[finite_mask]

    if len(pred_lfc) < 2 or np.std(pred_lfc) < 1e-8 or np.std(meas_lfc) < 1e-8:
        return float("nan")
    return float(_corr_stat(pearsonr(pred_lfc, meas_lfc)))


def delta_log2fc_metrics(pred_lfc: np.ndarray, meas_lfc: np.ndarray) -> dict[str, float]:
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

    pearson_r = _corr_stat(pearsonr(pred_lfc, meas_lfc))
    spearman_r = _corr_stat(spearmanr(pred_lfc, meas_lfc))
    var_thresh = np.percentile(np.abs(meas_lfc), 80)
    top_mask = np.abs(meas_lfc) >= var_thresh
    if top_mask.sum() >= 2 and np.std(pred_lfc[top_mask]) > 1e-8:
        top_r = _corr_stat(pearsonr(pred_lfc[top_mask], meas_lfc[top_mask]))
    else:
        top_r = float("nan")
    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "top_variance_pearson": float(top_r),
    }


def init_validation_metric_state() -> dict[str, Any]:
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
    delta_pearson = delta_metrics.get("delta_lfc/pearson", float("nan"))
    if np.isfinite(delta_pearson) and np.isfinite(raw_pearson):
        metrics["selection/balanced_score"] = float(0.5 * (delta_pearson + raw_pearson))
    elif np.isfinite(raw_pearson):
        metrics["selection/balanced_score"] = raw_pearson
    return metrics


def per_track_pearson(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")

    metrics: dict[int, float] = {}
    for track_idx in range(preds.shape[-1]):
        pred_vals = preds[:, track_idx]
        target_vals = targets[:, track_idx]
        finite_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
        if finite_mask.sum() < 2:
            metrics[track_idx] = float("nan")
            continue
        pred_vals = pred_vals[finite_mask]
        target_vals = target_vals[finite_mask]
        if np.std(pred_vals) < 1e-8 or np.std(target_vals) < 1e-8:
            metrics[track_idx] = float("nan")
        else:
            metrics[track_idx] = float(_corr_stat(pearsonr(pred_vals, target_vals)))
    return metrics


def per_track_spearman(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")

    metrics: dict[int, float] = {}
    for track_idx in range(preds.shape[-1]):
        pred_vals = preds[:, track_idx]
        target_vals = targets[:, track_idx]
        finite_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
        if finite_mask.sum() < 2:
            metrics[track_idx] = float("nan")
            continue
        pred_vals = pred_vals[finite_mask]
        target_vals = target_vals[finite_mask]
        if np.std(pred_vals) < 1e-8 or np.std(target_vals) < 1e-8:
            metrics[track_idx] = float("nan")
        else:
            metrics[track_idx] = float(_corr_stat(spearmanr(pred_vals, target_vals)))
    return metrics


def amplitude_calibration_per_track(
    preds: np.ndarray,
    targets: np.ndarray,
    quantile: float = 0.99,
) -> dict[int, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")

    metrics: dict[int, float] = {}
    for track_idx in range(preds.shape[-1]):
        pred_vals = preds[:, track_idx]
        target_vals = targets[:, track_idx]
        finite_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
        if finite_mask.sum() < 2:
            metrics[track_idx] = float("nan")
            continue
        pred_vals = pred_vals[finite_mask]
        target_vals = target_vals[finite_mask]
        pred_p = np.percentile(np.abs(pred_vals), quantile * 100)
        target_p = np.percentile(np.abs(target_vals), quantile * 100)
        metrics[track_idx] = float(pred_p / target_p) if target_p > 1e-8 else float("nan")
    return metrics


def per_track_mse(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")

    metrics: dict[int, float] = {}
    for track_idx in range(preds.shape[-1]):
        pred_vals = preds[:, track_idx]
        target_vals = targets[:, track_idx]
        finite_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
        if finite_mask.sum() == 0:
            metrics[track_idx] = float("nan")
        else:
            pred_vals = pred_vals[finite_mask]
            target_vals = target_vals[finite_mask]
            metrics[track_idx] = float(np.mean(np.square(pred_vals - target_vals)))
    return metrics


def per_track_mae(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")

    metrics: dict[int, float] = {}
    for track_idx in range(preds.shape[-1]):
        pred_vals = preds[:, track_idx]
        target_vals = targets[:, track_idx]
        finite_mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
        if finite_mask.sum() == 0:
            metrics[track_idx] = float("nan")
        else:
            pred_vals = pred_vals[finite_mask]
            target_vals = target_vals[finite_mask]
            metrics[track_idx] = float(np.mean(np.abs(pred_vals - target_vals)))
    return metrics
