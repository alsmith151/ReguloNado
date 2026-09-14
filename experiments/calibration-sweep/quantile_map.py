"""Held-out monotone quantile mapping for post-hoc amplitude calibration."""
from __future__ import annotations

import numpy as np


class PerTrackQuantileMap:
    """Map prediction amplitudes to held-out target quantiles, independently per track."""

    def __init__(self, pred_quantiles: np.ndarray, target_quantiles: np.ndarray) -> None:
        pred = np.asarray(pred_quantiles, dtype=float)
        target = np.asarray(target_quantiles, dtype=float)
        if pred.shape != target.shape or pred.ndim != 2:
            raise ValueError("pred_quantiles and target_quantiles must have shape [n_quantiles, n_tracks]")
        if np.any(np.diff(pred, axis=0) < 0) or np.any(np.diff(target, axis=0) < 0):
            raise ValueError("quantile knots must be monotone")
        self.pred = pred
        self.target = target

    @classmethod
    def fit(cls, predictions: np.ndarray, targets: np.ndarray, quantiles: np.ndarray | None = None):
        predictions = np.asarray(predictions, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if predictions.shape != targets.shape or predictions.ndim != 2:
            raise ValueError("predictions and targets must have shape [n_examples, n_tracks]")
        q = np.linspace(0, 1, 101) if quantiles is None else np.asarray(quantiles)
        return cls(np.quantile(predictions, q, axis=0), np.quantile(targets, q, axis=0))

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions, dtype=float)
        if values.shape[-1] != self.pred.shape[-1]:
            raise ValueError("prediction track count does not match fitted quantile map")
        return np.stack([
            np.interp(values[..., track], self.pred[:, track], self.target[:, track])
            for track in range(values.shape[-1])
        ], axis=-1)
