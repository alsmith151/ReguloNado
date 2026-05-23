from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_signal_comparison(
    pred: np.ndarray,
    actual: np.ndarray,
    title: str = "",
    track_names: list[str] | None = None,
) -> matplotlib.figure.Figure:
    pred = np.asarray(pred, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
        actual = actual[np.newaxis, :]

    n_tracks = min(pred.shape[0], 4)
    fig, axes = plt.subplots(n_tracks, 1, figsize=(12, 3 * n_tracks))
    if n_tracks == 1:
        axes = [axes]

    for idx in range(n_tracks):
        ax = axes[idx]
        ax.plot(pred[idx], label="Predicted", linewidth=1.5, alpha=0.8)
        ax.plot(actual[idx], label="Actual", linewidth=1.5, alpha=0.8)
        ax.set_ylabel("Signal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(track_names[idx] if track_names and idx < len(track_names) else f"Track {idx}")

    axes[-1].set_xlabel("Position")
    fig.suptitle(title or "Signal Comparison")
    plt.tight_layout()
    return fig


def plot_delta_lfc_scatter(
    pred_lfc: np.ndarray,
    meas_lfc: np.ndarray,
    pearson_r: float | None = None,
    title: str = "",
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    pred_lfc = np.asarray(pred_lfc, dtype=np.float32)
    meas_lfc = np.asarray(meas_lfc, dtype=np.float32)

    finite_mask = np.isfinite(pred_lfc) & np.isfinite(meas_lfc)
    pred_lfc = pred_lfc[finite_mask]
    meas_lfc = meas_lfc[finite_mask]
    ax.scatter(meas_lfc, pred_lfc, alpha=0.5, s=20)

    lim = max(np.abs(meas_lfc).max(), np.abs(pred_lfc).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5, label="Identity")
    ax.set_xlabel("Measured log2FC")
    ax.set_ylabel("Predicted log2FC")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)

    if pearson_r is not None and np.isfinite(pearson_r):
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {pearson_r:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.legend()
    fig.suptitle(title or "Delta Log2FC Scatter")
    plt.tight_layout()
    return fig


def plot_attribution_heatmap(scores: np.ndarray, title: str = "") -> matplotlib.figure.Figure:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.bar(range(len(scores)), scores, width=0.8, alpha=0.7)
        ax.set_xlabel("Position")
        ax.set_ylabel("Importance")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(scores, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_ylabel("Channel")
        ax.set_xlabel("Position")
        plt.colorbar(im, ax=ax, label="Importance")

    fig.suptitle(title or "Attribution Heatmap")
    plt.tight_layout()
    return fig


def plot_per_track_metrics(
    metrics: dict[int, float],
    metric_name: str,
    track_names: list[str] | None = None,
) -> matplotlib.figure.Figure:
    track_indices = sorted([key for key in metrics if isinstance(key, int)])
    values = [metrics[idx] for idx in track_indices]
    labels = [
        track_names[idx] if track_names and idx < len(track_names) else f"Track {idx}"
        for idx in track_indices
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(track_indices) * 0.25)))
    ax.barh(labels, values, alpha=0.7)
    ax.set_xlabel(metric_name)
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.suptitle(f"Per-Track {metric_name}")
    plt.tight_layout()
    return fig


def plot_training_curves(
    log_dir: Path | str, metrics: list[str] | None = None
) -> matplotlib.figure.Figure:
    if metrics is None:
        metrics = ["train/loss", "val/loss", "delta_lfc/pearson"]

    log_dir = Path(log_dir)
    all_curves: dict[str, list[float]] = {}
    for run_dir in log_dir.glob("*"):
        if not run_dir.is_dir():
            continue
        wandb_dir = run_dir / "wandb"
        if not wandb_dir.exists():
            continue

        for run_file in wandb_dir.glob("*/files/config.yaml"):
            summary_file = run_file.parent.parent / "files" / "summary.json"
            if not summary_file.exists():
                continue
            try:
                with summary_file.open() as handle:
                    summary = json.load(handle)
            except Exception:
                continue
            for metric in metrics:
                if metric in summary:
                    all_curves.setdefault(metric, []).append(summary[metric])

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric, values in all_curves.items():
        if values:
            ax.plot(values, marker="o", label=metric, linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("Training Curves")
    plt.tight_layout()
    return fig
