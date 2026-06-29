from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


def _inverse_signal_transform(
    x: np.ndarray,
    scale_factors: np.ndarray | None,
    apply_squash: bool,
    apply_scale: bool,
) -> np.ndarray:
    """Reverse the squash and/or scale applied by make_transform.

    Squash inverse: ``(x + 1)^(4/3) - 1``  (inverse of ``(x+1)^0.75 - 1``).
    Scale inverse:  divide by per-track scale factor to go from raw counts → normalised coverage.

    After unsquash only (apply_scale=False) the result is in raw read-count units.
    After both steps the result is in the original normalised BigWig units (RPKM).
    """
    y = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    if apply_squash:
        y = np.power(y + 1.0, 4.0 / 3.0) - 1.0
        np.maximum(y, 0.0, out=y)
    if apply_scale and scale_factors is not None:
        sf = np.asarray(scale_factors, dtype=np.float32).reshape(-1, 1)
        y = y / np.maximum(sf, 1e-8)
    return y


def _plot_examples(
    preds: np.ndarray,
    targets: np.ndarray,
    intervals: list[str],
    output_dir: Path,
    step: int,
    tracks_per_example: int = 3,
    track_names: list[str] | None = None,
) -> None:
    """Plot prediction vs target for a batch of examples.

    Args:
        preds: shape [B, n_tracks, n_bins]
        targets: shape [B, n_tracks, n_bins]
        intervals: interval strings for each example, e.g. "chr1:1000-2000"
    """
    if preds.shape[0] == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_dir = output_dir / "examples" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    n_tracks = preds.shape[1]
    n_pick = min(tracks_per_example, n_tracks)
    track_indices = random.sample(range(n_tracks), n_pick)
    for example_idx in range(preds.shape[0]):
        interval = (
            intervals[example_idx] if example_idx < len(intervals) else f"example_{example_idx}"
        )
        for track_idx in track_indices:
            pred = preds[example_idx, track_idx]
            target = targets[example_idx, track_idx]
            # Predictions may contain NaN/Inf (e.g. early in training); ignore
            # non-finite values when computing axis limits so set_ylim doesn't raise.
            finite = np.concatenate([pred, target])
            finite = finite[np.isfinite(finite)]
            if finite.size:
                ymax = max(float(finite.max()), 0.0)
                ymin = min(float(finite.min()), 0.0)
            else:
                ymax, ymin = 0.0, 0.0
            pad = (ymax - ymin) * 0.05 or 0.1
            track_name = (
                track_names[track_idx]
                if track_names and track_idx < len(track_names)
                else f"track {track_idx}"
            )
            fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
            ax.plot(target, label="real", linewidth=1.2)
            ax.plot(pred, label="predicted", linewidth=1.0, alpha=0.85)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_title(f"{track_name}  |  {interval}")
            ax.set_xlabel("bin")
            ax.set_ylabel("signal")
            ax.legend(loc="upper right", frameon=False)
            fig.tight_layout()
            fig.savefig(step_dir / f"example_{example_idx:02d}_track_{track_idx:04d}.png")
            plt.close(fig)


class _WandbConfigCallback(TrainerCallback):
    """Push the full resolved Hydra config to wandb.config on the first log event.

    HF Trainer only syncs TrainingArguments; backbone/head/loss/data settings
    are invisible in the W&B UI without this.

    We use on_log (not on_train_begin) because HF's WandbCallback calls wandb.init()
    inside its own on_train_begin handler, which runs after ours.  By the time the
    first on_log fires, wandb.run is guaranteed to exist.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._cfg = cfg
        self._uploaded = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._uploaded or not state.is_world_process_zero:
            return
        self._uploaded = True
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update({"regulonado": self._cfg}, allow_val_change=True)
        except Exception:
            pass


class _LRLogCallback(TrainerCallback):
    """Log per-param-group learning rates so head and backbone LRs are both visible."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        optimizer = kwargs.get("optimizer")
        if optimizer is None or not state.is_world_process_zero:
            return
        logs: dict[str, float] = {}
        names = ["backbone", "head"]
        for i, group in enumerate(optimizer.param_groups):
            label = names[i] if i < len(names) else f"group{i}"
            logs[f"learning_rate/{label}"] = group["lr"]
        if state.log_history:
            state.log_history[-1].update(logs)


class _EvalPlotCallback(TrainerCallback):
    """After each validation run, plot a handful of pred-vs-target examples.

    Runs the model directly on raw dataset items so predictions are the full
    [n_tracks, n_bins] signal — not reduced by preprocess_logits_for_metrics.
    The 'interval' field present on each dataset item is used in the plot title.
    """

    def __init__(
        self,
        *,
        dataset: Any,
        collate_fn: Callable,
        num_examples: int,
        output_dir: Path,
        track_names: list[str] | None,
        scale_factors: np.ndarray | None = None,
        apply_squash: bool = True,
        apply_scale: bool = True,
    ) -> None:
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._num_examples = num_examples
        self._output_dir = output_dir
        self._track_names = track_names
        self._scale_factors = scale_factors
        self._apply_squash = apply_squash
        self._apply_scale = apply_scale

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero or self._num_examples <= 0:
            return

        raw_items: list[dict] = []
        for item in self._dataset:
            raw_items.append(item)
            if len(raw_items) >= self._num_examples:
                break
        if not raw_items:
            return

        batch = self._collate_fn(raw_items)
        device = next(model.parameters()).device
        labels_tensor = batch["labels"].to(device)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k != "labels"
        }

        model.eval()
        with torch.no_grad():
            out = model(**inputs)
        preds_raw = (out["logits"] if isinstance(out, dict) else out).float().cpu().numpy()
        labels_raw = labels_tensor.float().cpu().numpy()

        # Both pred and target are in squash-transformed space; reverse to signal space
        # so the y-axis shows interpretable per-track signal magnitudes.
        def _inv(x: np.ndarray) -> np.ndarray:
            return _inverse_signal_transform(
                x, self._scale_factors, self._apply_squash, self._apply_scale
            )
        preds_plot  = np.stack([_inv(preds_raw[i])  for i in range(preds_raw.shape[0])])
        labels_plot = np.stack([_inv(labels_raw[i]) for i in range(labels_raw.shape[0])])

        intervals = [item.get("interval", f"example_{i}") for i, item in enumerate(raw_items)]
        _plot_examples(
            preds_plot,
            labels_plot,
            intervals,
            self._output_dir,
            int(state.global_step),
            track_names=self._track_names,
        )
