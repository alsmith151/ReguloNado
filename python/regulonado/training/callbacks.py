from __future__ import annotations

import csv
import json
import logging
import random
from collections.abc import Callable, Mapping, Sequence
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

logger = logging.getLogger(__name__)


def _inverse_signal_transform(
    x: np.ndarray,
    scale_factors: np.ndarray | None,
    apply_squash: bool,
    apply_scale: bool,
    background: np.ndarray | None = None,
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
        if background is not None:
            y = y + np.asarray(background, dtype=np.float32).reshape(-1, 1)
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
    # Degrade to a warning rather than raising: this runs inside the training loop, and
    # aborting a long run because an optional plotting dependency is absent is worse than
    # losing the diagnostic plots. matplotlib is declared in the `train` extra, so this
    # only trips for a hand-assembled environment.
    try:
        import matplotlib
    except ModuleNotFoundError:
        logger.warning(
            "matplotlib is not installed — skipping example plots. "
            "Install it with `pip install regulonado[train]` or `pip install matplotlib`."
        )
        return

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


class WandbConfigCallback(TrainerCallback):
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
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning(f"Failed to update wandb config: {e}")


class LRLogCallback(TrainerCallback):
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
        # Groups are named by parameter family in _build_optimizer; decay and no-decay
        # groups of one family share a learning rate, and empty families are omitted, so
        # position says nothing about which family a group belongs to.
        logs: dict[str, float] = {}
        for i, group in enumerate(optimizer.param_groups):
            logs[f"learning_rate/{group.get('name', f'group{i}')}"] = group["lr"]
        if state.log_history:
            state.log_history[-1].update(logs)


class EvalExampleDiagnostics(TrainerCallback):
    """Plot and record per-track validation magnitudes after each evaluation.

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
        background: np.ndarray | None = None,
        apply_squash: bool = True,
        apply_scale: bool = True,
        label_divisor: np.ndarray | None = None,
    ) -> None:
        """``label_divisor`` (count label space exposure) replaces the inverse transform:
        predictions are already in output units and labels are divided into them."""
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._num_examples = num_examples
        self._output_dir = output_dir
        self._track_names = track_names
        self._scale_factors = scale_factors
        self._background = background
        self._apply_squash = apply_squash
        self._apply_scale = apply_scale
        self._label_divisor = label_divisor

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
        # A group-contrast composite head concatenates [B, T, L] per-track logits with
        # [B, G, L] group-contrast logits; labels here are always per-track only. Slice
        # back to the leading T channels before comparing, the same split
        # runner._split_track_and_group_logits applies for compute_loss/prediction_step.
        track_channel_count = getattr(
            model, "track_channel_count", getattr(getattr(model, "module", None), "track_channel_count", None)
        )
        if track_channel_count is not None and preds_raw.shape[-2] > track_channel_count:
            preds_raw = preds_raw[..., :track_channel_count, :]
        if labels_raw.shape[-2:] != preds_raw.shape[-2:]:
            labels_raw = labels_raw.transpose(0, 2, 1)
        if labels_raw.shape != preds_raw.shape:
            raise ValueError(
                f"Diagnostic prediction/label shapes disagree after axis alignment: "
                f"pred={preds_raw.shape}, labels={labels_raw.shape}"
            )

        # Both pred and target are in squash-transformed space; reverse to signal space
        # so the y-axis shows interpretable per-track signal magnitudes.
        def _inv(x: np.ndarray) -> np.ndarray:
            return _inverse_signal_transform(
                x, self._scale_factors, self._apply_squash, self._apply_scale, self._background
            )

        if self._label_divisor is not None:
            divisor = np.asarray(self._label_divisor, dtype=np.float32).reshape(1, -1, 1)
            preds_plot = np.maximum(preds_raw, 0.0)
            labels_plot = labels_raw / divisor
        else:
            preds_plot = np.stack([_inv(preds_raw[i]) for i in range(preds_raw.shape[0])])
            labels_plot = np.stack([_inv(labels_raw[i]) for i in range(labels_raw.shape[0])])

        def _summary(values: np.ndarray) -> dict[str, list[float]]:
            return {
                "mean": np.nanmean(values, axis=-1).tolist(),
                "q99": np.nanquantile(values, 0.99, axis=-1).tolist(),
                "max": np.nanmax(values, axis=-1).tolist(),
            }

        self._output_dir.mkdir(parents=True, exist_ok=True)
        intervals = [item.get("interval", f"example_{i}") for i, item in enumerate(raw_items)]
        diagnostic_path = self._output_dir / f"eval_diagnostics_step_{int(state.global_step)}.json"
        diagnostic_path.write_text(
            json.dumps(
                {
                    "intervals": intervals,
                    "track_names": self._track_names,
                    "predictions": _summary(preds_plot),
                    "labels": _summary(labels_plot),
                },
                indent=2,
            )
            + "\n"
        )
        _plot_examples(
            preds_plot,
            labels_plot,
            intervals,
            self._output_dir,
            int(state.global_step),
            track_names=self._track_names,
        )


class PerTrackMetricsReport(TrainerCallback):
    """Write the per-track values behind each evaluation's median metrics.

    ``record`` is the ``per_track_sink`` of ``make_compute_metrics``. Validation evals
    write ``per_track_metrics/validation_step_<step>.csv``; the final test-split
    ``predict`` writes ``per_track_metrics/test.csv`` and, when W&B is active, puts the
    test metrics in the run summary and one ``test/per_track_metrics`` table rather than
    adding step-series panels.
    """

    def __init__(self, *, output_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
        self._dir = output_dir / "per_track_metrics"
        self._track_columns = {
            "track_name": [
                str(r.get("track_name") or Path(str(r.get("bigwig_path") or f"track_{i}")).stem)
                for i, r in enumerate(records)
            ],
            "group": [str(r.get("group") or "") for r in records],
            "assay_class": [str(r.get("assay_class") or "") for r in records],
        }
        self._latest: dict[str, np.ndarray] | None = None

    def record(self, per_track: Mapping[str, np.ndarray]) -> None:
        self._latest = dict(per_track)

    def _rows(self) -> tuple[list[str], list[list[Any]]]:
        assert self._latest is not None
        columns = [*self._track_columns, *self._latest]
        values = [*self._track_columns.values(), *(v.tolist() for v in self._latest.values())]
        return columns, [list(row) for row in zip(*values, strict=True)]

    def _write_csv(self, name: str) -> tuple[list[str], list[list[Any]]]:
        columns, rows = self._rows()
        self._dir.mkdir(parents=True, exist_ok=True)
        with (self._dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return columns, rows

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._latest is not None and state.is_world_process_zero:
            self._write_csv(f"validation_step_{int(state.global_step)}")
        self._latest = None

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._latest is None or not state.is_world_process_zero:
            self._latest = None
            return
        columns, rows = self._write_csv("test")
        self._latest = None
        try:
            import wandb
        except ModuleNotFoundError:
            return
        if wandb.run is None:
            return
        # "test_<metric>" -> "test/<metric>", matching HF's eval/<metric> naming.
        wandb.run.summary.update(
            {key.replace("test_", "test/", 1): value for key, value in (metrics or {}).items()}
        )
        wandb.log({"test/per_track_metrics": wandb.Table(columns=columns, data=rows)})


class PerGroupMetricsReport(TrainerCallback):
    """Write the per-group values behind the group-contrast head's evaluation metrics.

    Sibling of :class:`PerTrackMetricsReport`, kept separate rather than generalising that
    class: its column layout is keyed off ``records`` (one row per *track*), which does not
    map onto the group axis (one row per *group*, a different, smaller cardinality with no
    per-record fields of its own — just a name). ``record`` is the ``per_group_sink`` of
    ``make_compute_metrics``. Validation evals write
    ``per_group_metrics/validation_step_<step>.csv``; the final test-split ``predict``
    writes ``per_group_metrics/test.csv`` and, when W&B is active, logs one
    ``test/per_group_metrics`` table.
    """

    def __init__(self, *, output_dir: Path, group_names: Sequence[str]) -> None:
        self._dir = output_dir / "per_group_metrics"
        self._group_names = [str(name) for name in group_names]
        self._latest: dict[str, np.ndarray] | None = None

    def record(self, per_group: Mapping[str, np.ndarray]) -> None:
        self._latest = dict(per_group)

    def _rows(self) -> tuple[list[str], list[list[Any]]]:
        assert self._latest is not None
        columns = ["group", *self._latest]
        values = [self._group_names, *(v.tolist() for v in self._latest.values())]
        return columns, [list(row) for row in zip(*values, strict=True)]

    def _write_csv(self, name: str) -> tuple[list[str], list[list[Any]]]:
        columns, rows = self._rows()
        self._dir.mkdir(parents=True, exist_ok=True)
        with (self._dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return columns, rows

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._latest is not None and state.is_world_process_zero:
            self._write_csv(f"validation_step_{int(state.global_step)}")
        self._latest = None

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._latest is None or not state.is_world_process_zero:
            self._latest = None
            return
        columns, rows = self._write_csv("test")
        self._latest = None
        try:
            import wandb
        except ModuleNotFoundError:
            return
        if wandb.run is None:
            return
        wandb.log({"test/per_group_metrics": wandb.Table(columns=columns, data=rows)})
