"""Per-group correlation metrics for the region count model.

Ported from ``unique_enhancer_finding.modelling.metrics`` (UEF): ``GroupedCountMetrics``
pools per-track counts into per-group rates and scores them with
``PerTaskCorrelationMetrics``, keeping the metric names (``contrast_pearson_mean``,
``contrast_pearson_<group>``, ``contrast_pearson_top_decile_<target>``) identical so
runs of :class:`~regulonado.regions.model.RegionCountModel` compare directly with UEF.

Usable as ``transformers.Trainer(compute_metrics=...)``: ``GroupedCountMetrics.__call__``
takes the ``(predictions, labels)`` pair ``Trainer`` hands ``compute_metrics`` (an
``EvalPrediction`` or plain 2-tuple both unpack the same way) -- ``predictions`` are the
model's per-group rates (``RegionCountOutput.logits``, ``[N, n_groups]``) and ``labels``
are the raw per-track counts (``[N, n_tracks]``, ``NaN`` where masked).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

from regulonado.counts.dataset import group_count_rates

__all__ = ["GroupedCountMetrics", "PerTaskCorrelationMetrics", "group_count_rates"]


class PerTaskCorrelationMetrics:
    """Per-task Pearson/Spearman correlation, for ``Trainer(compute_metrics=...)``.

    Thin wrapper around ``torchmetrics.regression.{Pearson,Spearman}CorrCoef`` with
    ``num_outputs=n_tasks`` -- the statistics themselves are not hand-rolled, only the
    plumbing to and from ``Trainer``'s ``(predictions, labels)`` numpy pair and a flat
    metrics dict.

    Besides plain per-task correlation it reports ``contrast_pearson_<task>``: Pearson
    over regions of each task's ``log1p`` signal *minus that region's mean across
    tasks*. Plain correlation is dominated by regions open in every cell type; the
    contrast asks whether the model predicts where a task differs from the rest --
    cell type specificity.

    Parameters
    ----------
    task_names
        Names for the ``n_tasks`` output columns, used as dict-key suffixes (e.g.
        ``"pearson_HL-60"``). Length fixes ``n_tasks``.
    """

    def __init__(
        self,
        task_names: list[str],
        top_decile_task: str | None = None,
        loss_multiplier: float | None = None,
        contrast_multiplier: float | None = None,
    ) -> None:
        self.task_names = list(task_names)
        self.top_decile_task = top_decile_task
        self.loss_multiplier = loss_multiplier
        self.contrast_multiplier = (
            loss_multiplier if contrast_multiplier is None else contrast_multiplier
        )
        n_tasks = len(self.task_names)
        self._pearson = PearsonCorrCoef(num_outputs=n_tasks)
        self._contrast_pearson = PearsonCorrCoef(num_outputs=n_tasks)
        with warnings.catch_warnings():
            # "will save all targets and predictions in the buffer": expected, and
            # bounded by the reset in __call__ to one eval set at a time.
            warnings.filterwarnings(
                "ignore", message="Metric `SpearmanCorrcoef` will save", category=UserWarning
            )
            self._spearman = SpearmanCorrCoef(num_outputs=n_tasks)
        # float64 throughout: torchmetrics returns NaN for a correlation whose
        # variance / max-deviation**2 is below sqrt(eps) of the state dtype -- ~3.5e-4
        # in float32, which a single outlier prediction ~60 std out trips. float64
        # lowers the bound to ~1.5e-8.
        for metric in (self._pearson, self._contrast_pearson, self._spearman):
            metric.set_dtype(torch.float64)

    def __call__(self, eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        preds = torch.as_tensor(np.asarray(predictions), dtype=torch.float64)
        target = torch.as_tensor(np.asarray(labels), dtype=torch.float64)

        # Reset before each evaluation. A torchmetrics `Metric.__call__` returns the
        # value for THIS batch but also accumulates into the metric's global state,
        # and `Trainer` reuses one `compute_metrics` object for every eval in the run.
        # `SpearmanCorrCoef` keeps every prediction and target it has ever seen (it
        # needs them to rank), so without this the state grows by the whole eval set
        # on every epoch.
        self._pearson.reset()
        self._spearman.reset()
        self._contrast_pearson.reset()

        pearson = self._pearson(preds, target)
        spearman = self._spearman(preds, target)
        contrast = self._contrast_pearson(_contrast(preds), _contrast(target))

        metrics: dict[str, float] = {}
        for index, name in enumerate(self.task_names):
            metrics[f"pearson_{name}"] = float(pearson[index])
            metrics[f"spearman_{name}"] = float(spearman[index])
            metrics[f"contrast_pearson_{name}"] = float(contrast[index])
        metrics["pearson_mean"] = float(pearson.mean())
        metrics["spearman_mean"] = float(spearman.mean())
        metrics["contrast_pearson_mean"] = float(contrast.mean())

        if self.loss_multiplier is not None:
            metrics.update(self._loss_terms(preds, target))

        if self.top_decile_task is not None:
            index = self.task_names.index(self.top_decile_task)
            true_contrast = _contrast(target)[:, index]
            # 87% of the contrast sum of squares sits in the top decile of
            # |contrast|, so the full-set Pearson is decided there while being
            # diluted -- and destabilised -- by a flat majority whose median
            # |contrast| is within the replicate noise. Selecting on this decile is
            # selecting on the regions specificity lives in.
            cutoff = true_contrast.abs().quantile(0.9)
            keep = true_contrast.abs() >= cutoff
            name = f"contrast_pearson_top_decile_{self.top_decile_task}"
            if int(keep.sum()) < 2:
                metrics[name] = float("nan")
            else:
                decile = PearsonCorrCoef(num_outputs=1)
                decile.set_dtype(torch.float64)
                metrics[name] = float(decile(_contrast(preds)[:, index][keep], true_contrast[keep]))
        return metrics

    def _loss_terms(self, preds: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """The loss's three terms, reported separately, plus a contrast scale check.

        ``contrast_slope`` is the least-squares slope of true contrast on predicted
        contrast for the target task. ``contrast_pearson`` is scale-invariant, so a
        model can improve the *ranking* of contrast while its absolute scale drifts.
        Slope 1.0 means calibrated, below 1.0 means the spread of predictions is too
        wide for the truth, above 1.0 too narrow.
        """
        multiplier = float(self.loss_multiplier or 1.0)
        contrast_multiplier = float(self.contrast_multiplier or 1.0)

        log_pred = torch.log1p(multiplier * preds.clamp(min=0)) - torch.log1p(
            (multiplier * preds.clamp(max=0)).abs()
        )
        log_true = torch.log1p(multiplier * target.clamp(min=0))
        mse_term = float(((log_pred - log_true) ** 2).mean())

        normalised_pred = preds / preds.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        normalised_true = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        cosine_term = float(-(normalised_pred * normalised_true).sum(dim=-1).mean())

        if contrast_multiplier == multiplier:
            contrast_pred, contrast_true = log_pred, log_true
        else:
            contrast_pred = torch.log1p(contrast_multiplier * preds.clamp(min=0)) - torch.log1p(
                (contrast_multiplier * preds.clamp(max=0)).abs()
            )
            contrast_true = torch.log1p(contrast_multiplier * target.clamp(min=0))
        centred_pred = contrast_pred - contrast_pred.mean(dim=-1, keepdim=True)
        centred_true = contrast_true - contrast_true.mean(dim=-1, keepdim=True)
        contrast_term = float(((centred_pred - centred_true) ** 2).mean())

        terms = {
            "loss_mse_term": mse_term,
            "loss_cosine_term": cosine_term,
            "loss_contrast_term": contrast_term,
        }
        if self.top_decile_task is not None:
            index = self.task_names.index(self.top_decile_task)
            x = centred_pred[:, index]
            y = centred_true[:, index]
            variance = ((x - x.mean()) ** 2).sum()
            if float(variance) > 0:
                slope = float(((x - x.mean()) * (y - y.mean())).sum() / variance)
                terms[f"contrast_slope_{self.top_decile_task}"] = slope
        return terms


class GroupedCountMetrics:
    """:class:`PerTaskCorrelationMetrics` for the count model, at group level.

    The count model predicts one rate per group (cell type) while its labels are
    per-track counts. Labels are pooled to the same per-group normalised rates the
    heads predict (:func:`group_count_rates`) and the signal model's metrics are
    computed on those, so ``contrast_pearson_HL-60`` and its top-decile variant keep
    their names and meaning, and runs of the two models can be compared on one axis.

    Parameters
    ----------
    group_names
        One name per output group (``n_groups`` fixes the length).
    track_groups
        Per-track group index, ``[n_tracks]``, same order as the label columns.
    log_size_factors
        Per-track fixed log size factor, ``[n_tracks]``, same order and units as
        :class:`~regulonado.regions.model.CountHead`.
    top_decile_task
        Optional group name to also report ``contrast_pearson_top_decile_<name>`` for.
    """

    def __init__(
        self,
        group_names: list[str],
        track_groups: Any,
        log_size_factors: Any,
        top_decile_task: str | None = None,
    ) -> None:
        self.track_groups = np.asarray(track_groups, dtype=np.int64)
        self.log_size_factors = np.asarray(log_size_factors, dtype=np.float64)
        self.inner = PerTaskCorrelationMetrics(group_names, top_decile_task=top_decile_task)

    def __call__(self, eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        rates = group_count_rates(
            np.asarray(labels),
            self.track_groups,
            self.log_size_factors,
            len(self.inner.task_names),
        )
        # Regions where some group has every replicate masked have no observed rate
        # there; score the rest rather than letting NaN void the metric.
        complete = np.isfinite(rates).all(axis=1)
        return self.inner((np.asarray(predictions)[complete], rates[complete]))


def _contrast(values: torch.Tensor) -> torch.Tensor:
    """``log1p`` signal relative to each region's mean across tasks, ``[n_regions, n_tasks]``."""
    logged = torch.log1p(values.clamp(min=0))
    return logged - logged.mean(dim=1, keepdim=True)
