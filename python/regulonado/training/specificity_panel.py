"""Score a model's group-contrast channels against a fixed panel of observed-specificity regions.

The panel (``scripts/build_specificity_panel.py``) lists genomic regions with an observed
target-group specificity score computed from the atlas bigwigs, and the training-dataset
window each region was assigned to. At evaluation time :class:`SpecificityPanelEvaluator`
predicts on those windows (sequence read from the genome FASTA, context centred as the dataset
builds it), reduces the predicted group channels to one score per region with
:func:`region_scores`, and compares against the observed scores. It needs no labels, so every
run is scored on the same regions whichever folds it trains on.

The same :func:`region_scores` reduction is applied to *observed* group-contrast labels when
the panel is built, so the panel's sidecar JSON records the ceiling these metrics can reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from regulonado.genomics import Window, one_hot_context

__all__ = [
    "PanelWindow",
    "SpecificityPanelEvaluator",
    "panel_metrics",
    "panel_windows",
    "region_scores",
]


def region_scores(
    channels: torch.Tensor, region_mask: torch.Tensor, *, target_index: int, quantile: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-bin group channels ``[R, G, B]`` to a per-region score and target rank.

    Each group's region statistic is its maximum over the region's bins (``region_mask``,
    ``[R, B]``), mirroring the validated statistic's per-track max over the region. The score
    is the target group's statistic minus the ``quantile`` of the other groups' statistics
    (linear interpolation), and the rank is 1 + the number of groups strictly above the
    target. Gauge-centred channels are fine: the per-bin gauge shifts every group equally at
    a bin, so it only enters through which bin each group peaks at.
    """
    masked = channels.float().masked_fill(~region_mask[:, None, :], float("-inf"))
    per_group = masked.amax(dim=-1)  # [R, G]
    target = per_group[:, target_index]
    others = torch.cat([per_group[:, :target_index], per_group[:, target_index + 1 :]], dim=1)
    reference = torch.quantile(others, quantile, dim=1, interpolation="linear")
    rank = (per_group > target[:, None]).sum(dim=1) + 1
    return target - reference, rank


@dataclass(frozen=True)
class PanelWindow:
    """One prediction window and the panel regions it covers, as bin spans inside it."""

    window: Window
    rows: tuple[int, ...]
    bin_spans: tuple[tuple[int, int], ...]


def panel_windows(
    panel: pd.DataFrame, *, n_pred_bins: int, bin_size: int, context_length: int
) -> list[PanelWindow]:
    """The panel's own prediction windows, each with the regions assigned to it.

    Windows come from the panel's ``window_chrom``/``window_start``/``window_end`` columns —
    the training dataset's windows, so predictions stay in distribution — and the model
    context is centred on each, as the dataset builds it.
    """
    pred_bp = n_pred_bins * bin_size
    widths = (panel["window_end"] - panel["window_start"]).unique()
    if len(widths) != 1 or int(widths[0]) != pred_bp:
        raise ValueError(
            f"panel windows are {sorted(int(w) for w in widths)} bp; the model predicts "
            f"{pred_bp} bp (n_pred_bins * bin_size). Rebuild the panel from this dataset's "
            "intervals."
        )
    windows: list[PanelWindow] = []
    for (chrom, start), group in panel.groupby(["window_chrom", "window_start"], sort=True):
        start = int(start)
        ctx_start = start + pred_bp // 2 - context_length // 2
        window = Window(str(chrom), start, start + pred_bp, ctx_start, ctx_start + context_length)
        spans = tuple(
            ((int(s) - start) // bin_size, -(-(int(e) - start) // bin_size))
            for s, e in zip(group["region_start"], group["region_end"])
        )
        windows.append(PanelWindow(window, tuple(int(i) for i in group.index), spans))
    return windows


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC of ``scores`` separating ``positive`` from the rest."""
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def panel_metrics(panel: pd.DataFrame, score: np.ndarray, rank: np.ndarray) -> dict[str, float]:
    """Predicted-vs-observed agreement over the panel.

    * ``panel_curated_auc`` — curated regions vs the other candidate regions in the same
      windows.
    * ``panel_positive_auc`` — ``observed_positive`` vs everything else in the panel.
    * ``panel_spearman`` — rank agreement with ``observed_score`` over the whole panel.
    * ``panel_curated_rank1_fraction`` — curated regions where the target is the top group.
    """
    tier = panel["tier"].to_numpy()
    curated_or_candidate = np.isin(tier, ["curated", "candidate"])
    return {
        "panel_curated_auc": _auc(
            score[curated_or_candidate], tier[curated_or_candidate] == "curated"
        ),
        "panel_positive_auc": _auc(score, panel["observed_positive"].to_numpy(dtype=bool)),
        "panel_spearman": float(spearmanr(score, panel["observed_score"].to_numpy())[0]),
        "panel_curated_rank1_fraction": float((rank[tier == "curated"] == 1).mean()),
    }


class SpecificityPanelEvaluator:
    """Predict on the panel's windows and report :func:`panel_metrics`."""

    def __init__(
        self,
        panel_path: str | Path,
        fasta_path: str | Path,
        *,
        target_index: int,
        quantile: float,
        n_pred_bins: int,
        bin_size: int,
        context_length: int,
        batch_size: int = 4,
    ) -> None:
        import pyfaidx

        self.panel = pd.read_parquet(panel_path).reset_index(drop=True)
        self.fasta = pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=False)
        self.chrom_sizes = {name: len(self.fasta[name]) for name in self.fasta.keys()}
        self.target_index = target_index
        self.quantile = quantile
        self.context_length = context_length
        self.batch_size = batch_size
        self.windows = panel_windows(
            self.panel, n_pred_bins=n_pred_bins, bin_size=bin_size, context_length=context_length
        )
        lengths = [end - start for w in self.windows for start, end in w.bin_spans]
        self._max_bins = max(lengths)

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        from regulonado.inference import model_track_metadata

        was_training = model.training
        model.eval()
        param = next(model.parameters())
        track_metadata = model_track_metadata(model, str(param.device))
        track_channels = int(model.track_channel_count)
        channels: torch.Tensor | None = None
        mask = torch.zeros(len(self.panel), self._max_bins, dtype=torch.bool)
        try:
            for i in range(0, len(self.windows), self.batch_size):
                batch = self.windows[i : i + self.batch_size]
                seqs = np.stack(
                    [
                        one_hot_context(
                            self.fasta,
                            w.window,
                            self.context_length,
                            self.chrom_sizes[w.window.chrom],
                        )
                        for w in batch
                    ]
                )
                x = torch.from_numpy(seqs).to(device=param.device, dtype=param.dtype)
                group = model(x, **track_metadata)[:, track_channels:, :].float().cpu()
                if channels is None:
                    channels = torch.zeros(len(self.panel), group.shape[1], self._max_bins)
                for b, w in enumerate(batch):
                    for row, (start, end) in zip(w.rows, w.bin_spans):
                        channels[row, :, : end - start] = group[b, :, start:end]
                        mask[row, : end - start] = True
        finally:
            model.train(was_training)
        score, rank = region_scores(
            channels, mask, target_index=self.target_index, quantile=self.quantile
        )
        return panel_metrics(self.panel, score.numpy(), rank.numpy())
