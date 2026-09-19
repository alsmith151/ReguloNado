from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from regulonado.training.config import TrainerConfig
from regulonado.training.runner import _build_specificity_panel
from regulonado.training.specificity_panel import (
    SpecificityPanelEvaluator,
    panel_metrics,
    panel_windows,
    region_scores,
)

pytest.importorskip("pyfaidx")


def test_region_scores_uses_per_group_max_inside_the_region_only() -> None:
    channels = torch.zeros(1, 4, 6)
    channels[0, 1, 2] = 3.0  # target peak inside the region
    channels[0, 2, 0] = 9.0  # another group's peak outside the region
    channels[0, 3, 3] = 1.0
    mask = torch.tensor([[False, False, True, True, False, False]])

    score, rank = region_scores(channels, mask, target_index=1, quantile=1.0)

    assert score.item() == pytest.approx(2.0)  # 3.0 minus the max of the others inside (1.0)
    assert rank.item() == 1


def test_region_scores_rank_counts_groups_strictly_above_target() -> None:
    channels = torch.tensor([[[5.0], [1.0], [3.0], [0.0]]])
    score, rank = region_scores(channels, torch.ones(1, 1, dtype=torch.bool), target_index=1)
    assert rank.item() == 3
    assert score.item() == pytest.approx(
        1.0 - torch.quantile(torch.tensor([5.0, 3.0, 0.0]), 0.9).item()
    )


def _panel(rows: list[tuple[str, int, int, str, float]], window_bp: int = 1_024) -> pd.DataFrame:
    """Panel rows, each placed in the ``window_bp`` window grid cell containing it."""
    panel = pd.DataFrame(
        rows, columns=["chrom", "region_start", "region_end", "tier", "observed_score"]
    )
    return panel.assign(
        observed_positive=lambda d: d.observed_score >= 1.0,
        window_chrom=lambda d: d.chrom,
        window_start=lambda d: d.region_start // window_bp * window_bp,
        window_end=lambda d: d.region_start // window_bp * window_bp + window_bp,
    )


def test_panel_windows_groups_by_dataset_window_and_maps_bins_back_to_coordinates() -> None:
    panel = _panel(
        [
            ("chr1", 1_100, 1_200, "curated", 2.0),
            ("chr1", 1_300, 1_360, "candidate", 0.0),
            ("chr1", 9_000, 9_050, "candidate", 0.0),
            ("chr2", 1_100, 1_200, "candidate", 0.0),
        ]
    )
    windows = panel_windows(panel, n_pred_bins=32, bin_size=32, context_length=4_096)

    assert [w.rows for w in windows] == [(0, 1), (2,), (3,)]
    for w in windows:
        assert (
            w.window.ctx_start + 4_096 // 2 == w.window.pred_start + 1_024 // 2
        )  # centred context
        for row, (start, end) in zip(w.rows, w.bin_spans):
            assert 0 <= start < end <= 32
            assert w.window.pred_start + start * 32 <= panel.at[row, "region_start"]
            assert w.window.pred_start + end * 32 >= panel.at[row, "region_end"]


def test_panel_windows_rejects_a_panel_built_for_another_window_size() -> None:
    panel = _panel([("chr1", 1_100, 1_200, "curated", 2.0)], window_bp=2_048)
    with pytest.raises(ValueError, match="Rebuild the panel"):
        panel_windows(panel, n_pred_bins=32, bin_size=32, context_length=4_096)


def test_panel_metrics_perfect_prediction() -> None:
    panel = _panel(
        [
            ("chr1", 0, 10, "curated", 3.0),
            ("chr1", 20, 30, "curated", 2.0),
            ("chr1", 40, 50, "candidate", 0.5),
            ("chr1", 60, 70, "pool", -1.0),
        ]
    )
    score = panel.observed_score.to_numpy()
    metrics = panel_metrics(panel, score, np.array([1, 1, 4, 9]))

    assert metrics["panel_curated_auc"] == 1.0
    assert metrics["panel_positive_auc"] == 1.0
    assert metrics["panel_spearman"] == pytest.approx(1.0)
    assert metrics["panel_curated_rank1_fraction"] == 1.0


class _GContentModel(torch.nn.Module):
    """One track channel, then 3 group channels; group 1 is the per-bin G fraction."""

    track_channel_count = 1

    def __init__(self, context_length: int, n_pred_bins: int, bin_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(track_metadata={})
        self.offset = (context_length - n_pred_bins * bin_size) // 2
        self.n_pred_bins, self.bin_size = n_pred_bins, bin_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = x[:, 2, self.offset : self.offset + self.n_pred_bins * self.bin_size]
        g = pred.reshape(x.shape[0], self.n_pred_bins, self.bin_size).float().mean(-1)
        out = torch.zeros(x.shape[0], 4, self.n_pred_bins)
        out[:, 2] = g
        return out + self.weight


def test_evaluator_scores_sequence_features_at_the_right_coordinates(tmp_path) -> None:
    seq = ["A"] * 20_000
    positives = [(2_100, 2_164), (9_300, 9_364)]
    for start, end in positives:
        seq[start:end] = "G" * (end - start)
    (tmp_path / "g.fa").write_text(">chr1\n" + "".join(seq) + "\n")
    panel = _panel(
        [
            ("chr1", 2_100, 2_164, "curated", 2.0),
            ("chr1", 9_300, 9_364, "curated", 2.0),
            ("chr1", 2_500, 2_564, "candidate", 0.0),
            ("chr1", 14_400, 14_464, "candidate", 0.0),
        ]
    )
    panel.to_parquet(tmp_path / "panel.parquet")
    evaluator = SpecificityPanelEvaluator(
        tmp_path / "panel.parquet",
        tmp_path / "g.fa",
        target_index=1,
        quantile=0.9,
        n_pred_bins=32,
        bin_size=32,
        context_length=2_048,
    )
    model = _GContentModel(context_length=2_048, n_pred_bins=32, bin_size=32).train()

    metrics = evaluator.evaluate(model)

    assert metrics["panel_curated_auc"] == 1.0
    assert metrics["panel_curated_rank1_fraction"] == 1.0
    assert model.training  # restored after evaluation


def test_build_specificity_panel_is_off_without_a_path() -> None:
    assert _build_specificity_panel(TrainerConfig(), {}, group_contrast_group_names=["A"]) is None


@pytest.mark.parametrize(
    ("overrides", "groups", "match"),
    [
        ({"specificity_panel_fasta": None}, ["HL-60"], "specificity_panel_fasta"),
        ({}, [], "group_contrast_enabled"),
        ({"group_contrast_target": "K-562"}, ["HL-60"], "group_contrast_target"),
    ],
)
def test_build_specificity_panel_rejects_incomplete_config(overrides, groups, match) -> None:
    cfg = TrainerConfig(
        specificity_panel_path="panel.parquet",
        specificity_panel_fasta="g.fa",
        group_contrast_target="HL-60",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match=match):
        _build_specificity_panel(cfg, {}, group_contrast_group_names=groups)
