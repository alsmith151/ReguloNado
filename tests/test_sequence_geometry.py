"""Tests for the ``crop_bp`` threading fix (batch B): ``design/sequence.py`` and
``genomics.window_for_interval`` derive the predicted crop centred in the context by default,
which silently disagrees with the true crop of a ``TrunkWindow``-based region-count model. These
tests pin the derived-vs-true disagreement and confirm the new ``crop_bp`` override reproduces
the default exactly when omitted.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import write_bed as _write_bed
from regulonado.design.sequence import DatasetWindowIndex, context_bp_to_pred_bins
from regulonado.genomics import window_for_interval

# Tiny geometry for fast tests, matching test_design.py's convention.
N_PRED_BINS = 4
BIN_SIZE = 10
CONTEXT = 100


# --------------------------------------------------------------------------- #
# 1. The pin test — AlphaGenome-encoder geometry from the bug report          #
# --------------------------------------------------------------------------- #
def test_derived_and_true_crop_disagree_by_64bp():
    context_length = 4096
    n_pred_bins = 1
    bin_size = 1152

    derived_crop = (context_length - n_pred_bins * bin_size) // 2
    true_crop = 1408

    assert derived_crop == 1472
    assert true_crop == 1408
    assert derived_crop - true_crop == 64

    # The true crop maps context span [1408, 2560) onto bin 0...
    true_span = context_bp_to_pred_bins(
        1408,
        2560,
        context_length=context_length,
        n_pred_bins=n_pred_bins,
        bin_size=bin_size,
        crop_bp=true_crop,
    )
    assert true_span == slice(0, 1)

    # ...but the derived (default) crop does not agree: that same span falls outside
    # [1472, 2624), the crop the un-overridden formula claims.
    with pytest.raises(ValueError, match="outside the predicted crop"):
        context_bp_to_pred_bins(
            1408, 2560, context_length=context_length, n_pred_bins=n_pred_bins, bin_size=bin_size
        )


# --------------------------------------------------------------------------- #
# 2. crop_bp=None reproduces current behaviour                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "start,end,expected",
    [
        (30, 40, slice(0, 1)),
        (30, 70, slice(0, 4)),
        (40, 50, slice(1, 2)),
        (60, 70, slice(3, 4)),
    ],
)
def test_context_bp_to_pred_bins_none_matches_derived_default(start, end, expected):
    kwargs = dict(context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE)
    assert context_bp_to_pred_bins(start, end, **kwargs) == expected
    assert context_bp_to_pred_bins(start, end, crop_bp=None, **kwargs) == expected


# --------------------------------------------------------------------------- #
# 3. window_for_interval with and without crop_bp                            #
# --------------------------------------------------------------------------- #
def test_window_for_interval_crop_bp_none_matches_original_centring():
    kwargs = dict(context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE)
    window_default = window_for_interval("chr1", 500, 540, **kwargs)
    window_explicit_none = window_for_interval("chr1", 500, 540, crop_bp=None, **kwargs)
    assert window_default == window_explicit_none
    # center=520, pred_start=500, ctx_start=520-50=470 (original centred-on-midpoint rule).
    assert (window_default.pred_start, window_default.ctx_start) == (500, 470)


def test_window_for_interval_crop_bp_overrides_context_start():
    kwargs = dict(context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE)
    window = window_for_interval("chr1", 500, 540, crop_bp=25, **kwargs)
    # pred_start is unchanged (500); ctx_start is now pred_start - crop_bp, not centred.
    assert window.pred_start == 500
    assert window.ctx_start == 500 - 25
    assert window.ctx_end == window.ctx_start + CONTEXT


# --------------------------------------------------------------------------- #
# 3b. snap_bp — TrunkWindow's floor-snapped pred_start                       #
# --------------------------------------------------------------------------- #
def test_window_for_interval_snap_bp_none_matches_original_centring():
    kwargs = dict(context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE)
    window_default = window_for_interval("chr1", 500, 540, **kwargs)
    window_explicit_none = window_for_interval("chr1", 500, 540, snap_bp=None, **kwargs)
    assert window_default == window_explicit_none
    assert window_default.pred_start == 500  # centre-derived, unchanged


# TrunkWindow.window_start, worked example from the bug report: input_length=4096,
# target_width=1000, output_bin_size=128 -> k=9, pool_factor=1, first_bin=11,
# crop_bp = first_bin * bin_size = 1408, effective (snap_bp) = 128.
_TRUNK_WINDOW_CASES = [
    # (target_start, expected snapped pred_start)
    (1408, 1408),
    (1500, 1408),
    (1600, 1536),
    (2048, 2048),
    (3000, 2944),
    (12345, 12288),
]


@pytest.mark.parametrize("target_start,snapped", _TRUNK_WINDOW_CASES)
def test_window_for_interval_snap_bp_matches_trunk_window(target_start, snapped):
    crop_bp = 1408
    snap_bp = 128
    # end is irrelevant to pred_start once snap_bp is given (it snaps `start`, not `center`) —
    # pick a target_width of 1000bp, matching the worked example, purely for realism.
    window = window_for_interval(
        "chr1",
        target_start,
        target_start + 1000,
        context_length=4096,
        n_pred_bins=1,
        bin_size=1152,
        crop_bp=crop_bp,
        snap_bp=snap_bp,
    )
    assert window.pred_start == (target_start // snap_bp) * snap_bp
    assert window.pred_start == snapped
    assert window.ctx_start == window.pred_start - crop_bp


@pytest.mark.parametrize("target_start,snapped", _TRUNK_WINDOW_CASES)
def test_snapped_window_always_contains_the_target(target_start, snapped):
    """A future target_width change that breaks containment should fail loudly here."""
    target_width = 1000
    scored_span_width = 1152  # n_pred_bins * bin_size in the worked example
    assert snapped <= target_start
    assert target_start + target_width <= snapped + scored_span_width


# --------------------------------------------------------------------------- #
# 4. DatasetWindowIndex.from_region_table                                    #
# --------------------------------------------------------------------------- #
def test_from_region_table_parquet_matches_equivalent_bed(tmp_path):
    bed_path = _write_bed(
        tmp_path / "intervals.bed",
        [
            ("chr1", 500, 540, "train"),
            ("chr1", 600, 650, "valid"),
        ],
    )
    bed_index = DatasetWindowIndex.from_bed(
        bed_path, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
    )

    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [490, 590],  # deliberately different from target_start/end
            "end": [550, 660],
            "target_start": [500, 600],
            "target_end": [540, 650],
            "split": ["train", "valid"],
        }
    )
    parquet_path = tmp_path / "regions.parquet"
    regions.write_parquet(parquet_path)

    region_index = DatasetWindowIndex.from_region_table(
        parquet_path, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
    )

    bed_windows = [
        (iw.window.pred_start, iw.window.pred_end, iw.window.ctx_start, iw.fold_label)
        for windows in bed_index._by_chrom.values()
        for iw in windows
    ]
    region_windows = [
        (iw.window.pred_start, iw.window.pred_end, iw.window.ctx_start, iw.fold_label)
        for windows in region_index._by_chrom.values()
        for iw in windows
    ]
    assert bed_windows == region_windows


def test_from_region_table_falls_back_to_bed_for_non_parquet(tmp_path):
    bed_path = _write_bed(
        tmp_path / "intervals.bed",
        [("chr1", 500, 540, "train")],
    )
    index = DatasetWindowIndex.from_region_table(
        bed_path, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
    )
    windows = index._by_chrom["chr1"]
    assert len(windows) == 1
    assert (windows[0].window.pred_start, windows[0].window.pred_end) == (500, 540)
    assert windows[0].fold_label == "train"


def test_from_region_table_prefers_target_span_and_threads_crop_bp(tmp_path):
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [1000],
            "target_start": [500],
            "target_end": [540],
            "split": ["test"],
        }
    )
    parquet_path = tmp_path / "regions.parquet"
    regions.write_parquet(parquet_path)

    index = DatasetWindowIndex.from_region_table(
        parquet_path,
        context_length=CONTEXT,
        n_pred_bins=N_PRED_BINS,
        bin_size=BIN_SIZE,
        crop_bp=25,
    )
    iw = index._by_chrom["chr1"][0]
    # Predicted region is derived from target_start/target_end (500,540), not start/end
    # (0,1000); ctx_start honours the explicit crop_bp rather than centring.
    assert (iw.window.pred_start, iw.window.pred_end) == (500, 540)
    assert iw.window.ctx_start == 500 - 25
    assert iw.fold_label == "test"
    assert index.crop_bp == 25
