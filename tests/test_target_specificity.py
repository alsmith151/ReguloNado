from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import write_bigwig
from regulonado.normalization import track_window_stat
from regulonado.target_specificity import (
    group_index_from_records,
    group_mean,
    region_statistic,
    rolling_window_stat,
    target_specificity_score,
    to_anchor_units,
)

# ---------------------------------------------------------------------------
# rolling_window_stat / region_statistic
# ---------------------------------------------------------------------------


def test_rolling_window_stat_matches_track_window_stat_fixture():
    """Same fixture as test_normalization_anchor.py: linspace(1, 11, 10), bin 32, window 64."""
    values = np.linspace(1, 11, 10, dtype=np.float32)
    width = max(1, round(64 / 32))
    assert width == 2
    result = rolling_window_stat(values, width)
    np.testing.assert_allclose(result, 10.444444, rtol=1e-5)

    class _Reader:
        def values(self, chrom, start, end, *, bins, **kwargs):
            return np.linspace(1, 11, bins, dtype=np.float32)

    reference = track_window_stat(_Reader(), [("chr1", 0, 320)], bin_size=32, window_stat_bp=64)
    np.testing.assert_allclose(result, reference[0])


def test_region_statistic_ragged_regions_not_diluted_by_padding():
    """A short region's statistic must not be pulled toward the padding value (0)."""
    # Region 0: 4 valid bins, all 10.0; padded to 6 bins with zeros.
    # Region 1: 6 valid bins, all 10.0 (no padding).
    profiles = np.zeros((2, 1, 6), dtype=np.float32)
    profiles[0, 0, :4] = 10.0
    profiles[1, 0, :6] = 10.0
    lengths = np.array([4, 6])

    result = region_statistic(profiles, lengths, bin_size=1, window_stat_bp=1, stat="mean")
    np.testing.assert_allclose(result[:, 0], [10.0, 10.0])

    result_rolling = region_statistic(
        profiles, lengths, bin_size=1, window_stat_bp=2, stat="rolling-max"
    )
    np.testing.assert_allclose(result_rolling[:, 0], [10.0, 10.0])


def test_region_statistic_rolling_max_uses_valid_bins_only():
    # 5 valid bins [1, 2, 3, 20, 1], 2 bins of zero padding.
    profiles = np.zeros((1, 1, 7), dtype=np.float32)
    profiles[0, 0, :5] = [1.0, 2.0, 3.0, 20.0, 1.0]
    lengths = np.array([5])
    # width = round(2/1) = 2 -> rolling means: 1.5, 2.5, 11.5, 10.5 -> max 11.5
    result = region_statistic(profiles, lengths, bin_size=1, window_stat_bp=2, stat="rolling-max")
    np.testing.assert_allclose(result[0, 0], 11.5)


# ---------------------------------------------------------------------------
# group_index_from_records
# ---------------------------------------------------------------------------


def test_group_index_from_records_basic_and_filters():
    records = [
        {"group": "HL-60", "assay_class": "chip", "quality": "informative"},
        {"group": "HL-60", "assay_class": "chip", "quality": "informative"},
        {"group": "Jurkat", "assay_class": "atac", "quality": "informative"},
        {"group": "K562", "assay_class": "chip", "quality": "weak"},
        {"group": None, "assay_class": "chip", "quality": "informative"},
    ]
    group_index, group_names = group_index_from_records(records)
    assert group_names == ["HL-60", "Jurkat", "K562"]
    assert group_index.tolist() == [0, 0, 1, 2, -1]

    group_index, group_names = group_index_from_records(records, assay_classes={"chip"})
    assert group_names == ["HL-60", "K562"]
    assert group_index.tolist() == [0, 0, -1, 1, -1]

    group_index, group_names = group_index_from_records(records, quality={"informative"})
    assert group_names == ["HL-60", "Jurkat"]
    assert group_index.tolist() == [0, 0, 1, -1, -1]


# ---------------------------------------------------------------------------
# to_anchor_units / group_mean
# ---------------------------------------------------------------------------


def test_group_mean_averages_linear_and_ignores_excluded():
    signal = np.array(
        [
            [1.0, 3.0, 100.0],  # region 0: tracks 0,1 in group 0; track 2 excluded
            [2.0, 4.0, 200.0],  # region 1
        ]
    )
    group_index = np.array([0, 0, -1])
    result = group_mean(signal, group_index, n_groups=1)
    np.testing.assert_allclose(result[:, 0], [2.0, 3.0])


def test_to_anchor_units_matches_transform_signal_round_trip():
    signal = np.array([[1.0, 11.0, 21.0]])  # (R=1, T=3) with T being a single track dim
    result = to_anchor_units(
        signal, scale_factor=np.array([0.1]), background=np.array([1.0])
    )
    np.testing.assert_allclose(result, [[0.0, 1.0, 2.0]])


# ---------------------------------------------------------------------------
# target_specificity_score
# ---------------------------------------------------------------------------


def _make_signal(target_values, reference_values_by_group):
    """Build a (R=1, T) stored-signal row with scale_factor=1, background=0 (identity anchor)."""
    values = list(target_values)
    group_index = [0] * len(target_values)
    names = ["target"]
    for values_g in reference_values_by_group:
        group_index += [len(names)] * len(values_g)
        names.append(f"ref{len(names)}")
        values += list(values_g)
    signal = np.array([values], dtype=np.float64)
    return signal, np.array(group_index), names


def test_score_invariant_to_duplicating_a_reference_track():
    """Group quantile/top-k must operate on group means, not raw per-track values."""
    signal, group_index, names = _make_signal([10.0, 10.0], [[4.0], [2.0], [8.0]])
    scale = np.ones(signal.shape[1])
    background = np.zeros(signal.shape[1])

    base = target_specificity_score(
        signal,
        scale_factor=scale,
        background=background,
        group_index=group_index,
        group_names=names,
        target_group="target",
        pseudocount=0.1,
        reduction="quantile",
        quantile=0.9,
    )

    # Duplicate the "ref1" track (value 4.0) as another member of the same group: the
    # group mean for ref1 stays 4.0, so the score must be unchanged.
    signal2 = np.array([[10.0, 10.0, 4.0, 4.0, 2.0, 8.0]])
    group_index2 = np.array([0, 0, 1, 1, 2, 3])
    scale2 = np.ones(signal2.shape[1])
    background2 = np.zeros(signal2.shape[1])
    dup = target_specificity_score(
        signal2,
        scale_factor=scale2,
        background=background2,
        group_index=group_index2,
        group_names=names,
        target_group="target",
        pseudocount=0.1,
        reduction="quantile",
        quantile=0.9,
    )
    np.testing.assert_allclose(dup.score, base.score)
    np.testing.assert_allclose(dup.group_signal, base.group_signal)


def test_replicate_averaging_within_group():
    signal, group_index, names = _make_signal([6.0, 10.0], [[4.0]])
    result = target_specificity_score(
        signal,
        scale_factor=np.ones(signal.shape[1]),
        background=np.zeros(signal.shape[1]),
        group_index=group_index,
        group_names=names,
        target_group="target",
    )
    np.testing.assert_allclose(result.group_signal[0, 0], 8.0)  # mean(6, 10)


def test_quantile_one_equals_target_minus_max_reference():
    signal, group_index, names = _make_signal([10.0], [[4.0], [2.0], [8.0]])
    scale = np.ones(signal.shape[1])
    background = np.zeros(signal.shape[1])
    c = 0.1

    result_q1 = target_specificity_score(
        signal,
        scale_factor=scale,
        background=background,
        group_index=group_index,
        group_names=names,
        target_group="target",
        pseudocount=c,
        reduction="quantile",
        quantile=1.0,
    )
    expected = math.log2(10.0 + c) - math.log2(8.0 + c)
    np.testing.assert_allclose(result_q1.score, [expected])

    result_topk1 = target_specificity_score(
        signal,
        scale_factor=scale,
        background=background,
        group_index=group_index,
        group_names=names,
        target_group="target",
        pseudocount=c,
        reduction="top-k",
        top_k=1,
    )
    np.testing.assert_allclose(result_topk1.score, result_q1.score)


def test_quantile_commutes_with_log2_monotone_guard():
    """Reordering reference values must not change the quantile-reduced score (refactor guard)."""
    signal, group_index, names = _make_signal([10.0], [[1.0], [9.0], [3.0], [7.0], [5.0]])
    result = target_specificity_score(
        signal,
        scale_factor=np.ones(signal.shape[1]),
        background=np.zeros(signal.shape[1]),
        group_index=group_index,
        group_names=names,
        target_group="target",
        pseudocount=0.1,
        reduction="quantile",
        quantile=0.9,
    )
    reference_logs = np.log2(np.array([1.0, 9.0, 3.0, 7.0, 5.0]) + 0.1)
    expected_reference_log = np.quantile(reference_logs, 0.9, method="linear")
    np.testing.assert_allclose(result.reference_log, [expected_reference_log])
    np.testing.assert_allclose(
        result.score, [math.log2(10.0 + 0.1) - expected_reference_log]
    )


def test_adding_silent_group_changes_q90_but_not_q1():
    signal, group_index, names = _make_signal([10.0], [[4.0], [2.0], [8.0]])
    scale = np.ones(signal.shape[1])
    background = np.zeros(signal.shape[1])

    before_q90 = target_specificity_score(
        signal, scale_factor=scale, background=background, group_index=group_index,
        group_names=names, target_group="target", reduction="quantile", quantile=0.9,
    )
    before_q1 = target_specificity_score(
        signal, scale_factor=scale, background=background, group_index=group_index,
        group_names=names, target_group="target", reduction="quantile", quantile=1.0,
    )

    signal2, group_index2, names2 = _make_signal(
        [10.0], [[4.0], [2.0], [8.0], [0.0]]  # extra silent group
    )
    after_q90 = target_specificity_score(
        signal2, scale_factor=np.ones(signal2.shape[1]), background=np.zeros(signal2.shape[1]),
        group_index=group_index2, group_names=names2, target_group="target",
        reduction="quantile", quantile=0.9,
    )
    after_q1 = target_specificity_score(
        signal2, scale_factor=np.ones(signal2.shape[1]), background=np.zeros(signal2.shape[1]),
        group_index=group_index2, group_names=names2, target_group="target",
        reduction="quantile", quantile=1.0,
    )

    assert not np.allclose(after_q90.score, before_q90.score)
    np.testing.assert_allclose(after_q1.score, before_q1.score)


def test_background_modes_agree_above_background_differ_below():
    # Signal is orders of magnitude above the background floor, so subtracting a small,
    # constant background is a negligible perturbation in log space: the two modes should
    # nearly agree (not exactly -- subtracting a constant is never exactly a no-op in log
    # space), unlike the "differ below" case checked next.
    signal, group_index, names = _make_signal([1000.0], [[500.0]])
    scale = np.ones(signal.shape[1])
    background = np.full(signal.shape[1], 1.0)

    above_subtract = target_specificity_score(
        signal, scale_factor=scale, background=background, group_index=group_index,
        group_names=names, target_group="target", background_mode="subtract",
    )
    above_scale_only = target_specificity_score(
        signal, scale_factor=scale, background=background, group_index=group_index,
        group_names=names, target_group="target", background_mode="scale-only",
    )
    np.testing.assert_allclose(above_subtract.score, above_scale_only.score, atol=0.01)

    # Track below background: subtract floors at 0 -> log2(c); scale-only does not.
    signal_below, group_index_b, names_b = _make_signal([2.0], [[10.0]])
    background_b = np.full(signal_below.shape[1], 5.0)
    c = 0.1
    below_subtract = target_specificity_score(
        signal_below, scale_factor=np.ones(signal_below.shape[1]), background=background_b,
        group_index=group_index_b, group_names=names_b, target_group="target",
        background_mode="subtract", pseudocount=c,
    )
    below_scale_only = target_specificity_score(
        signal_below, scale_factor=np.ones(signal_below.shape[1]), background=background_b,
        group_index=group_index_b, group_names=names_b, target_group="target",
        background_mode="scale-only", pseudocount=c,
    )
    np.testing.assert_allclose(below_subtract.target_log, [math.log2(c)])
    assert below_scale_only.target_log[0] > below_subtract.target_log[0]
    assert not np.allclose(below_subtract.score, below_scale_only.score)


def test_target_excluded_from_its_own_reference_set():
    signal, group_index, names = _make_signal([10.0], [[4.0], [2.0]])
    result = target_specificity_score(
        signal,
        scale_factor=np.ones(signal.shape[1]),
        background=np.zeros(signal.shape[1]),
        group_index=group_index,
        group_names=names,
        target_group="target",
        reduction="quantile",
        quantile=1.0,
    )
    # If target were included in its own reference set, q=1.0 (max) would be the target
    # itself, giving score == 0. It is not.
    assert result.score[0] != 0.0
    np.testing.assert_allclose(result.reference_log, [math.log2(4.0 + 0.1)])


def test_target_group_absent_raises():
    signal, group_index, names = _make_signal([10.0], [[4.0]])
    with pytest.raises(ValueError):
        target_specificity_score(
            signal,
            scale_factor=np.ones(signal.shape[1]),
            background=np.zeros(signal.shape[1]),
            group_index=group_index,
            group_names=names,
            target_group="not-a-group",
        )


def test_target_rank_and_strongest_reference():
    # target=10 (highest), ref groups 4, 20, 8 -> target_rank should be 2 (ref2=20 highest)
    signal, group_index, names = _make_signal([10.0], [[4.0], [20.0], [8.0]])
    result = target_specificity_score(
        signal,
        scale_factor=np.ones(signal.shape[1]),
        background=np.zeros(signal.shape[1]),
        group_index=group_index,
        group_names=names,
        target_group="target",
    )
    assert result.target_rank[0] == 2
    assert names[int(result.strongest_reference[0])] == "ref2"


# ---------------------------------------------------------------------------
# End-to-end on two synthetic bigwigs
# ---------------------------------------------------------------------------


def test_end_to_end_on_synthetic_bigwigs(tmp_path):
    pybigtools = pytest.importorskip("pybigtools")

    # Two tracks: "hl60" strongly enriched in the region, "jurkat" flat/low.
    hl60_path = tmp_path / "hl60.bw"
    jurkat_path = tmp_path / "jurkat.bw"
    write_bigwig(hl60_path, [("chr1", 0, 320, 1.0), ("chr1", 320, 640, 20.0)])
    write_bigwig(jurkat_path, [("chr1", 0, 640, 1.0)])

    chrom, start, end = "chr1", 0, 640
    bin_size = 32
    n_bins = (end - start) // bin_size

    reader_hl60 = pybigtools.open(str(hl60_path))
    reader_jurkat = pybigtools.open(str(jurkat_path))
    hl60_values = np.asarray(
        reader_hl60.values(chrom, start, end, bins=n_bins, summary="mean", exact=True, missing=0),
        dtype=np.float32,
    )
    jurkat_values = np.asarray(
        reader_jurkat.values(
            chrom, start, end, bins=n_bins, summary="mean", exact=True, missing=0
        ),
        dtype=np.float32,
    )

    profiles = np.stack([hl60_values, jurkat_values])[np.newaxis, :, :]  # (R=1, T=2, B)
    lengths = np.array([n_bins])

    stats = region_statistic(profiles, lengths, bin_size=bin_size, window_stat_bp=64)
    assert stats.shape == (1, 2)
    assert stats[0, 0] > stats[0, 1]  # hl60 statistic higher than jurkat's

    group_index = np.array([0, 1])
    group_names = ["hl60", "jurkat"]
    result = target_specificity_score(
        stats,
        scale_factor=np.ones(2),
        background=np.zeros(2),
        group_index=group_index,
        group_names=group_names,
        target_group="hl60",
        reduction="quantile",
        quantile=1.0,
    )
    assert result.score[0] > 0.0
    assert result.target_rank[0] == 1
