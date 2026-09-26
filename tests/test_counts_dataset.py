"""``regulonado.counts.dataset``: ``RegionCountData`` parquet round-trip and helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from regulonado.counts.dataset import RegionCountData, group_count_rates


def _toy_data() -> RegionCountData:
    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [0, 1000, 0, 1000],
            "end": [1000, 2000, 1000, 2000],
            "split": ["train", "train", "val", "test"],
        }
    )
    counts = np.array(
        [
            [10.0, 0.0, 5.0],
            [1.0, 1.0, 1.0],
            [100.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    tracks = pl.DataFrame(
        {
            "track_name": ["a", "b", "c"],
            "group": ["hl60", "hl60", "k562"],
            "log_size_factor": [0.0, 0.1, -0.1],
        }
    )
    return RegionCountData.from_arrays(regions, counts, ["a", "b", "c"], tracks)


def test_constructor_validates_row_and_column_alignment():
    regions = pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})
    with pytest.raises(ValueError, match="row-aligned"):
        RegionCountData.from_arrays(regions, np.zeros((2, 1), dtype=np.float32), ["a"])
    with pytest.raises(ValueError, match="track names"):
        RegionCountData.from_arrays(regions, np.zeros((1, 2), dtype=np.float32), ["a"])


def test_constructor_requires_chrom_start_end():
    with pytest.raises(ValueError, match="missing columns"):
        RegionCountData.from_arrays(pl.DataFrame({"chrom": ["chr1"]}), np.zeros((1, 1)), ["a"])


def test_write_read_round_trip(tmp_path):
    data = _toy_data()
    data.write(tmp_path)

    assert (tmp_path / "regions.parquet").exists()
    assert (tmp_path / "counts.parquet").exists()
    assert (tmp_path / "tracks.parquet").exists()

    loaded = RegionCountData.read(tmp_path)
    assert loaded.track_names == data.track_names
    np.testing.assert_array_equal(loaded.counts, data.counts)
    assert loaded.regions.to_dicts() == data.regions.to_dicts()
    assert loaded.tracks.to_dicts() == data.tracks.to_dicts()


def test_read_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RegionCountData.read(tmp_path / "does-not-exist")


def test_split_keeps_only_the_named_split_and_preserves_alignment():
    data = _toy_data()
    train = data.split("train")
    assert train.n_regions == 2
    np.testing.assert_array_equal(train.counts, data.counts[:2])
    assert train.regions["chrom"].to_list() == ["chr1", "chr1"]


def test_split_without_a_split_column_raises():
    data = _toy_data()
    data = RegionCountData(
        regions=data.regions.drop("split"),
        counts=data.counts,
        track_names=data.track_names,
        tracks=data.tracks,
    )
    with pytest.raises(KeyError):
        data.split("train")


def test_take_keeps_the_masked_regions():
    data = _toy_data()
    mask = np.array([True, False, True, False])
    taken = data.take(mask)
    assert taken.n_regions == 2
    np.testing.assert_array_equal(taken.counts, data.counts[[0, 2]])


def test_track_index_and_log_size_factors():
    data = _toy_data()
    assert data.track_index("b") == 1
    with pytest.raises(KeyError):
        data.track_index("nope")
    np.testing.assert_allclose(data.log_size_factors(), [0.0, 0.1, -0.1])


def test_log_size_factors_raises_when_not_finite():
    data = _toy_data()
    tracks = data.tracks.with_columns(pl.Series("log_size_factor", [0.0, float("nan"), -0.1]))
    data = RegionCountData(data.regions, data.counts, data.track_names, tracks)
    with pytest.raises(ValueError, match="log_size_factor"):
        data.log_size_factors()


def test_group_count_rates_pools_replicates_by_group_with_size_factors():
    counts = np.array([[10.0, 20.0], [5.0, np.nan]], dtype=np.float32)
    groups = np.array([0, 0])
    log_size_factors = np.array([0.0, np.log(2.0)])
    rates = group_count_rates(counts, groups, log_size_factors, n_groups=1)
    # Region 0: (10 + 20) / (1 + 2) = 10.
    np.testing.assert_allclose(rates[0, 0], 10.0)
    # Region 1: only track 0 observed -> 5 / 1 = 5.
    np.testing.assert_allclose(rates[1, 0], 5.0)


def test_group_count_rates_is_nan_when_every_replicate_of_a_group_is_masked():
    counts = np.array([[np.nan, np.nan]], dtype=np.float32)
    groups = np.array([0, 0])
    log_size_factors = np.array([0.0, 0.0])
    rates = group_count_rates(counts, groups, log_size_factors, n_groups=1)
    assert np.isnan(rates[0, 0])


def test_count_mask_thresholds_and_mask_counts_flag_artefacts():
    data = _toy_data()
    thresholds = data.count_mask_thresholds(quantile=0.999, factor=20.0)
    assert thresholds.shape == (3,)

    small = RegionCountData.from_arrays(
        pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 100], "end": [100, 200]}),
        np.array([[1.0], [1000.0]], dtype=np.float32),
        ["a"],
    )
    small_thresholds = small.count_mask_thresholds(quantile=0.0, factor=20.0)
    masked = small.mask_counts(small_thresholds)
    assert np.isnan(masked.counts[1, 0])
    assert not np.isnan(masked.counts[0, 0])


def test_gini_is_zero_for_uniform_and_positive_for_specific_regions():
    data = _toy_data()
    gini = data.gini()
    assert gini.shape == (4,)
    # Region 1 ([1, 1, 1]) is uniform across tracks -> Gini ~ 0.
    assert gini[1] == pytest.approx(0.0, abs=1e-6)
    # Region 2 ([100, 0, 0]) is maximally specific -> higher Gini than region 1.
    assert gini[2] > gini[1]


def test_specific_mask_keeps_regions_above_mean_plus_threshold_std():
    data = _toy_data()
    mask = data.specific_mask(gini_std_threshold=0.0)
    gini = data.gini()
    np.testing.assert_array_equal(mask, gini > np.mean(gini))


def test_contrast_is_log1p_signal_minus_row_mean():
    data = _toy_data()
    contrast = data.contrast("a")
    logged = np.log1p(data.counts)
    expected = logged[:, 0] - logged.mean(axis=1)
    np.testing.assert_allclose(contrast, expected)
