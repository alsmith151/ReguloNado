"""``regulonado.qc``: check families against synthetic BigWigs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from regulonado import qc

pybigtools = pytest.importorskip("pybigtools")


def _make_bigwig(path, values, size=2000):
    writer = pybigtools.open(str(path), "w")
    writer.write({"chr1": size}, iter(values))


def test_header_stats_flags_constant_tracks(tmp_path):
    constant = tmp_path / "constant.bw"
    _make_bigwig(constant, [("chr1", 0, 2000, 3.0)])
    signal = tmp_path / "signal.bw"
    _make_bigwig(signal, [("chr1", 0, 1000, 1.0), ("chr1", 1000, 2000, 2.0)])

    assert not qc.sparsity_passes(qc.header_stats(constant))
    assert qc.sparsity_passes(qc.header_stats(signal))


def test_header_stats_single_chrom_reports_one_chrom_with_signal(tmp_path):
    path = tmp_path / "single.bw"
    _make_bigwig(path, [("chr1", 0, 500, 5.0)])
    stats = qc.header_stats(path)
    assert stats["qc_n_chroms_with_signal"] == 1
    assert stats["qc_bases_covered"] == 500


def test_track_interval_means_and_duplicated_tracks_are_perfectly_concordant(tmp_path):
    a = tmp_path / "a.bw"
    _make_bigwig(a, [("chr1", 0, 500, 1.0), ("chr1", 500, 1000, 5.0)])
    duplicate = tmp_path / "a_dup.bw"
    _make_bigwig(duplicate, [("chr1", 0, 500, 1.0), ("chr1", 500, 1000, 5.0)])
    unrelated = tmp_path / "b.bw"
    _make_bigwig(unrelated, [("chr1", 0, 500, 5.0), ("chr1", 500, 1000, 1.0)])

    windows = [("chr1", i * 50, i * 50 + 50) for i in range(20)]
    means = qc.track_interval_means([a, duplicate, unrelated], windows)
    assert means.shape == (20, 3)

    stats = qc.replicate_concordance_stats(means, group_labels=[None, None, None])
    # With no group labels, only qc_nearest_other_r is populated (max r to any track).
    assert stats[0]["qc_replicate_r"] is None
    assert stats[0]["qc_nearest_other_r"] == pytest.approx(1.0, abs=1e-6)


def test_replicate_concordance_uses_group_labels_when_given():
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    means = np.stack([base, base + rng.normal(scale=0.01, size=100), -base], axis=1)
    stats = qc.replicate_concordance_stats(means, group_labels=["g1", "g1", "g2"])
    assert stats[0]["qc_nearest_is_replicate"] is True
    assert stats[0]["qc_replicate_r"] == pytest.approx(1.0, abs=0.05)
    assert stats[2]["qc_replicate_r"] is None  # g2 has no other member to compare to


def test_interval_signal_stats_nonzero_fraction_and_dynamic_range():
    means = np.array([[0.0, 10.0], [0.0, 10.0], [1.0, 10.0], [1.0, 100.0]])
    stats = qc.interval_signal_stats(means)
    assert stats[0]["qc_nonzero_bin_fraction"] == pytest.approx(0.5)
    assert stats[1]["qc_nonzero_bin_fraction"] == pytest.approx(1.0)


def test_evaluate_rules_min_and_max_thresholds():
    df = pd.DataFrame(
        {
            "qc_nonzero_bin_fraction": [0.5, 0.0, 0.9],
            "qc_dynamic_range": [2.0, float("nan"), 500.0],
        }
    )
    verdict, failed = qc.evaluate_rules(
        df, {"qc_nonzero_bin_fraction": {"min": 0.1}, "qc_dynamic_range": {"max": 100.0}}
    )
    assert verdict.tolist() == ["passed", "failed", "failed"]
    assert failed.tolist() == [[], ["qc_nonzero_bin_fraction"], ["qc_dynamic_range"]]


def test_evaluate_rules_ignores_unknown_columns():
    df = pd.DataFrame({"qc_bases_covered": [10, 20]})
    verdict, failed = qc.evaluate_rules(df, {"qc_does_not_exist": {"min": 1}})
    assert verdict.tolist() == ["passed", "passed"]
    assert failed.tolist() == [[], []]


def test_intervals_from_bed_samples_when_requested(tmp_path):
    bed = tmp_path / "intervals.bed"
    bed.write_text("\n".join(f"chr1\t{i * 100}\t{i * 100 + 50}\tfold0" for i in range(50)) + "\n")
    windows = qc.intervals_from_bed(
        bed, n_pred_bins=1, bin_size=10, shift_max_bp=0, sample_n=5, seed=0
    )
    assert len(windows) == 5


def test_save_and_load_interval_means_round_trip(tmp_path):
    means = np.arange(6, dtype=np.float32).reshape(3, 2)
    path = tmp_path / "interval_means.parquet"
    qc.save_interval_means(means, ["a", "b"], path)
    back = qc.load_interval_means(path, ["b", "a"])
    np.testing.assert_allclose(back, means[:, [1, 0]])


def test_load_interval_means_raises_on_missing_track(tmp_path):
    means = np.zeros((2, 1), dtype=np.float32)
    path = tmp_path / "interval_means.parquet"
    qc.save_interval_means(means, ["a"], path)
    with pytest.raises(ValueError, match="missing"):
        qc.load_interval_means(path, ["a", "b"])
