from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from regulonado.dataset.build import inverse_transform_signal, transform_signal
from regulonado.normalization import anchor_scale_factors, track_window_stat


class _Reader:
    def values(self, chrom, start, end, *, bins, **kwargs):
        return np.linspace(1, 11, bins, dtype=np.float32)


def test_track_window_stat_uses_rolling_mean_maximum():
    result = track_window_stat(
        _Reader(), [("chr1", 0, 320)], bin_size=32, window_stat_bp=64
    )
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [10.444444])


def test_anchor_background_transform_round_trip():
    signal = np.array([[1.0, 11.0, 21.0]], dtype=np.float32)
    transformed = transform_signal(
        signal,
        np.array([0.1]),
        10.0,
        20.0,
        background=np.array([1.0]),
        apply_squash=False,
        apply_clip=False,
    )
    np.testing.assert_allclose(transformed, [[0.0, 1.0, 2.0]])
    recovered = inverse_transform_signal(
        transformed,
        np.array([0.1]),
        background=np.array([1.0]),
        apply_squash=False,
    )
    np.testing.assert_allclose(recovered, signal)


def _write_bigwig(path: Path, values: list[tuple[str, int, int, float]], size: int = 3000) -> None:
    pybigtools = pytest.importorskip("pybigtools")
    writer = pybigtools.open(str(path), "w")
    writer.write({"chr1": size}, iter(sorted(values, key=lambda v: v[1])))


def _write_bed(path: Path, rows: list[tuple[int, int]]) -> None:
    path.write_text("\n".join(f"chr1\t{start}\t{end}" for start, end in rows) + "\n")


def test_anchor_quality_bands_informative_and_failed(tmp_path):
    """The quality banding this refactor's QC gate is built around."""
    # Informative: high at anchor+heldout, low at background -> recovery=1.0.
    informative = tmp_path / "informative.bw"
    _write_bigwig(
        informative,
        [("chr1", 0, 500, 20.0), ("chr1", 500, 1000, 1.0), ("chr1", 1000, 1500, 20.0)],
    )
    # Failed: uniformly low everywhere, so anchor_reference <= background_q50.
    failed = tmp_path / "failed.bw"
    _write_bigwig(failed, [("chr1", 0, 1500, 1.0)])

    anchor_bed, background_bed, heldout_bed = (
        tmp_path / "anchor.bed",
        tmp_path / "background.bed",
        tmp_path / "heldout.bed",
    )
    _write_bed(anchor_bed, [(0, 500)])
    _write_bed(background_bed, [(500, 1000)])
    _write_bed(heldout_bed, [(1000, 1500)])

    df = anchor_scale_factors(
        [informative, failed],
        anchor_bed,
        background_bed,
        heldout_regions=heldout_bed,
        bin_size=10,
        window_stat_bp=10,
        background_sample=None,
    )
    by_index = df.set_index("track_index")
    assert by_index.loc[0, "quality"] == "informative"
    assert by_index.loc[1, "quality"] == "failed"
    # A failed track is written as-is (not forced to neutral values elsewhere) —
    # anchor_scale_factors's own fallback already neutralises it to scale_factor=1.
    assert by_index.loc[1, "scale_factor"] == pytest.approx(1.0)


def test_anchor_quality_band_weak_for_partial_heldout_recovery(tmp_path):
    """recovery in [0.30, 0.70) bands as 'weak' rather than informative/failed."""
    background_rows = [(i * 100, i * 100 + 100) for i in range(10)]
    background_vals = [("chr1", s, e, float(i)) for i, (s, e) in enumerate(background_rows)]
    anchor_vals = [("chr1", 1000, 1100, 20.0)]
    # 2/5 heldout windows exceed background_q99 -> recovery = 0.4.
    heldout_rows = [(1100, 1200), (1200, 1300), (1300, 1400), (1400, 1500), (1500, 1600)]
    heldout_vals = [
        ("chr1", 1100, 1200, 15.0),
        ("chr1", 1200, 1300, 15.0),
        ("chr1", 1300, 1400, 1.0),
        ("chr1", 1400, 1500, 1.0),
        ("chr1", 1500, 1600, 1.0),
    ]
    track = tmp_path / "weak.bw"
    _write_bigwig(track, background_vals + anchor_vals + heldout_vals)

    background_bed, anchor_bed, heldout_bed = (
        tmp_path / "background.bed",
        tmp_path / "anchor.bed",
        tmp_path / "heldout.bed",
    )
    _write_bed(background_bed, background_rows)
    _write_bed(anchor_bed, [(1000, 1100)])
    _write_bed(heldout_bed, heldout_rows)

    df = anchor_scale_factors(
        [track],
        anchor_bed,
        background_bed,
        heldout_regions=heldout_bed,
        bin_size=10,
        window_stat_bp=10,
        background_sample=None,
    )
    assert df.loc[0, "quality"] == "weak"
    assert df.loc[0, "heldout_recovery"] == pytest.approx(0.4)
