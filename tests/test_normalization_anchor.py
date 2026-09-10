from __future__ import annotations

import numpy as np
from regulonado.dataset.build import inverse_transform_signal, transform_signal
from regulonado.normalization import track_window_stat


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
