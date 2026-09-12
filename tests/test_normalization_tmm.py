"""Golden-value tests for TMM normalisation (audit F05 / R3 / F19-partial).

``compute_tmm_factors`` hand-reimplements edgeR::calcNormFactors(method="TMM")
and, before this test file, had no test coverage at all (audit F20). The
fixtures in ``tests/data/tmm/`` (see the README there for exact provenance)
give an independently-computed edgeR ground truth; ``_tmm_from_counts`` (the
pure counts -> factors kernel factored out of ``compute_tmm_factors``) is
checked against it directly.

edgeR is a fixture-generation tool only -- it is never imported here and is
not a runtime or test dependency of this project.

rnanorm (audit R3's suggested replacement) was evaluated and rejected: its
``TMM`` always derives each sample's library size as ``colSums`` of the input
count matrix and has no way to accept an externally supplied library size.
regulonado's usage is deliberately decoupled -- the M/A comparison runs over a
curated region subset (``means``/``counts``) while scaling uses each track's
true whole-library mapped-read count (``library_sizes``), equivalent to
calling edgeR with an explicit ``lib.size=`` that differs from
``colSums(counts)``. Swapping in rnanorm would silently substitute
``colSums(counts)`` for the caller-supplied ``library_sizes`` and, on the
fixture in this file, produces factors up to 0.40 away (in factor units) from
the correct edgeR-with-explicit-lib.size answer -- worse than any of the bugs
this test file was written to catch. See ``tests/data/tmm/README.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from regulonado.normalization import _tmm_from_counts, compute_tmm_factors

FIXTURE_DIR = Path(__file__).parent / "data" / "tmm"


def _load_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = pd.read_csv(FIXTURE_DIR / "counts.csv").to_numpy(dtype=np.float64)
    library_sizes = pd.read_csv(FIXTURE_DIR / "library_sizes.csv")["library_size"].to_numpy(
        dtype=np.float64
    )
    edger_factors = pd.read_csv(FIXTURE_DIR / "edger_norm_factors.csv")["tmm_factor"].to_numpy(
        dtype=np.float64
    )
    return counts, library_sizes, edger_factors


def test_tmm_from_counts_matches_edger_ground_truth():
    """The counts->factors kernel matches edgeR::calcNormFactors to ~1e-9.

    The fixture's library_sizes are deliberately NOT proportional to
    counts.csv's column sums, mirroring calling edgeR with an explicit
    lib.size (see tests/data/tmm/README.txt) rather than the default
    lib.size=colSums(counts).
    """
    counts, library_sizes, edger_factors = _load_fixture()
    factors = _tmm_from_counts(counts, library_sizes)
    np.testing.assert_allclose(factors, edger_factors, atol=1e-8, rtol=1e-8)


def test_compute_tmm_factors_end_to_end_matches_edger_ground_truth():
    """The public RPKM-based API reproduces the same edgeR ground truth.

    means is derived by inverting compute_tmm_factors's own pseudo-count
    formula (count = mean_RPKM * region_kb * lib/1e6) so that, up to floating
    point, compute_tmm_factors reconstructs the fixture's counts internally.
    """
    counts, library_sizes, edger_factors = _load_fixture()
    region_length_kb = 200.0
    means = counts / (region_length_kb * library_sizes[np.newaxis, :] / 1e6)

    factors = compute_tmm_factors(means, library_sizes, region_length_kb)
    np.testing.assert_allclose(factors, edger_factors, atol=1e-6, rtol=1e-6)


def test_tmm_factors_have_unit_geometric_mean():
    counts, library_sizes, _ = _load_fixture()
    factors = _tmm_from_counts(counts, library_sizes)
    geometric_mean = np.exp(np.mean(np.log(factors)))
    assert geometric_mean == pytest.approx(1.0, abs=1e-10)


def test_compute_tmm_factors_pinned_on_tiny_synthetic_input():
    """Regression pin: a tiny deterministic input's exact output.

    Independent of the edgeR fixture -- this exists so a future accidental
    change to the kernel is caught even if tests/data/tmm/ is ever
    regenerated. Values were computed once from the current implementation
    (itself verified against edgeR by the tests above) and hardcoded here.
    """
    rng = np.random.default_rng(7)
    n_regions = 40
    region_length_kb = 100.0
    library_sizes = np.array([5_000_000.0, 6_000_000.0, 4_000_000.0])

    baseline = rng.gamma(shape=2.0, scale=50.0, size=n_regions)
    sample_scale = np.array([1.0, 1.3, 0.8])
    counts = baseline[:, None] * sample_scale[None, :]
    counts = rng.poisson(counts).astype(float)
    counts[rng.random(counts.shape) < 0.1] = 0.0
    means = counts / (region_length_kb * library_sizes[np.newaxis, :] / 1e6)

    factors = compute_tmm_factors(means, library_sizes, region_length_kb)

    assert factors.shape == (3,)
    np.testing.assert_allclose(
        factors,
        [0.9737384835473567, 1.0879042226029538, 0.9439891513781027],
        atol=1e-9,
        rtol=1e-9,
    )
    geometric_mean = np.exp(np.mean(np.log(factors)))
    assert geometric_mean == pytest.approx(1.0, abs=1e-10)


def test_compute_tmm_factors_handles_zero_regions_without_nan():
    """Regions with zero counts in every track must not produce NaN/inf factors."""
    rng = np.random.default_rng(3)
    n_regions, n_tracks = 30, 2
    region_length_kb = 50.0
    library_sizes = np.array([2_000_000.0, 3_000_000.0])

    counts = rng.poisson(20.0, size=(n_regions, n_tracks)).astype(float)
    counts[:5, :] = 0.0  # a block of fully-zero regions
    means = counts / (region_length_kb * library_sizes[np.newaxis, :] / 1e6)

    factors = compute_tmm_factors(means, library_sizes, region_length_kb)
    assert np.all(np.isfinite(factors))
    assert factors.shape == (2,)
